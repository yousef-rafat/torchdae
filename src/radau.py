import math
import torch
import torch.func as func
from typing import Callable, Optional, Tuple, List
from functools import partial

from common import batched_newton_solve, StatefulJacobian, try_compile, DAESolution
from util import handle_step_events
from algorithms import solve_consistent_yp0
from bdf import prepare_solver_inputs, apply_event_reset

__all__ = ["solve_radau_iia5"]


def radau_iia5_step(
    F: Callable, 
    t_n: float, 
    h: float, 
    y_n: torch.Tensor,
    tol: float = 1e-8, 
    max_iter: int = 20, 
    damping: float = 1.0,
    strategy: str = "freeze_dynamic",
    recompute_every: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    3-stage, 5th-order fully Implicit Runge-Kutta Radau IIA step.
    y_n: shape (B, *state_shape)
    """
    # Gauss-Radau quadrature coefficients (Radau IIA, s=3)
    sq6 = math.sqrt(6.0)
    c1 = (4.0 - sq6) / 10.0
    c2 = (4.0 + sq6) / 10.0
    c3 = 1.0
    
    a11 = (88.0 - 7.0 * sq6) / 360.0
    a12 = (296.0 - 169.0 * sq6) / 1800.0
    a13 = (-2.0 + 3.0 * sq6) / 225.0
    
    a21 = (296.0 + 169.0 * sq6) / 1800.0
    a22 = (88.0 + 7.0 * sq6) / 360.0
    a23 = (-2.0 - 3.0 * sq6) / 225.0
    
    a31 = (16.0 - sq6) / 36.0
    a32 = (16.0 + sq6) / 36.0
    a33 = 1.0 / 9.0
    
    state_shape = y_n.shape[1:]
    ndof = y_n[0].numel()
    y_n_flat = y_n.flatten(start_dim=1)
    batch_size = y_n.shape[0]
    
    # Stack the three stage derivatives Y'_1, Y'_2, Y'_3 into a single vector of shape (3 * ndof,)
    def G_single(x_single: torch.Tensor, y_n_single: torch.Tensor):
        Yp_1 = x_single[:ndof]
        Yp_2 = x_single[ndof:2*ndof]
        Yp_3 = x_single[2*ndof:]
        
        # Compute stage values Y_1, Y_2, Y_3 using the Butcher matrix coefficients
        Y_1 = y_n_single + h * (a11 * Yp_1 + a12 * Yp_2 + a13 * Yp_3)
        Y_2 = y_n_single + h * (a21 * Yp_1 + a22 * Yp_2 + a23 * Yp_3)
        Y_3 = y_n_single + h * (a31 * Yp_1 + a32 * Yp_2 + a33 * Yp_3)
        
        # Evaluate residuals of the algebraic system at each of the three stages
        res1 = F(t_n + c1 * h, Y_1.view(state_shape), Yp_1.view(state_shape)).flatten()
        res2 = F(t_n + c2 * h, Y_2.view(state_shape), Yp_2.view(state_shape)).flatten()
        res3 = F(t_n + c3 * h, Y_3.view(state_shape), Yp_3.view(state_shape)).flatten()
        
        return torch.cat([res1, res2, res3])

    batched_residual = lambda x: func.vmap(G_single)(x, y_n_flat)  # noqa: E731
    batched_jacobian = lambda x: func.vmap(func.jacrev(G_single, argnums=0))(x, y_n_flat)  # noqa: E731
    
    # Initial guess: flat zeros for stage derivatives
    x0 = torch.zeros((batch_size, 3 * ndof), dtype=y_n.dtype, device=y_n.device)
    
    jacobian_fn = StatefulJacobian(batched_jacobian, strategy, recompute_every=recompute_every)
    
    x_sol = batched_newton_solve(
        residual_fn=batched_residual,
        jacobian_fn=jacobian_fn,
        x0=x0,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
    )
    
    # Extract the converged stage derivatives
    Yp_1 = x_sol[:, :ndof]
    Yp_2 = x_sol[:, ndof:2*ndof]
    Yp_3 = x_sol[:, 2*ndof:]
    
    # Compute the final state using the "stiff accuracy" property (y_next = Y_3, yp_next = Yp_3)
    y_next_flat = y_n_flat + h * (a31 * Yp_1 + a32 * Yp_2 + a33 * Yp_3)
    yp_next_flat = Yp_3
    
    return y_next_flat.view(y_n.shape), yp_next_flat.view(y_n.shape)


def solve_radau_iia5(
    F: Callable,
    t_span: Tuple[float, float],
    y0: torch.Tensor,
    args: Optional[Tuple] = None,
    yp0: Optional[torch.Tensor] = None,
    h: Optional[float] = None,
    n_steps: Optional[int] = None,
    ic_tol: float = 1e-8,
    step_tol: float = 1e-8,
    strict: bool = False,
    damping: float = 1.0,
    compiles_step: bool = False,
    strategy: str = "freeze_dynamic", 
    recompute_every: Optional[int] = None,
    event_fn: Callable = None,
    reset_fn: Callable = None,
    max_iter_for_events = 100
) -> DAESolution:
    """
    3-stage, 5th-order fully Implicit Runge-Kutta Radau IIA solver.
    """
    if args is not None:
        F = partial(F, *args)

    t0, t1 = t_span
    h, n_steps, yp_guess = prepare_solver_inputs(F, t_span, y0, yp0, h, n_steps, ic_tol, strict)

    yp = solve_consistent_yp0(F, t0, y0, yp_guess, tol=ic_tol)

    ys: List[torch.Tensor] = [y0.clone()]
    ts: List[float] = [t0]
    y_prev = y0
    yp_prev = yp
    t = t0

    step_fn = radau_iia5_step
    if compiles_step:
        step_fn = try_compile(step_fn)

    event_triggered = False
    event_mask = torch.zeros(y0.shape[0], dtype=torch.bool, device=y0.device) if y0.dim() > 1 else torch.tensor(False, device=y0.device)
    t_event, y_event = None, None

    # Step until t reaches the terminal simulation time t1
    while t < t1:
        t_next = min(t + h, t1)
        h_actual = t_next - t
        y_next, yp_next = step_fn(
            F, t, h_actual, y_prev, 
            tol=step_tol, damping=damping, strategy=strategy, recompute_every=recompute_every
        )
        
        # checks for overshoots and corrects them
        if event_fn is not None:
            t_evt, y_evt, yp_evt, evt_mask = handle_step_events(
                event_fn, t, t_next, y_prev, y_next, yp_prev, yp_next, tol=step_tol, max_iter=max_iter_for_events
            )
            
            if evt_mask.any():
                event_triggered = True
                event_mask = evt_mask
                t_event = t_evt
                y_event = y_evt

                t, y_prev, yp_prev = apply_event_reset(F, t_evt, y_evt, yp_evt, ys, ts, reset_fn, ic_tol=ic_tol)
                
                if reset_fn is None:
                    break
                else:
                    continue

        ys.append(y_next)
        ts.append(t_next)

        y_prev = y_next
        yp_prev = yp_next
        t = t_next

    return DAESolution(
        ts=torch.tensor(ts, dtype=y0.dtype, device=y0.device),
        ys=torch.stack(ys),
        yp_final=yp_prev,
        success=not event_triggered,
        event_triggered=event_triggered,
        event_mask=event_mask,
        t_event=t_event,
        y_event=y_event
    )

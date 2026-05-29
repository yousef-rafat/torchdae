import math
import torch
from .algorithms import solve_consistent_yp0
from typing import Callable, Optional, Tuple, List, Union
from .util import handle_step_events
from .common import (
    batched_newton_solve, StatefulJacobian, try_compile, DAESolution, DAEFunctions, resolve_dae_components, prepare_solver_inputs
)

__all__ = [
    "solve_bdf1",
    "solve_bdf2",
    "solve_tr_bdf2"
]

def apply_event_reset(
    F: Callable, t_evt: torch.Tensor, y_evt: torch.Tensor,
    yp_evt: torch.Tensor, ys: List[torch.Tensor], ts: List[float], 
    reset_fn: Optional[Callable], ic_tol: float,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    ys.append(y_evt)
    
    t_val = t_evt.item() if t_evt.dim() == 0 else t_evt[0].item()
    ts.append(t_val)
    
    if reset_fn is None:
        return t_val, y_evt, yp_evt
        
    y_reset = reset_fn(t_evt, y_evt)
    ys.append(y_reset)
    ts.append(t_val)
    
    yp_reset = solve_consistent_yp0(F, t_val, y_reset, yp_evt, tol=ic_tol)
    
    return t_val, y_reset, yp_reset

def bdf1_step(
    F: Callable, t: float, h: float, y_prev: torch.Tensor,
    tol: float = 1e-8, max_iter: int = 20, damping: float = 1.0,
    strategy="freeze_dynamic", recompute_every=None
) -> torch.Tensor:
    coeff = 1.0 / h
    state_shape = y_prev.shape[1:]
    
    y_prev_flat = y_prev.flatten(start_dim=1)
    
    # single-instance residual function (D,) -> (D,)
    def G_single(yf_single: torch.Tensor, y_prev_single: torch.Tensor):
        y_in = yf_single.view(state_shape)
        yp_in = (y_in - y_prev_single.view(state_shape)) * coeff
        res = F(t, y_in, yp_in)
        return res.flatten()

    # functions for batched inputs
    batched_residual = lambda yf: torch.func.vmap(G_single)(yf, y_prev_flat)  # noqa: E731
    batched_jacobian = lambda yf: torch.func.vmap(torch.func.jacrev(G_single, argnums=0))(yf, y_prev_flat) # noqa: E731
    jacobian_fn = StatefulJacobian(batched_jacobian, strategy, y_prev_flat, recompute_every=recompute_every)
    
    y_next_flat = batched_newton_solve(
        residual_fn=batched_residual,
        jacobian_fn=jacobian_fn,
        x0=y_prev_flat,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
    )
    
    return y_next_flat.view(y_prev.shape)


def bdf2_step(
    F: Callable, t: float, h: float, y_prev: torch.Tensor, y_prev2: torch.Tensor,
    tol: float = 1e-8, max_iter: int = 20, damping: float = 1.0,
    strategy="freeze_dynamic", recompute_every=None
) -> torch.Tensor:
    coeff = 1.0 / (2.0 * h)
    state_shape = y_prev.shape[1:]
    
    y_prev_flat = y_prev.flatten(start_dim=1)
    y_prev2_flat = y_prev2.flatten(start_dim=1)
    
    def G_single(yf_single: torch.Tensor, y_prev_single: torch.Tensor, y_prev2_single: torch.Tensor):
        y_in = yf_single.view(state_shape)
        yp_in = (3.0 * y_in - 4.0 * y_prev_single.view(state_shape) + y_prev2_single.view(state_shape)) * coeff
        res = F(t, y_in, yp_in)
        return res.flatten()

    batched_residual = lambda yf: torch.func.vmap(G_single)(yf, y_prev_flat, y_prev2_flat) # noqa: E731
    batched_jacobian = lambda yf: torch.func.vmap(torch.func.jacrev(G_single, argnums=0))(yf, y_prev_flat, y_prev2_flat)# noqa: E731
    jacobian_fn = StatefulJacobian(batched_jacobian, strategy, y_prev_flat, recompute_every=recompute_every)
    
    y_next_flat = batched_newton_solve(
        residual_fn=batched_residual,
        jacobian_fn=jacobian_fn,
        x0=y_prev_flat,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
    )
    
    return y_next_flat.view(y_prev.shape)

def tr_bdf2_step(
    F: Callable, t_n: float, h: float, y_n: torch.Tensor, yp_n: torch.Tensor,
    tol: float = 1e-8, max_iter: int = 20, damping: float = 1.0,
    strategy="freeze_dynamic", recompute_every=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Singly Diagonally Implicit Runge-Kutta TR-BDF2 step.
    y_n, yp_n: shape (B, *state_shape)
    """
    # uses one point in past, jumps a trapeziodal step to get another point and
    # solves Runge-Kutta method by the root finding newton solve function

    gamma = 2.0 - math.sqrt(2.0)  # optimal parameter for Jacobian reuse
    state_shape = y_n.shape[1:]
    
    y_n_flat = y_n.flatten(start_dim=1)
    yp_n_flat = yp_n.flatten(start_dim=1)
    
    # trapeziodal step
    t_star = t_n + gamma * h
    coeff_tr = 2.0 / (gamma * h)
    
    def G_tr_single(yf_single: torch.Tensor, y_n_single: torch.Tensor, yp_n_single: torch.Tensor):
        y_in = yf_single.view(state_shape)
        # y_prime_star = 2 / (gamma * h) * (y_star - y_n) - y_prime_n
        yp_in = (coeff_tr * (yf_single - y_n_single) - yp_n_single).view(state_shape)
        res = F(t_star, y_in, yp_in)
        return res.flatten()

    batched_residual_tr = lambda yf: torch.func.vmap(G_tr_single)(yf, y_n_flat, yp_n_flat)  # noqa: E731
    batched_jacobian_tr = lambda yf: torch.func.vmap(torch.func.jacrev(G_tr_single, argnums=0))(yf, y_n_flat, yp_n_flat)  # noqa: E731
    jacobian_fn_tr = StatefulJacobian(batched_jacobian_tr, strategy, y_n_flat, recompute_every=recompute_every)
    
    y_star_flat = batched_newton_solve(
        residual_fn=batched_residual_tr,
        jacobian_fn=jacobian_fn_tr,
        x0=y_n_flat,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
    )
    
    t_new = t_n + h
    c1 = (2.0 - gamma) / ((1.0 - gamma) * h)
    c2 = -1.0 / (gamma * (1.0 - gamma) * h)
    c3 = (1.0 - gamma) / (gamma * h)
    
    def G_bdf_single(yf_single: torch.Tensor, y_star_single: torch.Tensor, y_n_single: torch.Tensor):
        y_in = yf_single.view(state_shape)
        # y_prime_new = c1 * y_new + c2 * y_star + c3 * y_n
        yp_in = (c1 * yf_single + c2 * y_star_single + c3 * y_n_single).view(state_shape)
        res = F(t_new, y_in, yp_in)
        return res.flatten()

    batched_residual_bdf = lambda yf: torch.func.vmap(G_bdf_single)(yf, y_star_flat, y_n_flat)  # noqa: E731
    
    # SDIRK optimization
    # since gamma = 2 - sqrt(2), we can reuse the Jacobian factorizations
    if strategy == "always":
        batched_jacobian_bdf = lambda yf: \
            torch.func.vmap(torch.func.jacrev(G_bdf_single, argnums=0))(yf, y_star_flat, y_n_flat)  # noqa: E731
        jacobian_fn_bdf = StatefulJacobian(batched_jacobian_bdf, strategy, None, recompute_every=recompute_every)
    else:
        last_J = jacobian_fn_tr.cached_J
        
        class ReusedJacobian:
            def __init__(self, J):
                self.J = J
            def __call__(self, x, norm=None, iteration=None):
                return self.J
                
        jacobian_fn_bdf = ReusedJacobian(last_J)

    y_new_flat = batched_newton_solve(
        residual_fn=batched_residual_bdf,
        jacobian_fn=jacobian_fn_bdf,
        x0=y_star_flat,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
    )
    
    yp_new_flat = c1 * y_new_flat + c2 * y_star_flat + c3 * y_n_flat
    
    return y_new_flat.view(y_n.shape), yp_new_flat.view(y_n.shape)

def solve_bdf1(
    F: Callable,
    t_span: Tuple[float, float],
    y0: torch.Tensor,
    args: Optional[Tuple] = None,
    yp0: Optional[torch.Tensor] = None,
    event_fn: Callable = None,
    reset_fn: Callable = None,
    h: Optional[float] = None,
    n_steps: Optional[int] = None,
    ic_tol: float = 1e-8,
    step_tol: float = 1e-8,
    strict: bool = False,
    damping = 1.0,
    compiles_step = False,
    strategy="freeze_dynamic", recompute_every=None,
    max_iter_for_events = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        ys: (n_steps+1, *y0.shape) trajectory
        ts: (n_steps+1,) times
    """
    F, constraint_fn, projection_fn, event_fn, reset_fn = resolve_dae_components(F, args, step_tol)
    
    t0, t1 = t_span
    h, n_steps, yp_guess = prepare_solver_inputs(F, t_span, y0, yp0, h, n_steps, ic_tol, strict)

    ys: List[torch.Tensor] = [y0.clone()]
    ts: List[float] = [t0]
    y_prev = y0
    t = t0
    step_fn = bdf1_step
    if compiles_step:
        step_fn = try_compile(step_fn)

    yp_prev = yp_guess.clone()
    event_triggered = False
    event_mask = torch.zeros(y0.shape[0], dtype=torch.bool, device=y0.device) if y0.dim() > 1 else torch.tensor(False, device=y0.device)
    t_event, y_event = None, None
    while t < t1:
        t_next = min(t + h, t1)
        h_actual = t_next - t
        y_next = step_fn(
            F, t_next, h_actual, y_prev, tol=step_tol, damping=damping, strategy=strategy, recompute_every=recompute_every
        )
        yp_next = (y_next - y_prev) / h_actual

        if projection_fn is not None:
            y_next = projection_fn(y_next)
        
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
        t = t_next
        yp_prev = yp_next

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

def solve_bdf2(
    F: Union[Callable, DAEFunctions],
    t_span: Tuple[float, float],
    y0: torch.Tensor,
    args: Optional[Tuple] = None,
    yp0: Optional[torch.Tensor] = None,
    h: Optional[float] = None,
    n_steps: Optional[int] = None,
    ic_tol: float = 1e-8,
    step_tol: float = 1e-8,
    strict: bool = False,
    damping = 1.0,
    compiles_step = False,
    strategy="freeze_dynamic", recompute_every=None,
    max_iter_for_events = 100
) -> DAESolution:
    """
    BDF2 solver. Bootstraps with one BDF1 step, then switches to BDF2.
    """
    # constraint_fn could have some diagnostics benefits
    F, constraint_fn, projection_fn, event_fn, reset_fn = resolve_dae_components(F, args, step_tol)

    t0, t1 = t_span
    h, n_steps, yp_guess = prepare_solver_inputs(F, t_span, y0, yp0, h, n_steps, ic_tol, strict)

    ys: List[torch.Tensor] = [y0.clone()]
    ts: List[float] = [t0]
    y_prev = y0
    t = t0
    step_fn = bdf2_step
    if compiles_step:
        step_fn = try_compile(step_fn)

    yp_prev = yp_guess.clone()
    event_triggered = False
    event_mask = torch.zeros(y0.shape[0], dtype=torch.bool, device=y0.device) if y0.dim() > 1 else torch.tensor(False, device=y0.device)
    t_event, y_event = None, None

    needs_bootstrap = True
    while t < t1:
        t_next = min(t + h, t1)
        h_actual = t_next - t
        
        if needs_bootstrap:
            y_next = bdf1_step(
                F, t_next, h_actual, y_prev, tol=step_tol, damping=damping, strategy=strategy, recompute_every=recompute_every
            )
            yp_next = (y_next - y_prev) / h_actual
            needs_bootstrap = False
        else:
            y_next = step_fn(
                F, t_next, h_actual, y_prev, ys[-2], tol=step_tol, damping=damping, strategy=strategy, recompute_every=recompute_every
            )
            yp_next = (3.0 * y_next - 4.0 * y_prev + ys[-2]) / (2.0 * h_actual)

        if projection_fn is not None:
            y_next = projection_fn(y_next)

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
        t = t_next
        yp_prev = yp_next

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

def solve_tr_bdf2(
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
    max_iter_for_events = 100
) -> DAESolution:
    """
    Singly Diagonally Implicit Runge-Kutta TR-BDF2 solver.
    """
    F, constraint_fn, projection_fn, event_fn, reset_fn = resolve_dae_components(F, args, step_tol)

    t0, t1 = t_span
    h, n_steps, yp_guess = prepare_solver_inputs(F, t_span, y0, yp0, h, n_steps, ic_tol, strict)

    # for tr_bdf2, solving for intial yp0 is needed
    yp = solve_consistent_yp0(F, t0, y0, yp_guess, tol=ic_tol)

    ys: List[torch.Tensor] = [y0.clone()]
    ts: List[float] = [t0]
    y_prev = y0
    yp_prev = yp
    t = t0

    step_fn = tr_bdf2_step
    if compiles_step:
        step_fn = try_compile(step_fn)

    event_triggered = False
    event_mask = torch.zeros(y0.shape[0], dtype=torch.bool, device=y0.device) if y0.dim() > 1 else torch.tensor(False, device=y0.device)
    t_event, y_event = None, None

    while t < t1:
        t_next = min(t + h, t1)
        h_actual = t_next - t
        y_next, yp_next = step_fn(
            F, t, h_actual, y_prev, yp_prev, 
            tol=step_tol, damping=damping, strategy=strategy, recompute_every=recompute_every
        )

        if projection_fn is not None:
            y_next = projection_fn(y_next)

        
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

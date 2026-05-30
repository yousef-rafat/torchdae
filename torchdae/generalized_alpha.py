import math
import torch
import torch.func as func
from functools import partial
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List

from .common import batched_newton_solve, StatefulJacobian, try_compile
from .util import handle_step_events

__all__ = ["solve_generalized_alpha", "solve_consistent_a0", "MechanicalDAESolution"]


@dataclass(frozen=True)
class MechanicalDAESolution:
    """
    Standardized, immutable container for second-order mechanical DAE solver results.
    """
    ts: torch.Tensor                           # Timestamps of the solved trajectory: Shape (T,)
    qs: torch.Tensor                           # Positions of the solved trajectory: Shape (T, B, *state_shape)
    vs: torch.Tensor                           # Velocities of the solved trajectory: Shape (T, B, *state_shape)
    a_final: torch.Tensor                      # Final acceleration at termination: Shape (B, *state_shape)
    lam_final: torch.Tensor                    # Final Lagrange multipliers: Shape (B, ncon)
    success: bool                              # True if solver completed without errors or forced terminations
    
    # Event-handling metadata
    event_triggered: bool = False              # True if a boundary event stopped the solver early
    event_mask: Optional[torch.Tensor] = None  # Boolean mask of which batch elements triggered the event: Shape (B,)
    t_event: Optional[torch.Tensor] = None     # Exact times of events: Shape (B,)
    q_event: Optional[torch.Tensor] = None     # Exact positions at event time: Shape (B, *state_shape)
    v_event: Optional[torch.Tensor] = None     # Exact velocities at event time: Shape (B, *state_shape)


def _compute_params_from_rho_inf(rho_inf: float) -> Tuple[float, float, float, float]:
    """
    Optimal generalized-alpha parameters for second-order accuracy
    and unconditional stability.
    """
    alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
    alpha_f = rho_inf / (rho_inf + 1.0)
    beta = (1.0 - alpha_m + alpha_f) ** 2 / 4.0
    gamma = 0.5 - alpha_m + alpha_f
    return alpha_m, alpha_f, beta, gamma


def _make_M_callable(M):
    if isinstance(M, torch.Tensor):
        M_tensor = M.detach().clone()
        return lambda q: M_tensor
    return M


def solve_consistent_a0(
    M: Callable[[torch.Tensor], torch.Tensor],
    F: Callable[[float, torch.Tensor, torch.Tensor], torch.Tensor],
    g: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t0: float,
    q0: torch.Tensor,                                      # Shape: (B, *state_shape)
    v0: torch.Tensor,                                      # Shape: (B, *state_shape)
    tol: float = 1e-8,
    max_iter: int = 50,
    damping: float = 1.0,
    jacobian_strategy: str = "always",
    recompute_every: Optional[int] = None,
    args: Optional[Tuple] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Consistent initialization for index-2 mechanical DAE on batched inputs.
    Solves:
        M(q0) @ a0 + G(q0)^T @ lam0 = f(t0, q0, v0)
        g_q(q0,v0) @ v0 + g_v(q0,v0) @ a0 = 0
    """
    if args is not None:
        F = partial(F, *args)

    batch_size = q0.shape[0]
    state_shape = q0.shape[1:]
    ndof = q0[0].numel()

    # Determine constraint shape by running a single element
    dummy_g = g(q0[0:1].view(1, *state_shape), v0[0:1].view(1, *state_shape))
    ncon = dummy_g[0].numel()

    # Flatten coordinates per batch element
    q0_flat = q0.flatten(start_dim=1)
    v0_flat = v0.flatten(start_dim=1)

    # Precompute constraint Jacobians at initial condition
    # This prevents taking expensive nested derivatives inside Newton iterations
    def G_q_single(q_single, v_single):
        g_of_q = lambda q_unflat: g(q_unflat, v_single.view(state_shape)) # noqa: E731
        return func.jacrev(g_of_q)(q_single.view(state_shape)).reshape(ncon, ndof)

    def G_v_single(q_single, v_single):
        g_of_v = lambda v_unflat: g(q_single.view(state_shape), v_unflat)  # noqa: E731
        return func.jacrev(g_of_v)(v_single.view(state_shape)).reshape(ncon, ndof)

    G_q_batch = func.vmap(G_q_single)(q0_flat, v0_flat)  # (B, ncon, ndof)
    G_v_batch = func.vmap(G_v_single)(q0_flat, v0_flat)  # (B, ncon, ndof)

    # Single-instance algebraic residual for acceleration and multipliers
    def R_single(x_single, q0_s, v0_s, G_q_s, G_v_s):
        a = x_single[:ndof]
        lam = x_single[ndof:]

        q_unflat = q0_s.view(state_shape)
        v_unflat = v0_s.view(state_shape)

        M_eval = M(q_unflat)
        force = F(t0, q_unflat, v_unflat)

        R_dyn = M_eval @ a - force.flatten() + G_q_s.T @ lam
        R_con = G_q_s @ v0_s + G_v_s @ a

        return torch.cat([R_dyn, R_con])

    batched_residual = lambda x: func.vmap(R_single)(x, q0_flat, v0_flat, G_q_batch, G_v_batch)  # noqa: E731
    raw_jacobian_fn = lambda x: func.vmap(func.jacrev(R_single, argnums=0))(x, q0_flat, v0_flat, G_q_batch, G_v_batch) # noqa: E731

    # Initial guesses
    a0 = torch.zeros((batch_size, ndof), dtype=q0.dtype, device=q0.device)
    lam0 = torch.zeros((batch_size, ncon), dtype=q0.dtype, device=q0.device)
    x0 = torch.cat([a0, lam0], dim=-1)

    jacobian_wrapper = StatefulJacobian(
        raw_jac_fn=raw_jacobian_fn,
        strategy=jacobian_strategy,
        x0=x0,
        recompute_every=recompute_every,
    )

    x_sol = batched_newton_solve(
        residual_fn=batched_residual,
        jacobian_fn=jacobian_wrapper,
        x0=x0,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
    )

    a0_sol = x_sol[:, :ndof].view(q0.shape)
    lam0_sol = x_sol[:, ndof:].view(batch_size, ncon)
    return a0_sol, lam0_sol


def generalized_alpha_step(
    M: Callable[[torch.Tensor], torch.Tensor],
    F: Callable[[float, torch.Tensor, torch.Tensor], torch.Tensor],
    g: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t_n: float,
    h: float,
    q_n: torch.Tensor,                                      # Shape: (B, *state_shape)
    v_n: torch.Tensor,                                      # Shape: (B, *state_shape)
    a_n: torch.Tensor,                                      # Shape: (B, *state_shape)
    lam_n: torch.Tensor,                                    # Shape: (B, ncon)
    alpha_m: float,
    alpha_f: float,
    beta: float,
    gamma: float,
    tol: float = 1e-8,
    max_iter: int = 50,
    damping: float = 1.0,
    strategy: str = "always",
    recompute_every: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single Generalized-α step for index-2 DAE over batch dimension (dim 0).
    """
    batch_size = q_n.shape[0]
    state_shape = q_n.shape[1:]
    ndof = q_n[0].numel()
    ncon = lam_n.shape[1]

    q_n_flat = q_n.flatten(start_dim=1)
    v_n_flat = v_n.flatten(start_dim=1)
    a_n_flat = a_n.flatten(start_dim=1)

    # Single-instance residual calculation
    def G_single(x_single, q_n_s, v_n_s, a_n_s, lam_n_s):
        a_new = x_single[:ndof]
        lam_new = x_single[ndof:]

        # Generalized-α updates
        q_new = q_n_s + h * v_n_s + h**2 * ((0.5 - beta) * a_n_s + beta * a_new)
        v_new = v_n_s + h * ((1.0 - gamma) * a_n_s + gamma * a_new)

        t_af = t_n + h - alpha_f * h
        q_af = (1.0 - alpha_f) * q_new + alpha_f * q_n_s
        v_af = (1.0 - alpha_f) * v_new + alpha_f * v_n_s
        a_am = (1.0 - alpha_m) * a_new + alpha_m * a_n_s

        q_af_unflat = q_af.view(state_shape)
        v_af_unflat = v_af.view(state_shape)
        q_new_unflat = q_new.view(state_shape)
        v_new_unflat = v_new.view(state_shape)

        M_af = M(q_af_unflat)
        force = F(t_af, q_af_unflat, v_af_unflat)
        R_dyn = M_af @ a_am - force.flatten()

        # Constraint Jacobian G(q_new) = dg/dq at (q_new, v_new)
        g_of_q = lambda q: g(q, v_new_unflat) # noqa: E731
        G_q = func.jacrev(g_of_q)(q_new_unflat).reshape(ncon, ndof)

        R_dyn = R_dyn + G_q.T @ lam_new
        R_con = g(q_new_unflat, v_new_unflat).flatten()

        return torch.cat([R_dyn, R_con])

    batched_residual = lambda x: func.vmap(G_single)(x, q_n_flat, v_n_flat, a_n_flat, lam_n) # noqa: E731
    raw_jacobian_fn = lambda x: func.vmap(func.jacrev(G_single, argnums=0))(x, q_n_flat, v_n_flat, a_n_flat, lam_n) # noqa: E731

    # Build initial guess for Newton iteration
    x0 = torch.cat([a_n_flat, lam_n], dim=-1)

    jacobian_wrapper = StatefulJacobian(
        raw_jac_fn=raw_jacobian_fn,
        strategy=strategy,
        x0=x0,
        recompute_every=recompute_every,
    )

    x_sol = batched_newton_solve(
        residual_fn=batched_residual,
        jacobian_fn=jacobian_wrapper,
        x0=x0,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
    )

    a_new = x_sol[:, :ndof].view(q_n.shape)
    lam_new = x_sol[:, ndof:].view(batch_size, ncon)

    # Recover the updated states from the solved acceleration
    q_new = q_n + h * v_n + h**2 * ((0.5 - beta) * a_n + beta * a_new)
    v_new = v_n + h * ((1.0 - gamma) * a_n + gamma * a_new)

    return q_new, v_new, a_new, lam_new


def solve_generalized_alpha(
    M,
    F: Callable[[float, torch.Tensor, torch.Tensor], torch.Tensor],
    g: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t_span: Tuple[float, float],
    q0: torch.Tensor,                                      # Shape: (B, *state_shape)
    v0: torch.Tensor,                                      # Shape: (B, *state_shape)
    a0: Optional[torch.Tensor] = None,                     # Shape: (B, *state_shape)
    lam0: Optional[torch.Tensor] = None,                   # Shape: (B, ncon)
    h: Optional[float] = None,
    n_steps: Optional[int] = None,
    rho_inf: float = 0.8,
    alpha_m: Optional[float] = None,
    alpha_f: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    tol: float = 1e-8,
    max_iter: int = 50,
    damping: float = 1.0,
    strategy: str = "always",
    recompute_every: Optional[int] = None,
    compile_steps: bool = False,
    event_fn: Optional[Callable] = None,
    reset_fn: Optional[Callable] = None,
    max_iter_for_events: int = 100,
    args: Optional[Tuple] = None,
) -> MechanicalDAESolution:
    """
    Generalized-α solver for index-2 mechanical DAE with batch support.
    """
    if args is not None:
        F = partial(F, *args)
    M = _make_M_callable(M)
    t0, t1 = t_span

    if h is None and n_steps is None:
        raise ValueError("Provide h or n_steps")
    if h is None:
        h = (t1 - t0) / n_steps  # type: ignore
    if n_steps is None:
        n_steps = int(math.ceil((t1 - t0) / h))

    if alpha_m is None:
        alpha_m, alpha_f, beta, gamma = _compute_params_from_rho_inf(rho_inf)

    # for generalized_alpha, solving for consistent initial conditions is needed
    if a0 is None or lam0 is None:
        a, lam = solve_consistent_a0(
            M, F, g, t0, q0, v0, 
            tol=tol, max_iter=max_iter, damping=damping,
            jacobian_strategy=strategy,
            recompute_every=recompute_every,
        )
    else:
        a = a0.clone().detach()
        lam = lam0.clone().detach()

    qs: List[torch.Tensor] = [q0.clone()]
    vs: List[torch.Tensor] = [v0.clone()]
    ts: List[float] = [t0]

    q = q0
    v = v0
    t = t0

    step_fn = generalized_alpha_step
    if compile_steps:
        step_fn = try_compile(generalized_alpha_step)

    # Event tracking initialization
    ndof = q0[0].numel()
    event_triggered = False
    event_mask = torch.zeros(q0.shape[0], dtype=torch.bool, device=q0.device) if q0.dim() > 1 else torch.tensor(False, device=q0.device)
    t_event, q_event, v_event = None, None, None

    # Step until t reaches the terminal simulation time t1
    while t < t1 - 1e-10:
        t_next = min(t + h, t1)
        h_actual = t_next - t
        q_next, v_next, a_next, lam_next = step_fn(
            M, F, g, t, h_actual, q, v, a, lam,
            alpha_m, alpha_f, beta, gamma,
            tol=tol, max_iter=max_iter, damping=damping,
            strategy=strategy, recompute_every=recompute_every,
        )

        # checks for overshoots and corrects them
        if event_fn is not None:
            # Concatenate position and velocity to form the unified state y = [q, v]
            y_prev_cat = torch.cat([q.flatten(start_dim=1), v.flatten(start_dim=1)], dim=-1)
            y_next_cat = torch.cat([q_next.flatten(start_dim=1), v_next.flatten(start_dim=1)], dim=-1)

            # The derivative is yp = [v, a]
            yp_prev_cat = torch.cat([v.flatten(start_dim=1), a.flatten(start_dim=1)], dim=-1)
            yp_next_cat = torch.cat([v_next.flatten(start_dim=1), a_next.flatten(start_dim=1)], dim=-1)

            # Wrap the event function to parse the concatenated states
            def event_fn_wrap(t_val, y_val):
                q_val = y_val[..., :ndof].view(q0.shape)
                v_val = y_val[..., ndof:].view(v0.shape)
                return event_fn(t_val, q_val, v_val)

            t_evt, y_evt, yp_evt, evt_mask = handle_step_events(
                event_fn_wrap, t, t_next, y_prev_cat, y_next_cat, yp_prev_cat, yp_next_cat, 
                tol=tol, max_iter=max_iter_for_events
            )

            if evt_mask.any():
                event_triggered = True
                event_mask = evt_mask
                t_event = t_evt

                # Unpack the exact event states
                q_event = y_evt[..., :ndof].view(q0.shape)
                v_event = y_evt[..., ndof:].view(v0.shape)

                # record the exact boundary state
                qs.append(q_event)
                vs.append(v_event)
                # handles batched and unbatched
                ts.append(t_event.item() if t_event.dim() == 0 else t_event[0].item())

                if reset_fn is None:
                    # Retrieve final acceleration and multipliers at the event boundary
                    a_final = yp_evt[..., ndof:].view(q0.shape)
                    lam_final = lam_next
                    break
                else:
                    # calls restart function and restart the simulation
                    q_reset, v_reset = reset_fn(t_event, q_event, v_event)
                    qs.append(q_reset)
                    vs.append(v_reset)
                    ts.append(t_event.item() if t_event.dim() == 0 else t_event[0].item())

                    t = t_event.item() if t_event.dim() == 0 else t_event[0].item()
                    q = q_reset
                    v = v_reset

                    # Re-solve for consistent acceleration and Lagrange multipliers at the post-reset state
                    a, lam = solve_consistent_a0(
                        M, F, g, t, q, v, 
                        tol=tol, max_iter=max_iter, damping=damping,
                        jacobian_strategy=strategy, recompute_every=recompute_every
                    )
                    continue

        qs.append(q_next)
        vs.append(v_next)
        ts.append(t_next)

        q = q_next
        v = v_next
        a = a_next
        lam = lam_next
        t = t_next

    # Unpack outputs based on final convergence status
    if not event_triggered:
        a_final = a
        lam_final = lam

    return MechanicalDAESolution(
        ts=torch.tensor(ts, dtype=q0.dtype, device=q0.device),
        qs=torch.stack(qs),
        vs=torch.stack(vs),
        a_final=a_final,
        lam_final=lam_final,
        success=not event_triggered,
        event_triggered=event_triggered,
        event_mask=event_mask,
        t_event=t_event,
        q_event=q_event,
        v_event=v_event
    )

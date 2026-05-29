import torch
import torch.func as func
from typing import Callable, Optional, Tuple
from .projections import coordinate_projection
from .common import batched_newton_solve, StatefulJacobian

__all__ = ["compute_consistent_initial_conditions"]

def solve_consistent_yp0(
    F: Callable,
    t0: float,
    y0: torch.Tensor,
    yp0_guess: Optional[torch.Tensor] = None,
    tol: float = 1e-8,
    max_iter: int = 50,
    damping: float = 1.0,
) -> torch.Tensor:
    """
    Given batched y0, solve F(t0, y0, yp0) = 0 for yp0 using batched Newton.

    Assumes y0 is static and correct
    """
    yp = (
        yp0_guess.clone().detach()
        if yp0_guess is not None
        else torch.zeros_like(y0)
    )

    state_shape = y0.shape[1:]
    y0_flat = y0.flatten(start_dim=1)
    yp_flat = yp.flatten(start_dim=1)

    # single-instance residual function (D,) -> (D,)
    def R_single(ypf_single: torch.Tensor, y_single: torch.Tensor):
        y_in = y_single.view(state_shape)
        yp_in = ypf_single.view(state_shape)
        res = F(t0, y_in, yp_in)
        return res.flatten()

    batched_residual = lambda ypf: torch.func.vmap(R_single)(ypf, y0_flat)  # noqa: E731
    
    # argnums=0 ensures we compute the Jacobian with respect to ypf_single
    def batched_jacobian(ypf):
        return torch.func.vmap(
            torch.func.jacrev(R_single, argnums=0)
        )(ypf, y0_flat)

    yp_consistent_flat = batched_newton_solve(
        residual_fn=batched_residual,
        jacobian_fn=batched_jacobian,
        x0=yp_flat,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
    )

    return yp_consistent_flat.view(y0.shape)


def compute_consistent_initial_conditions(
    F: Callable,
    t0: float,
    y0: torch.Tensor,
    yp0_guess: Optional[torch.Tensor] = None,
    tol: float = 1e-8,
    max_iter: int = 50,
    projection_iter = 15,
    damping: float = 1.0,
    index: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Automates the calculation of consistent initial conditions (y0, yp0) 
    for Index-1, Index-2, and Index-3 DAEs.

    Assumes y0 is a guess
    """
    state_shape = y0.shape[1:]
    
    yp_guess = yp0_guess if yp0_guess is not None else torch.zeros_like(y0)
    
    y0_sample = y0[0]
    yp0_sample = yp_guess[0]
    
    # discover algebric equations using norms near zero
    def F_flat(yp_flat):
        return F(t0, y0_sample, yp_flat.view(state_shape)).flatten()
        
    J_yp = func.jacrev(F_flat)(yp0_sample.flatten())
    row_norms = torch.linalg.vector_norm(J_yp, dim=1)
    algebraic_indices = torch.where(row_norms < 1e-12)[0]
    
    # get the correct y0, so we can solve for yp0
    if len(algebraic_indices) > 0 and index >= 2:
        # wrapper to extract algebraic equations only
        def g_constraint(y_tensor):
            return F(t0, y_tensor, yp0_sample)[..., algebraic_indices]
        
        y0_consistent = coordinate_projection(g=g_constraint, y_trial=y0, tol=tol, max_iter=projection_iter)
    else:
        y0_consistent = y0.clone()
        
    # given y0, solve for yp0
    yp_flat = yp_guess.flatten(start_dim=1).clone()
    y0_cons_flat = y0_consistent.flatten(start_dim=1)
    
    def R_single(yp_s, y_s):
        return F(t0, y_s.view(state_shape), yp_s.view(state_shape)).flatten()
        
    batched_residual = lambda yp: func.vmap(R_single)(yp, y0_cons_flat)  # noqa: E731
    batched_jacobian = lambda yp: func.vmap(func.jacrev(R_single, argnums=0))(yp, y0_cons_flat) # noqa: E731
    
    jacobian_fn = StatefulJacobian(batched_jacobian, "always")
    
    yp_consistent_flat = batched_newton_solve(
        residual_fn=batched_residual,
        jacobian_fn=jacobian_fn,
        x0=yp_flat,
        tol=tol,
        max_iter=max_iter,
        damping=damping,
    )
    
    yp0_consistent = yp_consistent_flat.view_as(y0)
    return y0_consistent, yp0_consistent

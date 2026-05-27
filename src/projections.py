import torch
import torch.func as func
from typing import Callable

__all__ = ["coordinate_projection"]

def coordinate_projection(
    g: Callable[[torch.Tensor], torch.Tensor],  # Constraint function g(y) -> Tensor (B, C)
    y_trial: torch.Tensor,                      # Unprojected state from solver: Shape (B, D)
    tol: float = 1e-8,
    max_iter: int = 15,
) -> torch.Tensor:
    """
    Applies the Coordinate Projection Method (CPM) to project the batched trial state 
    y_trial onto the algebraic constraint manifold g(y) = 0 using a minimum-norm 
    least-squares correction.
    """
    state_shape = y_trial.shape[1:]
    y_flat = y_trial.flatten(start_dim=1).clone()
    y_prev_flat = y_flat.clone()
    
    def g_jacobian(yf):
        def jac_single(y_s):
            g_single = lambda y: g(y.view(state_shape)).flatten()  # noqa: E731
            return func.jacrev(g_single)(y_s)
        return func.vmap(jac_single)(yf)

    for _ in range(max_iter):
        g_val = func.vmap(lambda y_s: g(y_s.view(state_shape)).flatten())(y_flat)
        
        # Check convergence
        g_norm = torch.linalg.vector_norm(g_val, dim=-1)
        if g_norm.max() < tol:
            break
            
        J_g = g_jacobian(y_flat)
        
        # Solve the symmetric system: (J_g @ J_g^T) w = g(y) + J_g @ (y_prev - y)
        A = torch.bmm(J_g, J_g.transpose(-1, -2))
        
        # Compute right-hand side
        diff = (y_prev_flat - y_flat).unsqueeze(-1)
        rhs = g_val.unsqueeze(-1) + torch.bmm(J_g, diff)
        
        try:
            w = torch.linalg.solve(A, rhs)
        except RuntimeError:
            w = torch.linalg.lstsq(A, rhs).solution

        # Minimum-norm state update: y_next = y_prev - J_g^T @ w
        delta_y = torch.bmm(J_g.transpose(-1, -2), w).squeeze(-1)
        y_flat = y_prev_flat - delta_y

    return y_flat.view_as(y_trial)

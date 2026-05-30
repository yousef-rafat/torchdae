import torch
import torch.func as func
from typing import Callable, Tuple, List, Optional

class DAEAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        y0: torch.Tensor, 
        t_span: Tuple[float, float], 
        h: float, 
        p_flat: torch.Tensor, 
        F: Callable, 
        solver_fn: Callable
    ) -> torch.Tensor:
        with torch.no_grad():
            sol = solver_fn(F, t_span, y0, h=h)
            
        ctx.save_for_backward(sol.ys, sol.ts, p_flat)
        ctx.F = F
        ctx.h = h
        ctx.solver_fn = solver_fn
        return sol.ys

    @staticmethod
    def backward(ctx, grad_ys: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        ys, ts, p_flat = ctx.saved_tensors
        F = ctx.F
        h = ctx.h
        
        T_steps = ys.shape[0]
        grad_output = grad_ys[-1]
        
        # 1. Compute initial mass matrix M_{n+1} at the terminal boundary t = T
        t_np1 = ts[-1].item()
        y_np1 = ys[-1]
        yp_np1 = (ys[-1] - ys[-2]) / h if T_steps > 1 else torch.zeros_like(y_np1)
        
        M_np1 = func.vmap(func.jacrev(lambda yp_s: F(t_np1, y_np1[0], yp_s)))(yp_np1)
        M_np1_T = M_np1.transpose(-1, -2)
        
        try:
            lam = torch.linalg.solve(M_np1_T, grad_output.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            lam = torch.linalg.lstsq(M_np1_T, grad_output.unsqueeze(-1)).solution.squeeze(-1)
            
        grad_p = torch.zeros_like(p_flat)
        
        # 2. Integrate the Adjoint DAE backward in time
        for n in reversed(range(T_steps - 1)):
            t_n = ts[n].item()
            y_n = ys[n]
            yp_n = (ys[n+1] - ys[n]) / h
            
            # Compute dynamic J_y and M_n at the current step
            J_n = func.vmap(func.jacrev(lambda y_s: F(t_n, y_s, yp_n[0])))(y_n)
            M_n = func.vmap(func.jacrev(lambda yp_s: F(t_n, y_n[0], yp_s)))(yp_n)
            
            M_n_T = M_n.transpose(-1, -2)
            
            # Solve (M_n^T - h * J_n^T) * lam_n = M_{n+1}^T * lam_{n+1} + h * grad_ys[n]
            A = M_n_T - h * J_n.transpose(-1, -2)
            rhs = torch.bmm(M_np1_T, lam.unsqueeze(-1)).squeeze(-1) + h * grad_ys[n]
            
            try:
                lam = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                lam = torch.linalg.lstsq(A, rhs.unsqueeze(-1)).solution.squeeze(-1)
                
            # Update terminal mass matrix reference for the next backward step
            M_np1_T = M_n_T
            
        # Compute gradient with respect to initial state y0
        grad_y0 = torch.bmm(M_np1_T, lam.unsqueeze(-1)).squeeze(-1)
        
        return grad_y0, None, None, grad_p, None, None


def solve_dae_adjoint(
    F: Callable,
    t_span: Tuple[float, float],
    y0: torch.Tensor,
    h: float,
    params: List[torch.Tensor],
    solver_fn: Callable
) -> torch.Tensor:
    p_flat = torch.cat([p.flatten() for p in params]) if len(params) > 0 else torch.empty(0, device=y0.device)
    return DAEAdjointFunction.apply(y0, t_span, h, p_flat, F, solver_fn)

import torch
import torch.func as func
from typing import Callable, Tuple, List, Optional

__all__ = ["DAEAdjointFunction", "solve_dae_adjoint"]


class DAEAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        y0: torch.Tensor, 
        t_span: Tuple[float, float], 
        h: float, 
        F: Callable, 
        solver_fn: Callable,
        *params: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            sol = solver_fn(F, t_span, y0, h=h)
            
        ctx.save_for_backward(sol.ys, sol.ts, *params)
        ctx.F = F
        ctx.h = h
        ctx.solver_fn = solver_fn
        return sol.ys

    @staticmethod
    def backward(ctx, grad_ys: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        saved_tensors = ctx.saved_tensors
        ys = saved_tensors[0]
        ts = saved_tensors[1]
        params = saved_tensors[2:]
        
        F = ctx.F
        h = ctx.h
        
        T_steps = ys.shape[0]
        grad_output = grad_ys[-1]
        
        # compute the mass matrix: dF/dy_dot
        t_N = ts[-1].item()
        y_N = ys[-1]
        yp_N = (ys[-1] - ys[-2]) / h if T_steps > 1 else torch.zeros_like(y_N)
        
        J_N = func.vmap(func.jacrev(lambda y_s: F(t_N, y_s, yp_N[0])))(y_N)
        M_N = func.vmap(func.jacrev(lambda yp_s: F(t_N, y_N[0], yp_s)))(yp_N)
        
        M_N_T = M_N.transpose(-1, -2)
        J_N_T = J_N.transpose(-1, -2)
        
        # Solve (M_N^T + h * J_N^T) * lam = h * grad_output
        A_terminal = M_N_T + h * J_N_T
        rhs_terminal = h * grad_output
        
        # get the inital adjoint sensitivity from mass matrix
        try:
            lam = torch.linalg.solve(A_terminal, rhs_terminal.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            lam = torch.linalg.lstsq(A_terminal, rhs_terminal.unsqueeze(-1)).solution.squeeze(-1)
            
        grad_params_list = [torch.zeros_like(p) for p in params]
        
        # Accumulate parameter gradients at the terminal step n = N
        if len(params) > 0:
            y_temp = y_N.detach().requires_grad_(True)
            yp_temp = yp_N.detach().requires_grad_(True)
            
            with torch.enable_grad():
                res = F(t_N, y_temp, yp_temp)
                scalar_product = torch.sum(lam * res)
                
            grads = torch.autograd.grad(scalar_product, params, allow_unused=True)
            for i, g in enumerate(grads):
                if g is not None:
                    grad_params_list[i] = grad_params_list[i] - g
                    
        M_np1_T = M_N_T
        
        for n in reversed(range(1, T_steps - 1)):
            t_n = ts[n].item()
            y_n = ys[n]
            yp_n = (ys[n] - ys[n-1]) / h
            
            # Compute Jacobians at step n
            J_n = func.vmap(func.jacrev(lambda y_s: F(t_n, y_s, yp_n[0])))(y_n)
            M_n = func.vmap(func.jacrev(lambda yp_s: F(t_n, y_n[0], yp_s)))(yp_n)
            
            M_n_T = M_n.transpose(-1, -2)
            J_n_T = J_n.transpose(-1, -2)
            
            # Solve (M_n^T + h * J_n^T) * lam_n = M_{n+1}^T * lam_{n+1} + h * grad_ys[n]
            # to get the propogation of sensitivities across one timestep
            A = M_n_T + h * J_n_T
            rhs = torch.bmm(M_np1_T, lam.unsqueeze(-1)).squeeze(-1) + h * grad_ys[n]
            
            try:
                lam = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                lam = torch.linalg.lstsq(A, rhs.unsqueeze(-1)).solution.squeeze(-1)
                
            # map the sensitivity back to the parameter space (convert sensitivity into gradients)
            if len(params) > 0:
                y_temp = y_n.detach().requires_grad_(True)
                yp_temp = yp_n.detach().requires_grad_(True)
                
                with torch.enable_grad():
                    res = F(t_n, y_temp, yp_temp)
                    scalar_product = torch.sum(lam * res)
                
                grads = torch.autograd.grad(scalar_product, params, allow_unused=True)
                for i, g in enumerate(grads):
                    if g is not None:
                        grad_params_list[i] = grad_params_list[i] - g
                
            M_np1_T = M_n_T
            
        # compute gradient with respect to initial state y0
        grad_y0 = torch.bmm(M_np1_T, lam.unsqueeze(-1)).squeeze(-1) / h
        
        return (grad_y0, None, None, None, None, *grad_params_list)

def solve_dae_adjoint(
    F: Callable,
    t_span: Tuple[float, float],
    y0: torch.Tensor,
    h: float,
    params: List[torch.Tensor],
    solver_fn: Callable,
) -> torch.Tensor:
    return DAEAdjointFunction.apply(y0, t_span, h, F, solver_fn, *params)
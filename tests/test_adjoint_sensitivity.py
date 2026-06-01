"""
test_adjoint.py

verifies both the initial condition and parameter gradients of the continuous adjoint 
dae solver against numerical finite differences, and plots trajectories.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import torch
import torch.func as func
import matplotlib.pyplot as plt

from torchdae.bdf import solve_bdf1
from torchdae import solve_dae_adjoint


def _make_F(p_val: torch.Tensor):
    """shared residual for index-1 test problem. F takes the standard 3 arguments."""
    def F(t, y, yp):
        t_tensor = torch.as_tensor(t, dtype=y.dtype, device=y.device)
        y1, y2 = y[..., 0], y[..., 1]
        y1p, _ = yp[..., 0], yp[..., 1]
        f1 = y1p + y1 - p_val[0] * y2
        f2 = y1 + y2 - p_val[1] * torch.sin(t_tensor)
        return torch.stack([f1, f2], dim=-1)
    return F


def _analytical_solution(t: float, p_val: torch.Tensor) -> torch.Tensor:
    p1, p2 = p_val[0].item(), p_val[1].item()
    # analytical solution updated to use p1 and p2 parameters
    denom = 1.0 + p1
    y1 = (p1 * p2 * math.sin(t) - p2 * math.cos(t) + p2 * math.exp(-denom * t)) / (denom * denom + 1.0)
    y2 = p2 * math.sin(t) - y1
    return torch.tensor([y1, y2])


def test_adjoint_gradient_correctness():
    # force single-threaded execution to prevent cpu thread contention slowdowns
    torch.set_num_threads(1)
    
    print("running continuous adjoint gradient validation check...")
    
    # physical parameters p (requires grad)
    p = torch.tensor([1.5, 1.0], requires_grad=True)
    F = _make_F(p)
    
    t_span = (0.0, 1.0)
    h_step = 0.01
    
    # initial state y0 (requires grad)
    y0 = torch.tensor([[0.5, -0.5]], requires_grad=True)
    
    # 2. evaluate gradients using the continuous adjoint method
    ys = solve_dae_adjoint(
        F=F,
        t_span=t_span,
        y0=y0,
        h=h_step,
        params=[p],
        solver_fn=solve_bdf1,
    )
    
    # loss is the half squared norm of the final state at t = 1.0
    y_final = ys[-1]
    loss = 0.5 * torch.sum(y_final ** 2)
    
    # backpropagate to compute grad_y0 and grad_p via the continuous adjoint pass
    loss.backward()
    grad_y0_adjoint = y0.grad.clone()
    grad_p_adjoint = p.grad.clone()
    
    print(f"  adjoint y0 gradient: {grad_y0_adjoint.numpy()}")
    print(f"  adjoint p  gradient: {grad_p_adjoint.numpy()}")

    # 3. evaluate reference gradients using standard finite differences
    print("running reference numerical finite differences...")
    eps = 1e-5
    grad_y0_numerical = torch.zeros_like(y0)
    grad_p_numerical = torch.zeros_like(p)
    
    # finite differences for initial state y0
    for i in range(y0.shape[1]):
        y0_perturbed_pos = y0.clone().detach()
        y0_perturbed_pos[0, i] += eps
        F_pos = _make_F(p.clone().detach())
        sol_pos = solve_bdf1(F_pos, t_span, y0_perturbed_pos, h=h_step)
        loss_pos = 0.5 * torch.sum(sol_pos.ys[-1] ** 2)
        
        y0_perturbed_neg = y0.clone().detach()
        y0_perturbed_neg[0, i] -= eps
        F_neg = _make_F(p.clone().detach())
        sol_neg = solve_bdf1(F_neg, t_span, y0_perturbed_neg, h=h_step)
        loss_neg = 0.5 * torch.sum(sol_neg.ys[-1] ** 2)
        
        grad_y0_numerical[0, i] = (loss_pos - loss_neg) / (2.0 * eps)
        
    # finite differences for parameters p
    for i in range(p.shape[0]):
        p_perturbed_pos = p.clone().detach()
        p_perturbed_pos[i] += eps
        F_pos = _make_F(p_perturbed_pos)
        sol_pos = solve_bdf1(F_pos, t_span, y0.clone().detach(), h=h_step)
        loss_pos = 0.5 * torch.sum(sol_pos.ys[-1] ** 2)
        
        p_perturbed_neg = p.clone().detach()
        p_perturbed_neg[i] -= eps
        F_neg = _make_F(p_perturbed_neg)
        sol_neg = solve_bdf1(F_neg, t_span, y0.clone().detach(), h=h_step)
        loss_neg = 0.5 * torch.sum(sol_neg.ys[-1] ** 2)
        
        grad_p_numerical[i] = (loss_pos - loss_neg) / (2.0 * eps)
        
    print(f"  numerical y0 gradient: {grad_y0_numerical.numpy()}")
    print(f"  numerical p  gradient: {grad_p_numerical.numpy()}")
    
    # 4. verify that the continuous adjoint gradients match the numerical gradients
    diff_y0 = (grad_y0_adjoint - grad_y0_numerical).abs().max().item()
    diff_p = (grad_p_adjoint - grad_p_numerical).abs().max().item()
    print(f"  y0 gradient discrepancy: {diff_y0:.6e}")
    print(f"  p  gradient discrepancy: {diff_p:.6e}")
    
    assert diff_y0 < 1e-3, f"adjoint y0 gradient error too large: {diff_y0:.3e}"
    assert diff_p < 1e-3, f"adjoint p gradient error too large: {diff_p:.3e}"
    print("adjoint gradient check PASSED")

    # 5. plot the forward trajectory and the adjoint states
    print("generating visualization plots...")
    ts = torch.linspace(t_span[0], t_span[1], ys.shape[0]).detach().numpy()
    y1_traj = ys[:, 0, 0].detach().numpy()
    y2_traj = ys[:, 0, 1].detach().numpy()
    
    # evaluate the adjoint states backward for plotting representation
    lambdas = []
    
    t_final = ts[-1]
    y_final = ys[-1]
    yp_final = (ys[-1] - ys[-2]) / h_step
    
    # Compute Jacobians at t = T
    M_np1 = func.vmap(lambda y_s, yp_s: func.jacrev(lambda yp_val: F(t_final, y_s, yp_val))(yp_s))(y_final, yp_final)
    M_np1_T = M_np1.transpose(-1, -2)
    
    J_final = func.vmap(lambda y_s, yp_s: func.jacrev(lambda y_val: F(t_final, y_val, yp_s))(y_s))(y_final, yp_final)
    J_final_T = J_final.transpose(-1, -2)
    
    grad_ys_mock = torch.zeros_like(ys)
    grad_output = y_final.clone().detach()
    
    # Corrected terminal discrete adjoint step: (M_T^T + h * J_T^T) * lam = h * grad_output
    A_terminal = M_np1_T + h_step * J_final_T
    rhs_terminal = h_step * grad_output
    
    try:
        lam_val = torch.linalg.solve(A_terminal, rhs_terminal.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        lam_val = torch.linalg.lstsq(A_terminal, rhs_terminal.unsqueeze(-1)).solution.squeeze(-1)
        
    # Convert discrete multiplier to continuous-scale costate (divided by h)
    lam_val_continuous = lam_val / h_step
    lambdas.append(lam_val_continuous[0].clone())
    
    for n in reversed(range(ys.shape[0] - 1)):
        t_n = ts[n]
        y_n = ys[n]
        # Approximated derivative boundary at n = 0
        yp_n = (ys[n] - ys[n-1]) / h_step if n > 0 else (ys[1] - ys[0]) / h_step
        
        J_n = func.vmap(lambda y_s, yp_s: func.jacrev(lambda y_val: F(t_n, y_val, yp_s))(y_s))(y_n, yp_n)
        M_n = func.vmap(lambda y_s, yp_s: func.jacrev(lambda yp_val: F(t_n, y_s, yp_val))(yp_s))(y_n, yp_n)
        
        M_n_T = M_n.transpose(-1, -2)
        J_n_T = J_n.transpose(-1, -2)
        
        # Corrected recurrence: (M_n^T + h * J_n^T) * lam_n_continuous = M_{n+1}^T * lam_{n+1}_continuous + grad_ys[n]
        A = M_n_T + h_step * J_n_T
        rhs = torch.bmm(M_np1_T, lam_val_continuous.unsqueeze(-1)).squeeze(-1) + grad_ys_mock[n]
        
        try:
            lam_val_continuous = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            lam_val_continuous = torch.linalg.lstsq(A, rhs.unsqueeze(-1)).solution.squeeze(-1)
            
        lambdas.append(lam_val_continuous[0].clone())
        M_np1_T = M_n_T
        
    lambdas.reverse()
    lambdas_tensor = torch.stack(lambdas).detach().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # plot 1: forward states
    ax1.plot(ts, y1_traj, label="y1 (differential)", color="blue")
    ax1.plot(ts, y2_traj, label="y2 (algebraic)", color="red", linestyle="dashed")
    ax1.set_title("forward trajectory states")
    ax1.set_xlabel("time (seconds)")
    ax1.set_ylabel("states")
    ax1.grid(True)
    ax1.legend()
    
    # plot 2: backward adjoint variables (costates)
    ax2.plot(ts, lambdas_tensor[:, 0], label="lambda1 (differential)", color="blue")
    ax2.plot(ts, lambdas_tensor[:, 1], label="lambda2 (algebraic)", color="red", linestyle="dashed")
    ax2.set_title("backward adjoint costate variables (lambda)")
    ax2.set_xlabel("time (seconds)")
    ax2.set_ylabel("adjoint multipliers")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_adjoint_gradient_correctness()
    
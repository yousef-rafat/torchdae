import sys
import os

# Insert parent directory to allow direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import torch
import matplotlib.pyplot as plt
from torchdae.radau import solve_radau_iia5

# =====================================================================
# PROBLEM 1: Simple Linear Index-1 DAE (with Analytical Solution)
# =====================================================================

def _make_linear_F(p_val: torch.Tensor):
    def F(t, y, yp):
        t_tensor = torch.as_tensor(t, dtype=y.dtype, device=y.device)
        y1, y2 = y[..., 0], y[..., 1]
        y1p, _ = yp[..., 0], yp[..., 1]
        f1 = y1p + y1 - p_val[0] * y2
        f2 = y1 + y2 - p_val[1] * torch.sin(t_tensor)
        return torch.stack([f1, f2], dim=-1)
    return F

def _linear_analytical_solution(t: float, p_val: torch.Tensor) -> torch.Tensor:
    p1, p2 = p_val[0].item(), p_val[1].item()
    a = 1.0 + p1
    b = p1 * p2
    c_const = 0.5 + b / (a**2 + 1.0)
    y1 = c_const * math.exp(-a * t) + (b * a * math.sin(t) - b * math.cos(t)) / (a**2 + 1.0)
    y2 = p2 * math.sin(t) - y1
    return torch.tensor([y1, y2])


# =====================================================================
# PROBLEM 2: Robertson Stiff Chemical Kinetics DAE (Index-1)
# =====================================================================

def robertson_dae(t, y, yp):
    """
    Standard Robertson problem formulated as an implicit DAE.
    y1, y2: differential variables
    y3: algebraic variable satisfying the conservation equation
    """
    y1, y2, y3 = y[..., 0], y[..., 1], y[..., 2]
    y1p, y2p, _ = yp[..., 0], yp[..., 1], yp[..., 2]
    
    f1 = y1p + 0.04 * y1 - 1e4 * y2 * y3
    f2 = y2p - 0.04 * y1 + 1e4 * y2 * y3 + 3e7 * y2**2
    f3 = y1 + y2 + y3 - 1.0  # Conservation of mass (algebraic equation)
    return torch.stack([f1, f2, f3], dim=-1)


# =====================================================================
# Test Suite Execution
# =====================================================================

def test_radau_correctness():
    torch.set_num_threads(1)
    
    # -----------------------------------------------------------------
    # Test 1: Simple Linear DAE
    # -----------------------------------------------------------------
    print("--------------------------------------------------")
    print("Test 1: Linear DAE (with Analytical Solution)")
    print("--------------------------------------------------")
    p = torch.tensor([1.5, 1.0])
    F_linear = _make_linear_F(p)
    t_span = (0.0, 1.0)
    h_linear = 0.05
    y0_linear = torch.tensor([[0.5, -0.5]])
    
    sol_linear = solve_radau_iia5(
        F=F_linear, t_span=t_span, y0=y0_linear, h=h_linear, step_tol=1e-10, ic_tol=1e-10
    )
    
    y_final_num = sol_linear.ys[-1, 0]
    y_final_ana = _linear_analytical_solution(t_span[1], p)
    error_linear = (y_final_num - y_final_ana).abs().max().item()
    
    print(f"  Radau IIA-5 Final State: {y_final_num.numpy()}")
    print(f"  Analytical Final State:  {y_final_ana.numpy()}")
    print(f"  Absolute Error at t=1.0: {error_linear:.3e}")
    assert error_linear < 1e-6, f"Linear test failed. Error too large: {error_linear:.3e}"
    print("  --> Linear DAE test PASSED")
    
    # -----------------------------------------------------------------
    # Test 2: Robertson Stiff DAE
    # -----------------------------------------------------------------
    print("\n--------------------------------------------------")
    print("Test 2: Robertson Stiff Chemical Kinetics DAE")
    print("--------------------------------------------------")
    
    # Robertson initial conditions
    # y0 satisfies the algebraic constraint: 1.0 + 0.0 + 0.0 = 1.0
    y0_rob = torch.tensor([[1.0, 0.0, 0.0]])
    # Consistent initial derivatives: y1'(0) = -0.04, y2'(0) = 0.04, y3'(0) = 0.0
    yp0_rob = torch.tensor([[-0.04, 0.04, 0.0]])
    
    # Because of the extremely fast initial transient, we use a small step size
    h_rob = 0.001 
    t_span_rob = (0.0, 1.0)
    
    sol_rob = solve_radau_iia5(
        F=robertson_dae,
        t_span=t_span_rob,
        y0=y0_rob,
        yp0=yp0_rob,
        h=h_rob,
        step_tol=1e-12,
        ic_tol=1e-12
    )
    
    y_final_rob = sol_rob.ys[-1, 0]
    
    # Standard high-precision literature values for the Robertson problem at t = 1.0
    ref_y1 = 0.9664597396
    ref_y2 = 0.00003074626588
    ref_y3 = 0.03350951597
    ref_final = torch.tensor([ref_y1, ref_y2, ref_y3])
    
    error_rob = (y_final_rob - ref_final).abs().max().item()
    print(f"  Radau IIA-5 Final State: {y_final_rob.numpy()}")
    print(f"  Reference Final State:   {ref_final.numpy()}")
    print(f"  Absolute Error at t=1.0: {error_rob:.3e}")
    
    # Assert correctness
    assert error_rob < 1e-5, f"Robertson DAE test failed. Error too large: {error_rob:.3e}"
    print("  --> Robertson DAE test PASSED")
    
    # -----------------------------------------------------------------
    # Visualizations
    # -----------------------------------------------------------------
    print("\nGenerating visualization plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Plot 1: Linear DAE
    ts_lin = sol_linear.ts.numpy()
    ax1.plot(ts_lin, sol_linear.ys[:, 0, 0].numpy(), 'b-', label="y1 (Numerical)", linewidth=2)
    ax1.plot(ts_lin, [ _linear_analytical_solution(t, p)[0].item() for t in ts_lin], 'b--', label="y1 (Analytical)")
    ax1.plot(ts_lin, sol_linear.ys[:, 0, 1].numpy(), 'r-', label="y2 (Numerical)", linewidth=2)
    ax1.plot(ts_lin, [ _linear_analytical_solution(t, p)[1].item() for t in ts_lin], 'r--', label="y2 (Analytical)")
    ax1.set_title("Problem 1: Linear DAE Trajectory")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("States")
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Robertson DAE
    ts_rob = sol_rob.ts.numpy()
    # Species y2 has a tiny magnitude (around 1e-5) so we plot it on a separate scale or inspect y1 and y3
    ax2.plot(ts_rob, sol_rob.ys[:, 0, 0].numpy(), 'g-', label="y1 (A - reactant)", linewidth=2)
    ax2.plot(ts_rob, sol_rob.ys[:, 0, 2].numpy(), 'm-', label="y3 (C - product)", linewidth=2)
    
    # Plot y2 with an inset axis or secondary axis since it's very small
    ax2_twin = ax2.twinx()
    ax2_twin.plot(ts_rob, sol_rob.ys[:, 0, 1].numpy(), 'y-', label="y2 (B - intermediate, right scale)", linewidth=1.5)
    ax2_twin.set_ylabel("Intermediate State y2", color='y')
    ax2_twin.tick_params(axis='y', labelcolor='y')
    
    ax2.set_title("Problem 2: Robertson Stiff DAE")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Primary States")
    ax2.grid(True)
    
    # Collect legends from twin axis
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_radau_correctness()

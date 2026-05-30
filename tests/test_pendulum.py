"""
pendulum_simulation.py

Simulates an Index-3 Cartesian Pendulum, showing how Index-Reduction and
Coordinate Projection eliminate numerical constraint drift over time.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib.pyplot as plt


from src.bdf import solve_bdf2
from src.common import DAEFunctions
from src.index_reduction import simplify_dae

# =========================================================================
# 1. DEFINE PHYSICS & CONSTRAINTS
# =========================================================================

def pendulum_physics(t: float, y: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
    """
    y[0], y[1] = x, y coordinates (position)
    y[2], y[3] = u, v velocities
    y[4] = lambda (tension force multiplier)
    """
    x, y_coord, u, v, lam = y[..., 0], y[..., 1], y[..., 2], y[..., 3], y[..., 4]
    xp, yp_coord, up, vp, _ = yp[..., 0], yp[..., 1], yp[..., 2], yp[..., 3], yp[..., 4]
    
    g = 9.81
    res_x = xp - u
    res_y = yp_coord - v
    res_u = up + x * lam
    res_v = vp + y_coord * lam + g
    res_con = x**2 + y_coord**2 - 1.0  # Constraint: x^2 + y^2 - L^2 = 0
    
    return torch.stack([res_x, res_y, res_u, res_v, res_con], dim=-1)


def pendulum_position_constraint(y: torch.Tensor) -> torch.Tensor:
    """Position constraint: g(y) = x^2 + y^2 - 1.0 = 0"""
    return y[..., 0]**2 + y[..., 1]**2 - 1.0


# =========================================================================
# 2. RUN SIMULATIONS
# =========================================================================

if __name__ == "__main__":
    print("Setting up Cartesian Pendulum Simulation...")
    
    t_span = (0.0, 5.0)
    h_step = 0.01
    
    # Starting horizontally (x = 1.0, y = 0.0) with zero velocity
    y0 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    yp0_guess = torch.tensor([[0.0, 0.0, 0.0, -9.81, 0.0]])

    # -------------------------------------------------------------------------
    # RUN 1: Index Reduction only (Baumgarte) -> Prone to coordinate drift
    # -------------------------------------------------------------------------
    print("\n[Run 1] Integrating with Index Reduction (Baumgarte) only...")
    
    # Simplify the Index-3 system to Index-1 via Baumgarte differentiation
    F_reduced_1, y0_reduced_1, yp0_reduced_1 = simplify_dae(
        F=pendulum_physics,
        t0=t_span[0],
        y0=y0,
        yp0=yp0_guess,
        algorithm="differentiation",
        alpha=5.0
    )
    
    problem_no_proj = DAEFunctions(F=F_reduced_1)
    sol_no_proj = solve_bdf2(problem_no_proj, t_span, y0_reduced_1, yp0=yp0_reduced_1, h=h_step)

    # -------------------------------------------------------------------------
    # RUN 2: Index Reduction + Post-Step Coordinate Projection -> Zero Drift
    # -------------------------------------------------------------------------
    print("\n[Run 2] Integrating with Index Reduction + Coordinate Projection...")
    
    problem_with_proj = DAEFunctions(
        F=F_reduced_1,
        constrain_fn=pendulum_position_constraint  # Automatically configures the coordinate projector!
    )
    sol_with_proj = solve_bdf2(problem_with_proj, t_span, y0_reduced_1, yp0=yp0_reduced_1, h=h_step)

    # =========================================================================
    # 3. VISUALIZE RESULTS WITH MATPLOTLIB
    # =========================================================================
    print("\nPlotting results...")
    
    # Extract coordinates
    ts = sol_with_proj.ts.numpy()
    
    # Run 1 coordinates (No projection)
    xs_no_proj = sol_no_proj.ys[:, 0, 0].numpy()
    ys_no_proj = sol_no_proj.ys[:, 0, 1].numpy()
    
    # Run 2 coordinates (With projection)
    xs_with_proj = sol_with_proj.ys[:, 0, 0].numpy()
    ys_with_proj = sol_with_proj.ys[:, 0, 1].numpy()

    # Calculate constraint violations: |x^2 + y^2 - 1|
    drift_no_proj = [abs(x**2 + y**2 - 1.0) for x, y in zip(xs_no_proj, ys_no_proj)]
    drift_with_proj = [abs(x**2 + y**2 - 1.0) for x, y in zip(xs_with_proj, ys_with_proj)]

    # Set up matplotlib figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: The Pendulum Trajectory (circular arc)
    ax1.plot(xs_no_proj, ys_no_proj, linestyle="dashed", color="red", label="Without Projection (Baumgarte)")
    ax1.plot(xs_with_proj, ys_with_proj, color="blue", label="With Post-Step Projection")
    ax1.scatter([0.0], [0.0], color="black", marker="o", s=100, label="Anchor Point (0,0)")
    ax1.set_aspect("equal")
    ax1.set_title("Pendulum Trajectory (x-y Path)")
    ax1.set_xlabel("x (meters)")
    ax1.set_ylabel("y (meters)")
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Constraint Drift over time
    ax2.plot(ts, drift_no_proj, color="red", label="Without Projection (Baumgarte)")
    ax2.plot(ts, drift_with_proj, color="blue", label="With Post-Step Projection")
    ax2.set_yscale("log")
    ax2.set_title("Constraint Violation Drift Over Time (Log Scale)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Rope Length Drift |x^2 + y^2 - L^2|")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

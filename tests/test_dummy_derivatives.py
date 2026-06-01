"""
test_dummy_derivatives.py

Verifies the Mattsson-Söderlind Dummy Derivative index-reduction algorithm on
an Index-3 Cartesian pendulum system. Generates verification plots for state
trajectories and algebraic constraint preservation.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import traceback
import matplotlib.pyplot as plt

from torchdae.bdf import solve_bdf1
from torchdae.index_reduction import simplify_dae, analyze_pytorch_dae


def _make_cartesian_pendulum(m=1.0, g_acc=9.81, L=1.0):
    """
    Defines a 2D pendulum in Cartesian coordinates as an Index-3 DAE system.
    State vector: y = [x, y, u, v, lam]
    where:
      - x, y: position coordinates
      - u, v: velocity coordinates (u = x', v = y')
      - lam: Lagrange multiplier (constraint force / tension)
    """
    def F(t, y, yp):
        x, y_coord = y[..., 0], y[..., 1]
        u, v = y[..., 2], y[..., 3]
        lam = y[..., 4]

        xp, yp_coord = yp[..., 0], yp[..., 1]
        up, vp = yp[..., 2], yp[..., 3]

        f1 = xp - u
        f2 = yp_coord - v
        f3 = up + x * lam
        f4 = vp + y_coord * lam + g_acc
        f5 = x**2 + y_coord**2 - L**2  # Position constraint (Index-3)

        return torch.stack([f1, f2, f3, f4, f5], dim=-1)
    
    return F


def run_dummy_derivative_test():
    # Force single-threaded CPU execution for consistency
    torch.set_num_threads(1)
    
    print("==================================================")
    print("Testing DAE Index Reduction: Dummy Derivative Mode")
    print("==================================================")

    m, g_acc, L = 1.0, 9.81, 1.0
    F_orig = _make_cartesian_pendulum(m, g_acc, L)

    t_span = (0.0, 2.0)
    h_step = 0.01

    y0 = torch.tensor([[L, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    yp0 = torch.tensor([[0.0, 0.0, 0.0, -g_acc, 0.0]], dtype=torch.float64)

    # 1. Run structural analysis to verify dummy variables are selected
    print("Running structural analysis...")
    structure = analyze_pytorch_dae(F_orig, t_span[0], y0, yp0, algorithm="dummy_derivative")
    
    print(f"  Total variables (N): {structure.n_vars}")
    print(f"  Equation differentiation orders: {structure.differentiation_orders}")
    print(f"  Dummy derivative variables (indices): {structure.dummy_derivatives}")
    
    assert len(structure.dummy_derivatives) > 0, "No dummy derivative variables were identified."

    # 2. Simplify the DAE using the Mattsson-Söderlind method
    # This automatically builds the augmented state Z0 = [y0, u0] of size N + K
    print("\nSimplifying DAE using Dummy Derivative Method...")
    F_reduced, Z0, Zp0 = simplify_dae(
        F_orig, t_span[0], y0, yp0, algorithm="dummy_derivative"
    )
    
    print(f"  Augmented initial state Z0 shape: {Z0.shape}")
    print(f"  Augmented initial derivative Zp0 shape: {Zp0.shape}")

    # 3. Integrate the augmented system using solve_bdf1
    print("\nIntegrating stabilized Index-1 DAE...")
    try:
        sol = solve_bdf1(F_reduced, t_span, Z0, h=h_step)
        print("  --> Integration completed successfully.")
    except Exception:
        print("  [!] Integration FAILED:")
        traceback.print_exc()
        return

    # Extract original states from the augmented trajectory Z(t)
    ts = sol.ts.numpy()
    # Z has shape (T, Batch, N + K). Indexing [:, 0] gets the single batch element
    Z_traj = sol.ys[:, 0]
    
    xs = Z_traj[:, 0].numpy()
    ys = Z_traj[:, 1].numpy()
    lams = Z_traj[:, 4].numpy()
    
    # Extract the auxiliary dummy derivatives (u) stored in the augmented portion of Z
    dummy_u_states = Z_traj[:, structure.n_vars:].numpy()

    # 4. Verify constraint satisfaction
    # Under MS-reduction, the constraint violation should remain extremely small (~1e-15)
    constraint_violations = xs**2 + ys**2 - L**2
    max_violation = max(abs(val) for val in constraint_violations)
    
    print("\nVerification:")
    print(f"  Maximum position constraint violation: {max_violation:.3e} (meters)")
    assert max_violation < 1e-9, f"Position constraint drifted significantly: {max_violation:.3e}"
    print("  --> Constraint preservation check PASSED")

    # 5. Generate visualization plots
    print("\nGenerating visualization plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Trajectory coordinates of the pendulum
    ax1.plot(ts, xs, 'b-', label="X Position (differential)", linewidth=2)
    ax1.plot(ts, ys, 'r-', label="Y Position (differential)", linewidth=2)
    ax1.plot(ts, lams, 'm--', label="Lagrange Multiplier (tension)", linewidth=1.5)
    ax1.set_title("1. Pendulum States & Multiplier")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("States / Multipliers")
    ax1.grid(True)
    ax1.legend()

    # Subplot 2: Auxiliary dummy derivatives and constraint check
    for idx, dummy_idx in enumerate(structure.dummy_derivatives):
        ax2.plot(ts, dummy_u_states[:, idx], label=f"Dummy derivative u_{dummy_idx}", linestyle="--")
        
    ax2_twin = ax2.twinx()
    ax2_twin.plot(ts, constraint_violations, 'k:', label="Constraint violation (right axis)", linewidth=1.5)
    ax2_twin.set_ylabel("Constraint Violation (meters)", color='k')
    ax2_twin.tick_params(axis='y', labelcolor='k')
    
    ax2.set_title("2. Dummy Derivatives & Constraint Violation")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Dummy Variables")
    ax2.grid(True)
    
    # Merge legends from secondary axis
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_dummy_derivative_test()

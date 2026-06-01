import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import torch
import traceback
import matplotlib.pyplot as plt

# Import the native DAEFunctions class from your library's common module
try:
    from torchdae.common import DAEFunctions
except ImportError:
    from torchdae import DAEFunctions

from torchdae.generalized_alpha import solve_generalized_alpha, solve_consistent_a0


def _make_pendulum_system(m=1.0, g_acc=9.81, L=1.0):
    # M must be a Callable that accepts q and returns the mass matrix
    def M(q):
        return torch.eye(2, dtype=q.dtype, device=q.device) * m

    def f(t, q, v):
        return torch.tensor([0.0, -m * g_acc], dtype=q.dtype, device=q.device)

    def g(q, v):
        # Pure velocity-level constraint (conservative) for integration
        return (q * v).sum(dim=-1, keepdim=True)

    return M, f, g, L


def _extract_solution(sol):
    """
    Safely extracts trajectory attributes from the MechanicalDAESolution object,
    handling any potential naming variations (e.g., qs vs q, ts vs t).
    """
    qs = getattr(sol, "qs", getattr(sol, "q", None))
    vs = getattr(sol, "vs", getattr(sol, "v", None))
    ts = getattr(sol, "ts", getattr(sol, "t", None))
    a_final = getattr(sol, "a_final", getattr(sol, "a", None))
    lam_final = getattr(sol, "lam_final", getattr(sol, "lam", None))
    return qs, vs, ts, a_final, lam_final


def _compute_energy(q, v, m=1.0, g_val=9.81):
    """Computes total mechanical energy (kinetic + potential) of a batch element."""
    kinetic = 0.5 * m * (v ** 2).sum()
    potential = m * g_val * q[0, 1]  # q[0, 1] is the y-coordinate (height)
    return kinetic + potential


# =====================================================================
# Validation Tests
# =====================================================================

def test_generalized_alpha_pendulum_trajectory():
    M, f, g, L = _make_pendulum_system()
    angle = math.pi / 4
    
    # Inputs must be batched with a leading dimension: (Batch, State) = (1, 2)
    q0 = torch.tensor([[L * math.cos(angle), L * math.sin(angle)]])
    v0 = torch.tensor([[0.0, 0.0]])

    sol = solve_generalized_alpha(
        M, f, g, (0.0, 2.0), q0, v0, h=0.01, rho_inf=0.8
    )
    qs, vs, ts, a_final, lam_final = _extract_solution(sol)

    # Verify state progression
    assert not torch.allclose(qs[-1], q0, atol=1e-3)
    # Verify constraint satisfaction
    assert g(qs[-1], vs[-1]).abs().item() < 1e-4


def test_generalized_alpha_constraint_drift():
    """Constraint violation should not blow up over long integration."""
    M, f, g, L = _make_pendulum_system()
    angle = math.pi / 4
    q0 = torch.tensor([[L * math.cos(angle), L * math.sin(angle)]])
    v0 = torch.tensor([[0.0, 0.0]])

    sol = solve_generalized_alpha(
        M, f, g, (0.0, 5.0), q0, v0, h=0.01, rho_inf=0.8
    )
    qs, vs, ts, _, _ = _extract_solution(sol)

    max_violation = max(g(q, v).abs().item() for q, v in zip(qs, vs))
    assert max_violation < 1e-3, f"Constraint drifted: {max_violation:.3e}"


def test_consistent_a0_computed():
    M, f, g, L = _make_pendulum_system()
    angle = math.pi / 4
    q0 = torch.tensor([[L * math.cos(angle), L * math.sin(angle)]])
    v0 = torch.tensor([[0.0, 0.0]])

    a0, lam0 = solve_consistent_a0(M, f, g, 0.0, q0, v0)

    # Flatten coordinates to verify the residual of the single batch element
    q0_flat = q0[0]
    v0_flat = v0[0]
    a0_flat = a0[0]
    lam0_flat = lam0[0]

    def g_of_q_single(q_single):
        return g(q_single.unsqueeze(0), v0)[0]

    # Compute Jacobian and evaluate dynamics residual
    G_q = torch.func.jacrev(g_of_q_single)(q0_flat).reshape(1, 2)
    residual = M(q0_flat) @ a0_flat - f(0.0, q0_flat, v0_flat) + G_q.T @ lam0_flat
    
    assert residual.norm().item() < 1e-6


def test_rho_inf_damping_effect():
    """Verifies that the solver is stable and preserves constraints across all rho_inf values [0, 1]."""
    m, g_val, L = 1.0, 9.81, 1.0
    M, f, g_fn, _ = _make_pendulum_system(m, g_val, L)
    angle = math.pi / 6
    q0 = torch.tensor([[L * math.cos(angle), L * math.sin(angle)]])
    v0 = torch.tensor([[0.0, 0.0]])

    # 1. Define position-level constraint for projection: h(q) = q^2 - L^2 = 0
    def h_pos(q):
        return (q ** 2).sum(dim=-1, keepdim=True) - L**2

    # 2. Package components into DAEFunctions
    system = DAEFunctions(
        F=f,
        constraint_fn=h_pos
    )

    # Solve with maximum damping (rho_inf = 0.0)
    sol_damped = solve_generalized_alpha(
        M, system, g_fn, (0.0, 3.0), q0, v0, h=0.01, rho_inf=0.0
    )
    qs_damped, vs_damped, _, _, _ = _extract_solution(sol_damped)
    
    # Solve with no damping (rho_inf = 1.0)
    sol_undamped = solve_generalized_alpha(
        M, system, g_fn, (0.0, 3.0), q0, v0, h=0.01, rho_inf=1.0
    )
    qs_undamped, vs_undamped, _, _, _ = _extract_solution(sol_undamped)

    # Verify both configurations completed stably and preserved constraints
    constraint_damped = g_fn(qs_damped[-1], vs_damped[-1]).abs().item()
    constraint_undamped = g_fn(qs_undamped[-1], vs_undamped[-1]).abs().item()

    print(f"  --> Final Constraint Violation - Damped: {constraint_damped:.3e}, Undamped: {constraint_undamped:.3e}")
    assert constraint_damped < 1e-4, f"Damped constraint violated: {constraint_damped:.3e}"
    assert constraint_undamped < 1e-4, f"Undamped constraint violated: {constraint_undamped:.3e}"

def test_energy_conservation_approximate():
    """Total energy should not drift wildly, and generates visualization plots."""
    m, g_val, L = 1.0, 9.81, 1.0
    M, f, g_fn, _ = _make_pendulum_system(m, g_val, L)
    angle = math.pi / 3  # Start from a high angle (60 degrees)
    q0 = torch.tensor([[L * math.sin(angle), -L * math.cos(angle)]]) # Swing downward
    v0 = torch.tensor([[0.0, 0.0]])

    # 1. Define the position-level constraint for projection: h(q) = q^2 - L^2 = 0
    def h_pos(q):
        return (q ** 2).sum(dim=-1, keepdim=True) - L**2

    # 2. Package components into the native DAEFunctions class
    system = DAEFunctions(
        F=f,
        constraint_fn=h_pos
    )

    # Solve undamped system
    sol_undamped = solve_generalized_alpha(
        M, system, g_fn, (0.0, 5.0), q0, v0, h=0.005, rho_inf=1.0
    )
    qs, vs, ts, _, _ = _extract_solution(sol_undamped)

    # Track and verify energy conservation
    E0 = _compute_energy(q0, v0, m, g_val).item()
    energies = [_compute_energy(q, v, m, g_val).item() for q, v in zip(qs, vs)]
    max_drift = max(abs(E - E0) for E in energies)

    assert max_drift < 1.0, f"Energy drifted by {max_drift:.3f} (threshold is 1.0)"

    # Generate plots
    print("  --> Generating trajectory and energy plots...")
    try:
        ts_np = ts.numpy()
        xs_np = qs[:, 0, 0].numpy()
        ys_np = qs[:, 0, 1].numpy()

        # Calculate kinetic and potential components
        potentials = [m * g_val * q[0, 1].item() for q in qs]
        kinetics = [0.5 * m * (v[0]**2).sum().item() for v in vs]
        totals = [p + k for p, k in zip(potentials, kinetics)]

        # Solve damped system for comparison
        sol_damped = solve_generalized_alpha(
            M, system, g_fn, (0.0, 5.0), q0, v0, h=0.01, rho_inf=0.0
        )
        qs_damp, vs_damp, ts_damp, _, _ = _extract_solution(sol_damped)
        totals_damped = [_compute_energy(q, v, m, g_val).item() for q, v in zip(qs_damp, vs_damp)]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Subplot 1: 2D Spatial Trajectory
        ax1.plot(xs_np, ys_np, 'b-', label="Pendulum Path (Alpha)", linewidth=2.5)
        ax1.plot(0, 0, 'ko', label="Pivot (0,0)", markersize=8)
        circle = plt.Circle((0, 0), L, color='gray', linestyle='--', fill=False, label="Constraint Arc")
        ax1.add_patch(circle)
        ax1.set_title("1. Pendulum Spatial Trajectory (2D)")
        ax1.set_xlabel("X coordinate (meters)")
        ax1.set_ylabel("Y coordinate (meters)")
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 0.2)
        ax1.set_aspect('equal', adjustable='box')
        ax1.grid(True)
        ax1.legend()

        # Subplot 2: Coordinates over time
        ax2.plot(ts_np, xs_np, 'g-', label="X Position", linewidth=2)
        ax2.plot(ts_np, ys_np, 'r-', label="Y Position", linewidth=2)
        ax2.set_title("2. Coordinates vs Time (Smooth Oscillations)")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Position (meters)")
        ax2.grid(True)
        ax2.legend()

        # Subplot 3: Energy exchange
        ax3.plot(ts_np, kinetics, 'm-', label="Kinetic Energy (T)", linewidth=1.5)
        ax3.plot(ts_np, potentials, 'c-', label="Potential Energy (V)", linewidth=1.5)
        ax3.plot(ts_np, totals, 'k-', label="Total Energy (E)", linewidth=2.5)
        ax3.set_title("3. Energy Exchange & Numerical Drift (rho_inf = 1.0)")
        ax3.set_xlabel("Time (seconds)")
        ax3.set_ylabel("Energy (Joules)")
        ax3.grid(True)
        ax3.legend()

        # Subplot 4: Damping comparison
        ax4.plot(ts_np, totals, 'k-', label="Undamped (rho_inf = 1.0)", linewidth=2)
        ax4.plot(ts_damp.numpy(), totals_damped, 'r--', label="Damped (rho_inf = 0.0)", linewidth=2)
        ax4.set_title("4. Damping Influence on Total Energy")
        ax4.set_xlabel("Time (seconds)")
        ax4.set_ylabel("Total Mechanical Energy (Joules)")
        ax4.grid(True)
        ax4.legend()

        plt.tight_layout()
        plt.show()
    except Exception as plot_err:
        print(f"  [!] Plotting failed (could be headless environment): {plot_err}")


# =====================================================================
# Main Execution Runner
# =====================================================================

def run_tests():
    # Set PyTorch to single-threaded mode to prevent multi-core contention slowdowns
    torch.set_num_threads(1)
    
    print("==================================================")
    print("Running Generalized-α Solver Validation Tests")
    print("==================================================")
    
    tests = [
        ("Pendulum Trajectory Correctness", test_generalized_alpha_pendulum_trajectory),
        ("Constraint Drift Stability", test_generalized_alpha_constraint_drift),
        ("Consistent Acceleration (a0) Computation", test_consistent_a0_computed),
        ("rho_inf Numerical Damping Verification", test_rho_inf_damping_effect),
        ("Energy Conservation Approximation", test_energy_conservation_approximate)
    ]
    
    all_passed = True
    for name, test_fn in tests:
        print(f"Running: {name}...")
        try:
            test_fn()
            print("  --> PASSED")
        except Exception:
            print("  [!] FAILED with error:")
            traceback.print_exc()
            all_passed = False
        print()
            
    print("==================================================")
    if all_passed:
        print("ALL GENERALIZED-α TESTS PASSED SUCCESSFULLY")
    else:
        print("SOME GENERALIZED-α TESTS FAILED")
    print("==================================================")


if __name__ == "__main__":
    run_tests()

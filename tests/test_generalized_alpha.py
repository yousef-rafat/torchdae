import math
import torch
from generalized_alpha import solve_generalized_alpha, solve_consistent_a0


def _make_pendulum_system(m=1.0, g=9.81, L=1.0):
    M = torch.eye(2) * m

    def f(t, q, v):
        return torch.tensor([0.0, -m * g])

    def g(q, v):
        # Velocity-level constraint: x*vx + y*vy = 0
        return (q * v).sum(dim=-1, keepdim=True)

    return M, f, g, L


def test_generalized_alpha_pendulum_trajectory():
    M, f, g, L = _make_pendulum_system()
    angle = math.pi / 4
    q0 = torch.tensor([L * math.cos(angle), L * math.sin(angle)])
    v0 = torch.tensor([0.0, 0.0])

    qs, vs, ts, a_final, lam_final = solve_generalized_alpha(
        M, f, g, (0.0, 2.0), q0, v0, h=0.01, rho_inf=0.8
    )

    # Should have moved
    assert not torch.allclose(qs[-1], q0, atol=1e-3)
    # Constraint should stay small
    assert g(qs[-1], vs[-1]).abs().item() < 1e-4


def test_generalized_alpha_constraint_drift():
    """Constraint violation should not blow up over long integration."""
    M, f, g, L = _make_pendulum_system()
    angle = math.pi / 4
    q0 = torch.tensor([L * math.cos(angle), L * math.sin(angle)])
    v0 = torch.tensor([0.0, 0.0])

    qs, vs, ts, _, _ = solve_generalized_alpha(
        M, f, g, (0.0, 5.0), q0, v0, h=0.01, rho_inf=0.8
    )

    max_violation = max(g(q, v).abs().item() for q, v in zip(qs, vs))
    assert max_violation < 1e-3, f"Constraint drifted: {max_violation:.3e}"


def test_consistent_a0_computed():
    M, f, g, L = _make_pendulum_system()
    angle = math.pi / 4
    q0 = torch.tensor([L * math.cos(angle), L * math.sin(angle)])
    v0 = torch.tensor([0.0, 0.0])

    a0, lam0 = solve_consistent_a0(M, f, g, 0.0, q0, v0)

    # Check dynamics residual: M*a0 + G^T*lam0 - f should be ~0
    def g_of_q(q):
        return g(q, v0)

    G_q = torch.func.jacrev(g_of_q)(q0).reshape(1, 2)
    residual = M @ a0.flatten() - f(0.0, q0, v0).flatten() + G_q.T @ lam0.flatten()
    assert residual.norm().item() < 1e-6


def test_rho_inf_damping_effect():
    """Lower rho_inf = more numerical damping = less oscillation amplitude."""
    M, f, g, L = _make_pendulum_system()
    angle = math.pi / 6
    q0 = torch.tensor([L * math.cos(angle), L * math.sin(angle)])
    v0 = torch.tensor([0.0, 0.0])

    # High damping
    qs_damped, _, _, _, _ = solve_generalized_alpha(
        M, f, g, (0.0, 3.0), q0, v0, h=0.01, rho_inf=0.0
    )
    # Low damping
    qs_undamped, _, _, _, _ = solve_generalized_alpha(
        M, f, g, (0.0, 3.0), q0, v0, h=0.01, rho_inf=1.0
    )

    # Measure max amplitude of y-coordinate
    y_max_damped = qs_damped[:, 1].abs().max().item()
    y_max_undamped = qs_undamped[:, 1].abs().max().item()

    assert y_max_damped < y_max_undamped, "Damping should reduce amplitude"


def test_energy_conservation_approximate():
    """Total energy should not drift wildly."""
    m, g, L = 1.0, 9.81, 1.0
    M, f, g_fn, _ = _make_pendulum_system(m, g, L)
    angle = math.pi / 4
    q0 = torch.tensor([L * math.cos(angle), L * math.sin(angle)])
    v0 = torch.tensor([0.0, 0.0])

    qs, vs, _, _, _ = solve_generalized_alpha(
        M, f, g_fn, (0.0, 2.0), q0, v0, h=0.005, rho_inf=1.0
    )

    def energy(q, v):
        kinetic = 0.5 * m * (v ** 2).sum()
        potential = m * g * q[1]  # y is height
        return kinetic + potential

    E0 = energy(q0, v0).item()
    energies = [energy(q, v).item() for q, v in zip(qs, vs)]
    max_drift = max(abs(E - E0) for E in energies)

    assert max_drift < 0.5, f"Energy drifted by {max_drift:.3f}"


if __name__ == "__main__":
    test_generalized_alpha_pendulum_trajectory()
    test_generalized_alpha_constraint_drift()
    test_consistent_a0_computed()
    test_rho_inf_damping_effect()
    test_energy_conservation_approximate()
    print("\nAll Generalized-α tests passed.")

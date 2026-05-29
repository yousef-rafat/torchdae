import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import torch
import torch.func as func

from src.bdf import solve_bdf1, solve_bdf2, solve_tr_bdf2, solve_consistent_yp0
from src.algorithms import compute_consistent_initial_conditions


def _make_F():
    """Shared residual for index-1 test problem."""
    def F(t, y, yp):
        y1, y2 = y[..., 0], y[..., 1]
        y1p, _ = yp[..., 0], yp[..., 1]
        f1 = y1p + y1 - y2
        f2 = y1 + y2 - torch.sin(t)
        return torch.stack([f1, f2], dim=-1)
    return F


def _analytical_solution(t: float) -> torch.Tensor:
    y1 = (2 * math.sin(t) - math.cos(t) + math.exp(-2 * t)) / 5
    y2 = (3 * math.sin(t) + math.cos(t) - math.exp(-2 * t)) / 5
    return torch.tensor([y1, y2])


def test_bdf1_analytical():
    F = _make_F()
    y0 = torch.tensor([0.0, 0.0])
    sol = solve_bdf1(F, (0.0, 1.0), y0, h=0.001, strict=True)

    y_final = sol.ys[-1]
    y_exact = _analytical_solution(1.0)
    err = (y_final - y_exact).abs().max().item()

    assert err < 1e-3, f"BDF1 error too large: {err:.3e}"


def test_tr_bdf2_analytical():
    """Test that SDIRK TR-BDF2 integrates correctly."""
    F = _make_F()
    y0 = torch.tensor([0.0, 0.0])
    sol = solve_tr_bdf2(F, (0.0, 1.0), y0, h=0.001, strict=True)

    y_final = sol.ys[-1]
    y_exact = _analytical_solution(1.0)
    err = (y_final - y_exact).abs().max().item()

    assert err < 1e-4, f"TR-BDF2 error too large: {err:.3e}"


def test_bdf2_more_accurate_than_bdf1():
    """BDF2 should be ~O(h^2) vs BDF1's O(h)."""
    F = _make_F()
    y0 = torch.tensor([0.0, 0.0])
    h = 0.01

    sol1 = solve_bdf1(F, (0.0, 1.0), y0, h=h, strict=True)
    sol2 = solve_bdf2(F, (0.0, 1.0), y0, h=h, strict=True)

    y_exact = _analytical_solution(1.0)
    err1 = (sol1.ys[-1] - y_exact).abs().max().item()
    err2 = (sol2.ys[-1] - y_exact).abs().max().item()

    assert err2 < err1 / 10, f"BDF2 ({err2:.3e}) not significantly better than BDF1 ({err1:.3e})"


def test_bdf1_vs_bdf2_convergence():
    """Both should converge, BDF2 faster."""
    F = _make_F()
    y0 = torch.tensor([0.0, 0.0])

    errors_bdf1 = []
    errors_bdf2 = []
    errors_tr_bdf2 = []
    hs = [0.02, 0.01, 0.005]

    for h in hs:
        sol1 = solve_bdf1(F, (0.0, 1.0), y0, h=h, strict=True)
        sol2 = solve_bdf2(F, (0.0, 1.0), y0, h=h, strict=True)
        sol3 = solve_tr_bdf2(F, (0.0, 1.0), y0, h=h, strict=True)
        y_exact = _analytical_solution(1.0)

        errors_bdf1.append((sol1.ys[-1] - y_exact).abs().max().item())
        errors_bdf2.append((sol2.ys[-1] - y_exact).abs().max().item())
        errors_tr_bdf2.append((sol3.ys[-1] - y_exact).abs().max().item())

    # BDF2 error should roughly quarter when h halves (O(h^2))
    ratio_bdf2 = errors_bdf2[0] / errors_bdf2[1]
    assert 3.0 < ratio_bdf2 < 5.0, f"BDF2 convergence ratio {ratio_bdf2:.2f} not ~4"

    # TR-BDF2 error should also roughly quarter when h halves (O(h^2))
    ratio_tr_bdf2 = errors_tr_bdf2[0] / errors_tr_bdf2[1]
    assert 3.0 < ratio_tr_bdf2 < 5.0, f"TR-BDF2 convergence ratio {ratio_tr_bdf2:.2f} not ~4"


def test_inconsistent_ic_raises():
    F = _make_F()
    y0 = torch.tensor([1.0, 1.0])  # violates constraint

    try:
        solve_bdf1(F, (0.0, 1.0), y0, h=0.01, strict=True)
        assert False, "ValueError was expected, but not raised"
    except ValueError:
        pass


def test_consistent_yp0_computed():
    F = _make_F()
    y0 = torch.tensor([0.0, 0.0])
    sol = solve_bdf1(F, (0.0, 0.01), y0, h=0.01, strict=False)
    yp0 = sol.yp_final

    residual = F(0.0, y0, yp0)
    norm = residual.norm().item()
    assert norm < 1e-6, f"Consistent yp0 residual: {norm:.3e}"


def testsolve_consistent_yp0_directly():
    """Test the private solve_consistent_yp0 function."""
    F = _make_F()
    y0 = torch.tensor([0.0, 0.0])
    yp0 = solve_consistent_yp0(F, 0.0, y0, tol=1e-10)

    residual = F(0.0, y0, yp0)
    assert residual.norm().item() < 1e-8


def test_compute_consistent_initial_conditions_directly():
    """Test the public compute_consistent_initial_conditions function."""
    F = _make_F()
    y0 = torch.tensor([0.1, 0.1])  # Slightly inconsistent coordinate
    y0_cons, yp0_cons = compute_consistent_initial_conditions(F, 0.0, y0, tol=1e-10)

    residual = F(0.0, y0_cons, yp0_cons)
    assert residual.norm().item() < 1e-8


def test_bdf1_batched_vmap():
    """Test that vmap works over initial conditions."""
    F = _make_F()

    def solve_one(y0):
        sol = solve_bdf1(F, (0.0, 0.1), y0, h=0.01, strict=True)
        return sol.ys[-1]

    y0_batch = torch.stack([
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
    ])
    results = func.vmap(solve_one)(y0_batch)
    assert results.shape == (2, 2)


def test_bdf1_batched_direct():
    """Test that solvers handle native batch dimensions correctly."""
    F = _make_F()
    y0_batch = torch.stack([
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
    ])
    sol = solve_bdf1(F, (0.0, 0.1), y0_batch, h=0.01, strict=True)
    
    # Expected output: 11 time steps, 2 batch elements, 2 state variables
    assert sol.ys.shape == (11, 2, 2)


if __name__ == "__main__":
    test_bdf1_analytical()
    test_tr_bdf2_analytical()
    test_bdf2_more_accurate_than_bdf1()
    test_bdf1_vs_bdf2_convergence()
    test_inconsistent_ic_raises()
    test_consistent_yp0_computed()
    testsolve_consistent_yp0_directly()
    test_compute_consistent_initial_conditions_directly()
    test_bdf1_batched_vmap()
    test_bdf1_batched_direct()
    print("\nAll BDF and TR-BDF2 tests passed successfully.")

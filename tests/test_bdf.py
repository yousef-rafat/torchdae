import math
import torch
from bdf import solve_bdf1, solve_bdf2
from common import solve_consistent_yp0


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
    ys, ts, yp0 = solve_bdf1(F, (0.0, 1.0), y0, h=0.001, strict=True)

    y_final = ys[-1]
    y_exact = _analytical_solution(1.0)
    err = (y_final - y_exact).abs().max().item()

    assert err < 1e-3, f"BDF1 error too large: {err:.3e}"


def test_bdf2_more_accurate_than_bdf1():
    """BDF2 should be ~O(h^2) vs BDF1's O(h)."""
    F = _make_F()
    y0 = torch.tensor([0.0, 0.0])
    h = 0.01

    ys1, _, _ = solve_bdf1(F, (0.0, 1.0), y0, h=h, strict=True)
    ys2, _, _ = solve_bdf2(F, (0.0, 1.0), y0, h=h, strict=True)

    y_exact = _analytical_solution(1.0)
    err1 = (ys1[-1] - y_exact).abs().max().item()
    err2 = (ys2[-1] - y_exact).abs().max().item()

    assert err2 < err1 / 10, f"BDF2 ({err2:.3e}) not significantly better than BDF1 ({err1:.3e})"


def test_bdf1_vs_bdf2_convergence():
    """Both should converge, BDF2 faster."""
    F = _make_F()
    y0 = torch.tensor([0.0, 0.0])

    errors_bdf1 = []
    errors_bdf2 = []
    hs = [0.02, 0.01, 0.005]

    for h in hs:
        ys1, _, _ = solve_bdf1(F, (0.0, 1.0), y0, h=h, strict=True)
        ys2, _, _ = solve_bdf2(F, (0.0, 1.0), y0, h=h, strict=True)
        y_exact = _analytical_solution(1.0)

        errors_bdf1.append((ys1[-1] - y_exact).abs().max().item())
        errors_bdf2.append((ys2[-1] - y_exact).abs().max().item())

    # BDF2 error should roughly quarter when h halves (O(h^2))
    ratio = errors_bdf2[1] / errors_bdf2[0]
    assert 3.0 < ratio < 5.0, f"BDF2 convergence ratio {ratio:.2f} not ~4"


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
    _, _, yp0 = solve_bdf1(F, (0.0, 0.01), y0, h=0.01, strict=False)

    residual = F(0.0, y0, yp0)
    norm = residual.norm().item()
    assert norm < 1e-6, f"Consistent yp0 residual: {norm:.3e}"


def test_solve_consistent_yp0_directly():
    """Test the common.py function without going through solve_bdf1."""
    F = _make_F()
    y0 = torch.tensor([0.0, 0.0])
    yp0 = solve_consistent_yp0(F, 0.0, y0, tol=1e-10)

    residual = F(0.0, y0, yp0)
    assert residual.norm().item() < 1e-8


def test_bdf1_batched():
    """Test that vmap works over initial conditions."""
    F = _make_F()

    def solve_one(y0):
        ys, _, _ = solve_bdf1(F, (0.0, 0.1), y0, h=0.01, strict=True)
        return ys[-1]

    y0_batch = torch.stack([
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
    ])
    results = torch.vmap(solve_one)(y0_batch)
    assert results.shape == (2, 2)


if __name__ == "__main__":
    test_bdf1_analytical()
    test_bdf2_more_accurate_than_bdf1()
    test_bdf1_vs_bdf2_convergence()
    test_inconsistent_ic_raises()
    test_consistent_yp0_computed()
    test_solve_consistent_yp0_directly()
    test_bdf1_batched()
    print("\nAll BDF tests passed.")

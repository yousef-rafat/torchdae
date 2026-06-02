# Index-1 Solvers

Index-1 in `torchdae`:
* **BDF1 (Backward Euler)**: First-order, highly dissipative and stable.
* **BDF2**: Second-order, general-purpose solver.
* **TR-BDF2**: One-step composite solver mixing the Trapezoidal Rule and BDF2.
* **Radau IIA-5**: 3-stage, 5th-order fully implicit solver with high-order L-stability.

This page contain explanation for the input parameters for all of these solvers.

> A lot of these of parameters are shared with the [Higher Index Solvers](usage/higher_index_solvers.md)

## Input Parameters

### `damping` (default: `1.0`)
A scalar value used in newton solve function to help make convergence safer

### `strategy` (default: `"freeze_dynamic"`)

The type of strategy the jacobian will take.

* `always`: Always compute the jacobian at every step
* `freeze_dynamic`: Calculates the Jacobian based on the convergence and divergence
* `freeze_step`: Calculate the first jacobian and reuse

### `recompute_every` (default: `None`)

When the `freeze_dynamic` jacobian strategy is choosen, it controls when to recalculate the jacobian after X steps, overwriting the native convergence controller.

### `args` (default: `None`)

Additional arguments that goes with the main function `F`.

### `max_iter_for_events` (default: `100`)

In case of an event happening, the parameter controls how many times the correction loop runs

## Coding Example

```python
import torch
from torchdae import solve_bdf2

# a physics function that accept more arguments
def physical_system(t, y, yp, g_acc, damping_coeff):
    y1, y2 = y[..., 0], y[..., 1]
    y1p, y2p = yp[..., 0], yp[..., 1]
    
    # Differential equation incorporating the damping coefficient
    f1 = y1p + damping_coeff * y1 - y2
    
    # Algebraic constraint incorporating gravity
    t_tensor = torch.as_tensor(t, dtype=y.dtype, device=y.device)
    f2 = y1 + y2 - g_acc * torch.sin(t_tensor)
    return torch.stack([f1, f2], dim=-1)

# Initial conditions
y0 = torch.tensor([[0.5, -0.5]])

sol = solve_bdf2(
    F=physical_system,
    t_span=(0.0, 1.0),
    y0=y0,
    h=0.01,
    args=(9.81, 0.5),             # Passes (g_acc, damping_coeff) to physics
    damping=0.8,                  # Newton step damping to prevent divergence
    strategy="freeze_dynamic",    # Freeze Jacobian within each step
    recompute_every=3             # Reuse Jacobian across 3 steps
)

print("Simulation complete. Steps executed:", len(sol.ts))
```

# Getting Started

This page is made to help you get started with solving custom problems with `torchdae`

`torchdae` expects the input to be an implicit function returning a residual to be optimized to zero.

The solvers that are supported in the library are: 

* **First-Order DAE Solvers**: BDF1, BDF2, SDIRK TR-BDF2, and 5th-order Radau IIA.
* **Second-Order Mechanical Solvers**: Generalized-$\alpha$.

## Defining a function

Function must have all pytorch operations in the implicit form of `f(t, y0, yp0) = 0`. An index-1 example:

```python
import torch

def my_dae(t: float, y: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
    """
    Implements the implicit residual F(t, y, yp) = 0.
    """
    # unpack states and derivatives while preserving the batch dimension
    y1, y2 = y[..., 0], y[..., 1]
    y1p, _ = yp[..., 0], yp[..., 1]
    
    # define the differential equation: y1' + y1 - y2 = 0
    f1 = y1p + y1 - y2
    
    # define the algebraic constraint: y1 + y2 - sin(t) = 0
    t_tensor = torch.as_tensor(t, dtype=y.dtype, device=y.device)
    f2 = y1 + y2 - torch.sin(t_tensor)
    
    # stack and return the residual vector
    return torch.stack([f1, f2], dim=-1)
```

## Consistent yp0 initialization

In case an the guess is outside of the allowed tolerance is detected, a warning will be logged. To initalize a yp0, you can use the `compute_consistent_initial_conditions`

An example warning message:
```text
WARNING:torchdae.util:Inconsistent initial conditions at t=0.0. ||F(t0, y0, yp0)|| = 1.000e+00 (tol = 1.000e-08). Ensure F(t0, y0, yp0) ≈ 0 before integration.
```

An example of using `compute_consistent_initial_conditions`:

```python
import torch
from torchdae import compute_consistent_initial_conditions

# initial state to satisfy y1 + y2 = 0
y0 = torch.tensor([[0.5, -0.5]])

# random guess
yp0_guess = torch.tensor([[0.0, 0.0]])

# compute inital at time 0
y0_consistent, yp0_consistent = compute_consistent_initial_conditions(
    F=my_dae,
    t0=0.0,
    y0=y0,
    yp0_guess=yp0_guess,
    tol=1e-8
)

print("Consistent y0: ", y0_consistent.numpy())   # [[0.5, -0.5]]
print("Consistent yp0:", yp0_consistent.numpy())  # [[-1.0,  0.0]]
```


## Solving for trajectory

Putting the pieces together to solve:

```
from torchdae import solve_bdf2

sol = solve_bdf2(
    F=my_dae,
    t_span=(0.0, 1.0),
    y0=y0_consistent,
    yp0=yp0_consistent,
    h=0.01
)
```

Note that you have to either define the number of steps `n_steps` or the step size `h` for all of the solvers.

All index-1 solvers return `DAESolution` object, it includes `ts` for timestamps, `ys` for the states of solved trajectory, `yp_final` for the final derivative, and success if worked without needing an event handling.

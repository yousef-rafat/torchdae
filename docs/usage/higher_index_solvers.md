# Higher Index Solvers

Currently, `torchdae` supports `generalized_alpha` as the native higher-order solver that doesn't require index reduction to work

A lot of parameters are shared with [Index-1 Solvers](https://yousef-rafat.github.io/torchdae/usage/index_1_solvers/).

This page will explain the special Higher Index parameters.

## Mechanical Parameters

### `M` (Mass Matrix)

A tensor representing the mass matrix

### `g` (Constraint Function)

The constraint function $g(q, v) = 0$.

### `q0` and `v0`
The initial position (displacement) and velocity vectors.

### `a0` and `lam0` (Optional)
The initial acceleration $a(0)$ and initial Lagrange multiplier $\lambda(0)$ 

## Stability and Damping Parameters

### `rho_inf` (default: `0.8`)
The spectral radius at infinity, controlling the amount of high-frequency numerical damping.

* `rho_inf = 1.0`: Zero numerical damping.
* `rho_inf = 0.0`: Maximum numerical damping.

---

## Return Class: `MechanicalDAESolution`

Class specific for high-index mechanical problems including:
* `ts`: Timestamps of the solved trajectory.
* `qs`: Position (displacement) trajectory tensor.
* `vs`: Velocity trajectory tensor.
* `a_final`: Final acceleration at termination.
* `lam_final`: Final Lagrange multipliers (constraint forces).

---

## Coding Example

```python
import torch
import math
from torchdae import solve_generalized_alpha

# Define physical operators
def mass_matrix(q):
    return torch.eye(2, dtype=q.dtype, device=q.device) * 1.0  # Mass = 1.0 kg

def force_vector(t, q, v):
    return torch.tensor([0.0, -9.81], dtype=q.dtype, device=q.device)  # Gravity

def constraint(q, v):
    # Combined velocity/position constraint for stable unprojected integration
    return (q * v).sum(dim=-1, keepdim=True) + 0.5 * ((q**2).sum(dim=-1, keepdim=True) - 1.0)

# initial positions
q0 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
v0 = torch.tensor([[0.0, 0.0]], dtype=torch.float64)

sol = solve_generalized_alpha(
    M=mass_matrix,
    F=force_vector,
    g=constraint,
    t_span=(0.0, 3.0),
    q0=q0,
    v0=v0,
    h=0.01,
    rho_inf=0.8,
)

print("Pendulum integration successful!")
print("Final coordinates at t=3.0:", sol.qs[-1, 0].numpy())
```

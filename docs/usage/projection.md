# Constraint & Projection Functions

Projection functions are used when the predicted next value has violated the constrain function, `g(y) != 0`. It helps pulling it back into the physical manifold of the constrain.

Currently, `torchdae` supports the `Coordinate Projection Method (CPM)`.

To use a projection function, you must define a constrain function along with the projection function inside the `DAEFunctions`.

## Coding Example

```
import torch
from torchdae import DAEFunctions, solve_bdf2

# define the physics: y = [x, y]
def physics(t, y, yp):
    x, y_coord = y[..., 0], y[..., 1]
    xp, _ = yp[..., 0], yp[..., 1]
    
    # Differential equation: x' = y
    f1 = xp - y_coord
    # Algebraic constraint: x^2 + y^2 = 1 (enforced during steps)
    f2 = x**2 + y_coord**2 - 1.0
    return torch.stack([f1, f2], dim=-1)

# constraint function
def circle_constraint(y):
    # Returns the constraint residual. Shape: (Batch, 1)
    return (y ** 2).sum(dim=-1, keepdim=True) - 1.0

# the solver automatically defaults to 'coordinate_projection' by default
# you can pass a manual function as projection_fn=my_func
system = DAEFunctions(
    F=physics,
    constraint_fn=circle_constraint
)

# initialize
y0 = torch.tensor([[1.0, 0.0]])

sol = solve_bdf2(
    F=system,  # Pass the packaged DAEFunctions object
    t_span=(0.0, 2.0),
    y0=y0,
    h=0.01
)
```
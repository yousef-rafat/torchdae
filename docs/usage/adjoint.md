# Adjoint Sensitivity

Adjoint Sensitivity allows you to compute gradients of your loss function in continuous space.

The library contains implementation of Adjoint Sensitivity specific for DAEs

To access the API, you can use `DAEAdjointFunction.apply` or its functional wrapper `solve_dae_adjoint`

A coding example:

```python
import torch
from torchdae import solve_dae_adjoint, solve_bdf1

# parameters that require grad
p = torch.tensor([1.5, 1.0], requires_grad=True)

# define a function that uses these parameters
def physics(t, y, yp):
    y1, y2 = y[..., 0], y[..., 1]
    y1p, _ = yp[..., 0], yp[..., 1]
    
    f1 = y1p + y1 - p[0] * y2
    t_tensor = torch.as_tensor(t, dtype=y.dtype, device=y.device)
    f2 = y1 + y2 - p[1] * torch.sin(t_tensor)
    return torch.stack([f1, f2], dim=-1)

# initial state that requires gradients
y0 = torch.tensor([[0.5, -0.5]], requires_grad=True)

# this returns the trajectory state from DAESolution: sol.ys
ys = solve_dae_adjoint(
    F=physics,
    t_span=(0.0, 1.0),
    y0=y0,
    h=0.01,
    params=[p],
    solver_fn=solve_bdf1
)

# loss_fn: half-squared norm of the final state
loss = 0.5 * torch.sum(ys[-1] ** 2)

loss.backward()

print("Gradient with respect to initial conditions y0:")
print(y0.grad.numpy())

print("\nGradient with respect to parameters p:")
print(p.grad.numpy())
```

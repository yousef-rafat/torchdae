<h1 align='center'>TorchDAE</h1>
<h2 align='center'>Numerical Differential-Algebraic Equation solvers in PyTorch. Autodifferentiable and GPU-capable.</h2>

<p align="center">
  <img width="3584" height="1024" alt="torchdae_up" src="https://github.com/user-attachments/assets/c82a36ab-af54-420a-a3ae-226263912e29" />
</p>

TorchDAE is a [PyTorch](https://github.com/pytorch/pytorch)-based library providing numerical Differential-Algebraic Equation (DAE) solvers.

Features include:

- **Implicit Solvers**: Multiple stiff solvers (including BDF1, BDF2, SDIRK TR-BDF2, and 5th-order Radau IIA);
- **Automatic Index Reduction**: lowering high-index DAEs to Index-1 using Pantelides' algorithm and Mattsson-Söderlind Dummy Derivatives;
- **Manifold Stabilization**: Coordinate Projection Method (CPM) and Baumgarte feedback to eliminate numerical constraint drift;
- **Events & Resets**: Vectorized, differentiable event handling with continuous-time Hermite state interpolation and resets;
- **Adjoint Methods**: Continuous adjoint sensitivity backward-in-time for constant-memory backpropagation;
- **Vmap support**: Full support for PyTorch `vmap` and batched input on GPU and CPU pipelines.

## Installation

```bash
pip install torchdae
```
Requires Python 3.8+ and PyTorch 2.0+.

## Documentation
Available at [https://yousef-rafat.github.io/torchdae/](https://yousef-rafat.github.io/torchdae/).

## Quick Example 
A simple example of how to solve an Index-1 DAE with BDF2.

```python
import torch
from torchdae import solve_bdf2

# Define a simple Index-1 DAE: F(t, y, yp) = 0
def physics(t, y, yp):
    # supporting batching
    y1, y2 = y[..., 0], y[..., 1]
    y1p, _ = yp[..., 0], yp[..., 1]
    
    # f1 (differential): y1' + y1 - y2 = 0
    f1 = y1p + y1 - y2
    
    # f2 (algebraic): y1 + y2 - sin(t) = 0
    t_tensor = torch.as_tensor(t, dtype=y.dtype, device=y.device)
    f2 = y1 + y2 - torch.sin(t_tensor)
    
    return torch.stack([f1, f2], dim=-1)

y0 = torch.tensor([[0.5, -0.5]])

sol = solve_bdf2(physics, t_span=(0.0, 1.0), y0=y0, h=0.01)

print("Solved states at t=1.0:", sol.ys[-1, 0].numpy())
```

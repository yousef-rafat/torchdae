```markdown
# TorchDAE in a nutshell

TorchDAE is a PyTorch-based library providing numerical solvers for Differential-Algebraic Equations (DAEs).

Features include:
- **Implicit Solvers**: BDF1, BDF2, SDIRK TR-BDF2, and 5th-order Radau IIA.
- **Index Reduction**: Automatic structural index lowering using Pantelides' algorithm and Mattsson-Söderlind Dummy Derivatives.
- **Drift Correction**: Post-step coordinate projection to keep variables strictly on their physical manifolds.
- **Events & Resets**: Vectorized, differentiable event handling to catch boundary crossings and apply discontinuous state resets.
- **Constant Memory Backpropagation**: Continuous adjoint sensitivity to train Neural DAEs without running out of GPU memory.
- **Batch Parallelism**: Full support for PyTorch `vmap` and `torch.compile` to integrate directly with deep learning pipelines.

From a technical point of view, the library is written entirely in PyTorch using functional transforms (`vmap`, `jacrev`, `jvp`). This means you can embed any PyTorch neural network directly inside your DAEs and backpropagate through the entire solver with zero language-bridging overhead.

## Installation

```bash
pip install torchdae
```

Requires Python 3.8+.

## Quick example

Here is a simple example showing how to solve an Index-1 DAE:

```python
import torch
from torchdae.bdf import solve_bdf2

# Define a simple Index-1 DAE: F(t, y, yp) = 0
def physics(t, y, yp):
    t_tensor = torch.as_tensor(t, dtype=y.dtype, device=y.device)
    y1, y2 = y[..., 0], y[..., 1]
    y1p, _ = yp[..., 0], yp[..., 1]
    
    f1 = y1p + y1 - y2
    f2 = y1 + y2 - torch.sin(t_tensor)
    return torch.stack([f1, f2], dim=-1)

# Initial state with a batch dimension: Shape (1, 2)
y0 = torch.tensor([[0.0, 0.0]])

# Solve the system
sol = solve_bdf2(physics, t_span=(0.0, 1.0), y0=y0, h=0.01)

print("Solved states at t=1.0:", sol.ys[-1, 0])
```

## Next steps

Have a look at the Getting Started page.
```
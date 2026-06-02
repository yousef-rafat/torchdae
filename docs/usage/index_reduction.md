# Index Reduction

`torchdae`'s Index Reduction supports `differentiation` and `dummy_derivative` modes.

To start with the Index Reduction, we run the `simplify_dae` along with the desired mode.

The function will return the function wrapped in either `IndexReducedDAE` or `DummyDerivativeDAE` depending on the mode along with the the current state `y0` and its derivative `yp0`.

Internally, the `simplify_dae` function runs the `analyze_pytorch_dae` function to get the `DAEStructure` that contains equations and properties of the inputted function.

The `differentiation` takes the derivative of the equation depending on the `differentiation_orders` of the inputted equations using the Pantelides' algorithm and implements Baumgarte stabilization for higher index equations, requiring an `alpha` parameter.

The `dummy_derivative` mode implements the Mattsson-Söderlind algorithim without requiring an `alpha` parameter.


An example with `simplify_dae`:

```python
import torch
from torchdae import simplify_dae, solve_bdf1

# Index-3 pendulum system
# (Positions x, y constraint: x^2 + y^2 = L^2)
# x^2 + y^2 - L^2 = 0
def pendulum_physics(t, y, yp):
    # ... (Pendulum equations of motion) ...
    ...

t0 = 0.0
# y0 = [x, y, vx, vy, lambda]
y0 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
yp0_guess = torch.tensor([[0.0, 0.0, 0.0, -9.81, 0.0]], dtype=torch.float64)

F_reduced, Z0, Zp0 = simplify_dae(
    F=pendulum_physics,
    t0=t0,
    y0=y0,
    yp0=yp0_guess,
    algorithm="dummy_derivative"
)

sol = solve_bdf1(
    F=F_reduced,
    t_span=(t0, 2.0),
    y0=Z0,
    h=0.01
)
```

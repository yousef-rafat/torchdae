# Reset Functions

Reset functions work in parallel with [Event functions](events.md).

After the Event function handling is done, the reset function is called, usually used for resetting the simulation's values.
 
When an event occurs and the reset function is set to None, the simulation is ended with the event triggering.

To keep the simulation running, you will have to define a reset_fn inside the `DAEFunctions`.

```python
import torch
from torchdae import DAEFunctions

# State vector: y = [x, y, vx, vy]
# constraint: Pendulum hits a wall at x = 0.5
def wall_event(t, y):
    return y[..., 0] - 0.5  # triggers when x crosses 0.5

# reset_fn: bounce off the wall
def bounce_reset(t_evt, y_evt):
    y_new = y_evt.clone()
    # reverse the x velocity with a restitution loss of 30%
    y_new[..., 2] = -0.7 * y_evt[..., 2]
    
    return y_new

system = DAEFunctions(
    F=pendulum_physics,
    event_fn=wall_event,
    reset_fn=bounce_reset
)

```

When a reset function is an identity (e.g. lambda t, y: y), the simulation continues with the correction of the event handling logic whenever the boundary is crossed
# Events

In `torchdae`, an event is the case where the defined constrains are violated per timestep update

Events are logged in the `DAESolution` return class. It contains: 

* `event_triggered`: a boolean if any event was triggered in the simulation,
* `event_mask`: a mask of which batched elements triggered the event,
* `t_event`: exact time events, and
* `y_event`: exact states of the events

The events are handled using bisection for finding the exact timestep of where the constrain was violated and, using hermite interpolation, it corrects the issue


## The event function (`event_fn`)

To define an event, you must define an `event_fn`. An `event_fn` takes the timestep and the state of the time as first and second inputs respectivly

To define an `event_fn`, you will add it in the `DAEFunctions`:

```
import torch
from torchdae import DAEFunctions, solve_bdf2

def physics(t, y, yp):
    ...

system = DAEFunctions(
    F=physics,
    event_fn=floor_event
)

# NOTE: Because no 'reset_fn' is specified, this acts as a TERMINAL event.
# The simulation will stop immediately when the height crosses zero.
sol = solve_bdf2(
    F=system,  # Pass the packaged DAEFunctions object
    t_span=(0.0, 5.0),
    y0=y0,
    h=0.01
)

if sol.event_triggered:
    print(f"Simulation stopped early because an event triggered at t = {sol.t_event.item():.4f}!")
    print(f"State at impact: {sol.y_event.numpy()}")
```

An identity event function (e.g. lambda t, y: y) will trigger the correction whenever the constrain crosses zero
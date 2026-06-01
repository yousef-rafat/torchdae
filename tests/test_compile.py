import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import traceback
import inspect
from torchdae import (
    solve_bdf1,
    solve_bdf2,
    solve_tr_bdf2,
    solve_generalized_alpha,
    solve_radau_iia5
)

# A simple implicit DAE to test tracing.
# Accepts variable positional arguments to accommodate first-order and second-order solver signatures.
def simple_dae(t, *args):
    # args[0] is state (y or u), args[1] is derivative (yp or up)
    y = args[0]
    yp = args[1] if len(args) > 1 else torch.zeros_like(y)
    return yp + y

def run_compile_test():
    # Force single-threaded execution to prevent CPU overhead during compilation
    torch.set_num_threads(1)
    
    solvers = {
        "solve_bdf1": solve_bdf1,
        "solve_bdf2": solve_bdf2,
        "solve_tr_bdf2": solve_tr_bdf2,
        "solve_generalized_alpha": solve_generalized_alpha,
        "solve_radau_iia5": solve_radau_iia5
    }
    
    print("==================================================")
    print("Testing torch.compile (Eager Backend) Compatibility")
    print("==================================================")
    print("Note: Using backend='eager' to test tracing without needing a C++ compiler (cl/clang).\n")
    
    y0 = torch.tensor([[1.0]])  # Shape (Batch, State) = (1, 1)
    t_span = (0.0, 0.1)
    h_step = 0.05               # Solves in exactly 2 steps to minimize compilation overhead
    
    all_passed = True
    
    for name, solver_fn in solvers.items():
        print(f"Testing {name}...")
        
        # 1. Dynamically inspect the solver's parameter signature
        sig = inspect.signature(solver_fn)
        
        # 2. Build the correct keyword argument dictionary
        kwargs = {
            "F": simple_dae,
            "t_span": t_span,
            "h": h_step,
        }
        
        # Position / displacement state mapping
        if "y0" in sig.parameters:
            kwargs["y0"] = y0
        if "u0" in sig.parameters:
            kwargs["u0"] = y0
        if "q0" in sig.parameters:
            kwargs["q0"] = y0
            
        # Velocity / derivative state mapping
        if "v0" in sig.parameters:
            kwargs["v0"] = torch.zeros_like(y0)
        if "yp0" in sig.parameters:
            kwargs["yp0"] = torch.zeros_like(y0)
            
        # Second-order constrained DAE specifics (Generalized-alpha)
        if "M" in sig.parameters:
            # Identity mass matrix
            kwargs["M"] = lambda q: torch.eye(q.shape[-1], dtype=q.dtype, device=q.device)
        if "g" in sig.parameters:
            # Dummy constraint returning flat zeros of shape (B, 1)
            kwargs["g"] = lambda q, v: torch.zeros((q.shape[0], 1), dtype=q.dtype, device=q.device)
            
        # 3. Wrap the solver inside a compiled function with backend="eager"
        # fullgraph=False is essential to allow Python loop and list operations to graph-break safely
        @torch.compile(backend="eager", fullgraph=False)
        def compiled_solve(y0_val):
            # We copy kwargs to prevent modifying the original dictionary across iterations
            step_kwargs = kwargs.copy()
            if "y0" in step_kwargs:
                step_kwargs["y0"] = y0_val
            elif "u0" in step_kwargs:
                step_kwargs["u0"] = y0_val
            elif "q0" in step_kwargs:
                step_kwargs["q0"] = y0_val
            return solver_fn(**step_kwargs)
            
        # 4. Trigger compilation and execute
        try:
            sol = compiled_solve(y0)
            
            # Extract states and verify structure
            ys = getattr(sol, "ys", getattr(sol, "qs", None))
            ts = getattr(sol, "ts", None)
            
            if ys is None or ts is None:
                raise ValueError("Solver returned None values.")
            if len(ys) != 3:
                raise ValueError(f"Expected 3 states (t0, t1, t2), got {len(ys)}")
                
            print(f"  --> {name} traced and executed successfully.")
            
        except Exception as e:
            print(f"  [!] Tracing FAILED for {name}")
            traceback.print_exc()
            all_passed = False
            print()
            
    print("==================================================")
    if all_passed:
        print("ALL SOLVERS PASSED COMPILATION CHECKS (TRACING)")
    else:
        print("SOME SOLVERS FAILED COMPILATION CHECKS (TRACING)")
    print("==================================================")

if __name__ == "__main__":
    run_compile_test()

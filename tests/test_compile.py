##############################
# THIS IS FUTURE WORK
##############################

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import inspect
import traceback

try:
    import torch._inductor.config as inductor_config
    inductor_config.cpp.enabled = False
except Exception:
    pass

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
    torch.set_num_threads(1)
    
    solvers = {
        "solve_bdf1": solve_bdf1,
        "solve_bdf2": solve_bdf2,
        "solve_tr_bdf2": solve_tr_bdf2,
        "solve_generalized_alpha": solve_generalized_alpha,
        "solve_radau_iia5": solve_radau_iia5
    }
    
    print("==================================================")
    print("Testing Native Step Compilation on DAE Solvers")
    print("==================================================")
    
    y0 = torch.tensor([[1.0]])  # Shape (Batch, State) = (1, 1)
    t_span = (0.0, 0.1)
    h_step = 0.05
    
    all_passed = True
    
    for name, solver_fn in solvers.items():
        print(f"Testing {name}...")
        
        sig = inspect.signature(solver_fn)
        
        kwargs = {
            "F": simple_dae,
            "t_span": t_span,
            "h": h_step,
        }
        
        # First-order vs Second-order state parameter mapping
        if "y0" in sig.parameters:
            kwargs["y0"] = y0
        elif "u0" in sig.parameters:
            kwargs["u0"] = y0
            
        # Optional derivatives / velocity parameters
        if "yp0" in sig.parameters:
            kwargs["yp0"] = torch.zeros_like(y0)
        if "v0" in sig.parameters:
            kwargs["v0"] = torch.zeros_like(y0)
            
        # Detect and set the native step compilation flag
        if "compiles_step" in sig.parameters:
            kwargs["compiles_step"] = True
        elif "compile_steps" in sig.parameters:
            kwargs["compile_steps"] = True
            
        try:
            sol = solver_fn(**kwargs)
            
            if sol.ys is None or sol.ts is None:
                raise ValueError("Solver returned None values.")
            
            print(f"  --> {name} compiled and ran successfully.")
            
        except Exception:
            print(f"  [!] Step Compilation FAILED for {name}")
            traceback.print_exc()
            all_passed = False
            print()
            
    print("==================================================")
    if all_passed:
        print("ALL SOLVERS PASSED NATIVE COMPILATION CHECKS")
    else:
        print("SOME SOLVERS FAILED COMPILATION CHECKS")
    print("==================================================")

if __name__ == "__main__":
    run_compile_test()
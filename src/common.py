import torch
import torch.func as func
from typing import Callable, Optional
import warnings
from dataclasses import dataclass

__all__ = ["DAESolution"]

@dataclass(frozen=True)
class DAESolution:
    """
    Standardized, immutable container for all DAE/ODE solver results.
    """
    ts: torch.Tensor                       # Timestamps of the solved trajectory: Shape (T,)
    ys: torch.Tensor                       # States of the solved trajectory: Shape (T, B, *state_shape)
    yp_final: torch.Tensor                 # Final derivative state at termination: Shape (B, *state_shape)
    success: bool                          # True if the solver completed without errors or forced terminations
    
    # Event-handling metadata (optional)
    event_triggered: bool = False          # True if a boundary event stopped the solver early
    event_mask: Optional[torch.Tensor] = None  # Boolean mask of which batch elements triggered the event: Shape (B,)
    t_event: Optional[torch.Tensor] = None     # Exact times of events: Shape (B,)
    y_event: Optional[torch.Tensor] = None     # Exact states of events: Shape (B, *state_shape)

def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)


def _shape(x_flat: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    return x_flat.reshape(template.shape)


def try_compile(fn: Callable, fullgraph=False, dynamic=True, **compile_kwargs) -> Callable:
    try:
        if hasattr(torch, "compile"):
            compiled_fn = torch.compile(fn, fullgraph=fullgraph, dynamic=dynamic, **compile_kwargs)
            return compiled_fn
    except Exception as e:
        warnings.warn(
            f"torch.compile failed or is not supported. Error: {e}. "
            f"Falling back to torch.jit.script..."
        )

    try:
        return torch.jit.script(fn)
    except Exception as e:
        warnings.warn(
            f"torch.jit.script failed. Error: {e}. "
            f"Falling back to eager mode (uncompiled Python execution)."
        )
    return fn

class StatefulJacobian:
    # states are: "always", "freeze_step", "freeze_dynamic"
    # the "always" state always calculates the jacobian
    # the "freeze_step" calculates at the first x0 step and reuse the jacobian
    # the "freeze_dynamic" calculates the jacobian depending on the convergence or recompute_every value
    def __init__(
        self, 
        raw_jac_fn: Callable[[torch.Tensor], torch.Tensor], 
        strategy: str, 
        x0: torch.Tensor,
        recompute_every: Optional[int] = None
    ):
        self.raw_jac_fn = raw_jac_fn
        self.strategy = strategy
        self.recompute_every = recompute_every
        self.counter = 0
        self.prev_norm = None  # To store the residual norm from the previous iteration
        
        if strategy in ("freeze_step", "freeze_dynamic"):
            self.cached_J = self.raw_jac_fn(x0)
        else:
            self.cached_J = None

    def __call__(
        self, 
        x: torch.Tensor, 
        norm: Optional[torch.Tensor] = None, 
        iteration: Optional[int] = None
    ) -> torch.Tensor:
        if self.strategy == "always":
            self.counter += 1
            return self.raw_jac_fn(x)
            
        elif self.strategy == "freeze_step":
            self.counter += 1
            return self.cached_J
            
        elif self.strategy == "freeze_dynamic":
            if norm is not None:
                recompute_mask = torch.zeros_like(norm, dtype=torch.bool)
            else:
                recompute_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
            
            # override the dynamic norm ratio convergence handling if recompute_every is provided
            if self.recompute_every is not None:
                if self.counter > 0 and self.counter % self.recompute_every == 0:
                    recompute_mask = torch.ones_like(recompute_mask)
            
            # norm/prev_norm tells us if the convergences slows down or started to diverge
            else:
                if self.prev_norm is not None and norm is not None:
                    ratio = norm / (self.prev_norm + 1e-12)
                    slow_or_diverging = ratio > 0.9
                    recompute_mask = recompute_mask | slow_or_diverging

            if recompute_mask.any():
                new_J = self.raw_jac_fn(x)
                # expand mask from (B,) to (B, 1, 1) to match (B, D, D) Jacobian dimensions
                mask_expanded = recompute_mask.unsqueeze(-1).unsqueeze(-1)
                self.cached_J = torch.where(mask_expanded, new_J, self.cached_J)
                
            self.counter += 1
            if norm is not None:
                self.prev_norm = norm.clone()
                
            return self.cached_J
            
        else:
            raise ValueError(f"Unknown Jacobian strategy: {self.strategy}")

def newton_solve(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    tol: float = 1e-8,
    max_iter: int = 50,
    damping: float = 1.0,
) -> torch.Tensor:
    """
    Generic adaptive damped Newton: solve residual_fn(x) = 0 for x.
    x0 -> inital guess
    """
    x = _flat(x0).clone().detach()

    for _ in range(max_iter):
        r = residual_fn(x)
        norm = r.norm().item()
        if norm < tol:
            return _shape(x, x0)

        J = func.jacrev(residual_fn)(x)

        try:
            delta = torch.linalg.solve(J, -r)
        except RuntimeError:
            delta = torch.linalg.lstsq(J, -r).solution

        # started user value then halves every time it
        alpha = damping
        while alpha > 1e-4:
            x_trial = x + alpha * delta
            if residual_fn(x_trial).norm().item() < norm:
                break
            alpha *= 0.5
        
        x = x + alpha * delta

    raise RuntimeError(f"Newton solve failed. Final residual: {norm:.3e}")

def batched_newton_solve(
    residual_fn,
    jacobian_fn: StatefulJacobian,
    x0: torch.Tensor,
    tol: float = 1e-8,
    max_iter: int = 50,
    damping: float = 1.0,
) -> torch.Tensor:
    """
    Vectorized Newton-Raphson solver with batch-wise backtracking line search.
    """
    x = x0.clone().detach()
    batch_size = x0.shape[0]
    
    converged = torch.zeros(batch_size, dtype=torch.bool, device=x0.device)
    
    for iteration in range(max_iter):
        r = residual_fn(x)
        norm = torch.linalg.vector_norm(r, dim=-1)
        
        converged = converged | (norm < tol)
        if converged.all():
            break
            
        J = jacobian_fn(x, norm=norm, iteration=iteration)
        
        try:
            delta = torch.linalg.solve(J, -r.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            delta = torch.linalg.lstsq(J, -r.unsqueeze(-1)).solution.squeeze(-1)
            
        # line search
        alpha = torch.full((batch_size, 1), damping, device=x0.device)
        selected_alpha = torch.zeros((batch_size, 1), device=x0.device)
        active_ls = ~converged
        
        # 14 iterations brings alpha down to 0.5^14 (~6e-5)
        for _ in range(14):
            if not active_ls.any():
                break
                
            x_trial = x + alpha * delta
            r_trial = residual_fn(x_trial)
            norm_trial = torch.linalg.vector_norm(r_trial, dim=-1)
            
            improved = norm_trial < norm
            newly_improved = active_ls & improved
            
            selected_alpha = torch.where(newly_improved.unsqueeze(-1), alpha, selected_alpha)
            active_ls = active_ls & ~improved
            
            alpha = alpha * 0.5
            
        # if an element never improved, fall back to the smallest step size
        never_improved = ~converged & (selected_alpha.squeeze(-1) == 0.0)
        selected_alpha = torch.where(never_improved.unsqueeze(-1), alpha * 2.0, selected_alpha)
        
        # step only the active elements
        x = torch.where(converged.unsqueeze(-1), x, x + selected_alpha * delta)
        
    return x

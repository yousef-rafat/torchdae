import logging
import torch

logger = logging.getLogger(__name__)

# checks if the output is a tensor and is differentiable
def validate_residual_function(F, t0, y0, yp0):

    # handle batched inputs
    if y0.ndim == 1:
        raise ValueError("Batch dim must exist")

    if y0.shape[0] > 1:
        y0 = y0[0:1]    
        yp0 = yp0[0:1]
        if isinstance(t0, torch.Tensor) and t0.ndim > 0 and t0.shape[0] > 1:
            t0 = t0[0:1]

    with torch.no_grad():
        result = F(t0, y0, yp0)
    
    if not isinstance(result, torch.Tensor):
        raise TypeError(
            f"Residual function F must return a torch.Tensor, "
            f"got {type(result).__name__}. "
            f"If you are using NumPy, convert your function to use torch operations."
        )
    
    if result.shape != y0.shape:
        raise ValueError(
            f"F output shape {tuple(result.shape)} does not match y0 shape {tuple(y0.shape)}"
        )
    
    try:
        _ = torch.func.jacrev(lambda y: F(t0, y, yp0).sum())(y0)
    except Exception as e:
        raise RuntimeError(
            f"F does not appear to be differentiable with torch.func.jacrev. "
            f"Ensure F uses only PyTorch operations. Original error: {e}"
        )
    
def interpolate_step_hermite(t_query, t_n, t_next, y_n, y_next, yp_n, yp_next):
    h = t_next - t_n
    theta = (t_query - t_n) / h
    
    # Cubic Hermite basis functions for state interpolation
    h00 = 2.0 * theta**3 - 3.0 * theta**2 + 1.0
    h10 = theta**3 - 2.0 * theta**2 + theta
    h01 = -2.0 * theta**3 + 3.0 * theta**2
    h11 = theta**3 - theta**2
    
    y_interp = h00 * y_n + (h10 * h) * yp_n + h01 * y_next + (h11 * h) * yp_next
    
    # derivative basis functions (accounting for chain-rule dtheta/dt = 1/h)
    yp_interp = ((6.0 * theta**2 - 6.0 * theta) / h) * (y_n - y_next) + \
                (3.0 * theta**2 - 4.0 * theta + 1.0) * yp_n + \
                (3.0 * theta**2 - 2.0 * theta) * yp_next
                
    return y_interp, yp_interp


# uses bisection + hermite interpolation for correcting
def handle_step_events(
        event_fn, t_n: float, t_next: float, y_n: torch.Tensor, y_next: torch.Tensor,
        yp_n: torch.Tensor, yp_next: torch.Tensor, tol: float = 1e-8, max_iter: int = 100,
    ):

    """
    Checks if any event/constraint was crossed during the step t_n -> t_next.
    If a violation (overshoot) occurred, interpolates to find the exact root t*.
    """
    is_batched = y_n.shape[0] > 1
    
    def process_single(
        y_n_s: torch.Tensor, y_next_s: torch.Tensor,
        yp_n_s: torch.Tensor, yp_next_s: torch.Tensor, batch_idx = None
    ):
        idx_suffix = f" at batch index {batch_idx}" if batch_idx is not None else ""
        
        with torch.no_grad():
            h_n = event_fn(t_n, y_n_s).item()
            h_next = event_fn(t_next, y_next_s).item()
            
        # Detect zero-crossing (opposite signs check)
        if h_n * h_next >= 0:
            return t_next, y_next_s, yp_next_s, False
            
        logger.info(f"Constraint crossing detected{idx_suffix} in interval [{t_n:.4f}, {t_next:.4f}]. Locating root...")
        
        # Bisection loop to find t* such that h(t*, y(t*)) = 0
        t_l, t_r = t_n, t_next
        h_l, _ = h_n, h_next
        
        for _ in range(max_iter):
            if abs(t_r - t_l) < tol:
                break
                
            t_m = 0.5 * (t_l + t_r)
            y_m, _ = interpolate_step_hermite(t_m, t_n, t_next, y_n_s, y_next_s, yp_n_s, yp_next_s)
            
            with torch.no_grad():
                h_m = event_fn(t_m, y_m).item()
                
            if abs(h_m) < tol:
                t_l = t_r = t_m
                break
                
            if h_l * h_m < 0:
                t_r = t_m
            else:
                t_l = t_m
                h_l = h_m
                
        t_star = 0.5 * (t_l + t_r)
        y_star, yp_star = interpolate_step_hermite(t_star, t_n, t_next, y_n_s, y_next_s, yp_n_s, yp_next_s)
        
        logger.info(f"Event localized{idx_suffix} at t* = {t_star:.6e} (||h(t*)|| ≈ {abs(h_l):.3e}).")
        return t_star, y_star, yp_star, True

    if is_batched:
        batch_size = y_n.shape[0]
        t_events = torch.full((batch_size,), t_next, dtype=y_n.dtype, device=y_n.device)
        y_events = y_next.clone()
        yp_events = yp_next.clone()
        event_mask = torch.zeros(batch_size, dtype=torch.bool, device=y_n.device)
        
        for b in range(batch_size):
            t_star, y_star, yp_star, triggered = process_single(
                y_n[b], y_next[b], yp_n[b], yp_next[b], batch_idx=b
            )
            if triggered:
                t_events[b] = t_star
                y_events[b] = y_star
                yp_events[b] = yp_star
                event_mask[b] = True
                
        return t_events, y_events, yp_events, event_mask
    else:
        t_star, y_star, yp_star, triggered = process_single(y_n, y_next, yp_n, yp_next)
        t_event = torch.tensor(t_star, dtype=y_n.dtype, device=y_n.device)
        event_mask = torch.tensor(triggered, dtype=torch.bool, device=y_n.device)
        return t_event, y_star, yp_star, event_mask

def check_initial_condition_violation(F, t0, y0,  yp0, tol=1e-8, strict=False, index=1) -> None:
    """
    Verifies if initial conditions satisfy both explicit and hidden (high-index) 
    constraints.
    """
    is_batched = y0.shape[0] > 1
    def check_single(y0_s: torch.Tensor, yp0_s: torch.Tensor, batch_idx = None) -> None:
        idx_suffix = f" at batch index {batch_idx}" if batch_idx is not None else ""
        
        with torch.no_grad():
            result = F(t0, y0_s, yp0_s)
            norm = result.norm().item()

        if norm > tol:
            msg = (
                f"Inconsistent initial conditions{idx_suffix} at t={t0}. "
                f"||F(t0, y0, yp0)|| = {norm:.3e} (tol = {tol:.3e}). "
                f"Ensure F(t0, y0, yp0) ≈ 0 before integration."
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)

        if index >= 2:
            try:
                # discover algebraic equations by finding rows in dF/dyp that are zero
                def F_flat(yp_flat):
                    return F(t0, y0_s, yp_flat.view_as(y0_s)).flatten()
                
                J_yp = torch.func.jacrev(F_flat)(yp0_s.flatten())
                row_norms = torch.linalg.vector_norm(J_yp, dim=1)
                algebraic_indices = torch.where(row_norms < 1e-12)[0]
                
                if len(algebraic_indices) > 0:
                    # Isolate discovered constraints as a standalone function: g(t, y) = 0
                    def g_func(t_val, y_val):
                        return F(t_val, y_val.view_as(y0_s), yp0_s).flatten()[algebraic_indices]
                    
                    # Compute dg/dy * yp0_s using a Jacobian-Vector Product (JVP)
                    _, dg_dy_yp = torch.func.jvp(lambda y: g_func(t0, y), (y0_s.flatten(),), (yp0_s.flatten(),))
                    
                    # Compute dg/dt using a central difference
                    h_t = 1e-6
                    dg_dt = (g_func(t0 + h_t, y0_s.flatten()) - g_func(t0 - h_t, y0_s.flatten())) / (2.0 * h_t)
                    
                    # total time derivative representing the hidden constraint
                    hidden_residual = dg_dy_yp + dg_dt
                    hidden_norm = hidden_residual.norm().item()
                    
                    if hidden_norm > tol:
                        level_str = "Index-2/Velocity" if index == 2 else "Index-3/Velocity"
                        msg = (
                            f"Inconsistent initial hidden constraints ({level_str} level){idx_suffix} at t={t0}. "
                            f"||d/dt[g(y(t), t)]|| = {hidden_norm:.3e} (tol = {tol:.3e}). "
                            f"Ensure constraints are consistent before integration."
                        )
                        if strict:
                            raise ValueError(msg)
                        logger.warning(msg)
                        
            except Exception as e:
                logger.debug(f"Could not automatically verify high-index hidden constraints: {e}")

    if is_batched:
        for b in range(y0.shape[0]):
            check_single(y0[b], yp0[b], batch_idx=b)
    else:
        check_single(y0, yp0)

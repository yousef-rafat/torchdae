"""
Provides mathematical structural analysis and automatic index reduction 
for Differential-Algebraic Equations (DAEs).
"""

import torch
import logging
import torch.func as func
from dataclasses import dataclass
from typing import Callable, List, Set, Tuple, Dict

__all__ = [
    "DAEStructure", "pantelides_reduction", "dummy_derivative_reduction", "analyze_pytorch_dae",
    "IndexReducedDAE", "simplify_dae", "dummy_derivative_reduction", "DummyDerivativeDAE"
]

logging.basicConfig(level=logging.INFO)

@dataclass(frozen=True)
class DAEStructure:
    """
    Unified metadata representation of a DAE system's structural properties.
    """
    equations: List[Set[Tuple[int, int]]]        # Bipartite graph dependencies: List of Sets of (var_idx, order)
    n_vars: int                                  # Total state dimension
    algebraic_equations: List[int]               # Indices of equations containing NO yp (derivative) dependencies
    differential_equations: List[int]            # Indices of equations containing yp dependencies
    algebraic_variables: List[int]               # State variables whose derivatives never appear in any equation
    unassigned_equations: List[int]              # Equations with empty dependency sets (likely modeling bugs)
    differentiation_orders: List[int]            # Number of differentiations required per equation to reach Index-1
    is_structurally_singular: bool               # True if no perfect bipartite matching exists over state appearances
    reduction_algorithm: str                     # Name of the algorithm used for index reduction analysis
    dummy_derivatives: List[int] = None          # Indices of variables selected as dummy derivatives

def dfs(eq_idx: int, visited: Set[int], equations, matching):
    for var_idx, _ in equations[eq_idx]:
        if var_idx not in visited:
            visited.add(var_idx)
            if var_idx not in matching or dfs(matching[var_idx], visited, equations, matching)[0]:
                matching[var_idx] = eq_idx
                return True, matching
    return False, matching

def pantelides_reduction(
    equations: List[Set[Tuple[int, int]]]
) -> List[int]:
    """
    Applies the pantelides algorithm to determine how many times each equation 
    must be differentiated to reduce the dae system to structural index-1.
    """
    n_eqs = len(equations)
    eq_diff_orders = [0] * n_eqs
    matching: Dict[Tuple[int, int], Tuple[int, int]] = {}
    
    # track the set of active equation nodes
    active_eqs = set((i, 0) for i in range(n_eqs))

    def get_w_values() -> List[int]:
        # compute the highest derivative order of each variable among all active equations
        w = [-1] * n_eqs
        for eq_idx, d_eq in active_eqs:
            for var_idx, d_var_orig in equations[eq_idx]:
                w[var_idx] = max(w[var_idx], d_var_orig + d_eq)
        return w

    def dfs_pantelides(
        eq_node: Tuple[int, int], 
        visited_vars: Set[Tuple[int, int]], 
        visited_eqs: Set[Tuple[int, int]],
        w: List[int]
    ) -> bool:
        eq_idx, d_eq = eq_node
        visited_eqs.add(eq_node)
        
        for var_idx, d_var_orig in equations[eq_idx]:
            d_var = d_var_orig + d_eq
            # edge only exists if the derivative is the highest active derivative in the system
            if d_var == w[var_idx]:
                var_node = (var_idx, d_var)
                
                if var_node not in visited_vars:
                    visited_vars.add(var_node)
                    
                    if var_node not in matching or dfs_pantelides(matching[var_node], visited_vars, visited_eqs, w):
                        matching[var_node] = eq_node
                        matching[eq_node] = var_node
                        return True
                        
        return False

    for i in range(n_eqs):
        queue = [(i, eq_diff_orders[i])]
        
        while len(queue) > 0:
            eq_node = queue.pop(0)
            
            # recalculate highest active derivatives
            w = get_w_values()
            
            visited_vars: Set[Tuple[int, int]] = set()
            visited_eqs: Set[Tuple[int, int]] = set()
            
            if not dfs_pantelides(eq_node, visited_vars, visited_eqs, w):
                # dfs failed: all visited equations form the singular subset
                for (e_idx, d_e) in visited_eqs:
                    new_d = d_e + 1
                    eq_diff_orders[e_idx] = max(eq_diff_orders[e_idx], new_d)
                    
                    new_eq_node = (e_idx, new_d)
                    active_eqs.add(new_eq_node)
                    queue.append(new_eq_node)
                    
                    # break the obsolete matching of the differentiated node
                    matched_var = matching.get((e_idx, d_e))
                    if matched_var:
                        matching.pop((e_idx, d_e), None)
                        matching.pop(matched_var, None)

    return eq_diff_orders


def dummy_derivative_reduction(
    equations: List[Set[Tuple[int, int]]]
) -> Tuple[List[int], List[int]]:
    """    
    Returns:
        eq_diff_orders: Number of differentiations required per equation
        dummy_vars: List of variable indices chosen as dummy derivatives
    """
    # run pantelides to get diff equations
    eq_diff_orders = pantelides_reduction(equations)
    dummy_vars = []
    n_eqs = len(equations)
    
    if any(d > 0 for d in eq_diff_orders):
        matching = {}

        for i in range(n_eqs):
            if eq_diff_orders[i] > 0:
                _, matching = dfs(i, set(), equations, matching)
                
        dummy_vars = sorted(list(matching.keys()))
        
    return eq_diff_orders, dummy_vars


REDUCTION_ALGORITHMS = {
    "differentiation": lambda eqs: (pantelides_reduction(eqs), []),
    "dummy_derivative": dummy_derivative_reduction,
}


def _has_perfect_matching(equations: List[Set[Tuple[int, int]]], n_vars: int) -> bool:
    matching = {}
    for i in range(n_vars):
        visited = set()
        _, matching = dfs(i, visited, equations, matching)
        
    return len(matching) == n_vars


def analyze_pytorch_dae(
    F: Callable[[float, torch.Tensor, torch.Tensor], torch.Tensor],
    t0: float,
    y0: torch.Tensor,
    yp0: torch.Tensor,
    algorithm: str = "dummy_derivative"
) -> DAEStructure:
    """
    Performs structural analysis on a DAE system.
    """
    algo_key = algorithm.lower().strip()
    if algo_key not in REDUCTION_ALGORITHMS:
        raise ValueError(
            f"Unsupported index reduction algorithm: '{algorithm}'. "
            f"Supported options: {list(REDUCTION_ALGORITHMS.keys())}"
        )

    y0_flat = y0.flatten()
    yp0_flat = yp0.flatten()
    n_vars = y0_flat.numel()
    
    def F_flat_y(y):
        return F(t0, y.view_as(y0), yp0_flat.view_as(yp0)).flatten()
        
    def F_flat_yp(yp):
        return F(t0, y0_flat.view_as(y0), yp.view_as(yp0)).flatten()

    J_y = func.jacrev(F_flat_y)(y0_flat)
    J_yp = func.jacrev(F_flat_yp)(yp0_flat)
    
    # generate the graph (j, val)
    equations= []
    for i in range(n_vars):
        eq_dependencies = set()
        for j in range(n_vars):
            if torch.abs(J_yp[i, j]) > 1e-12:
                eq_dependencies.add((j, 1))
            if torch.abs(J_y[i, j]) > 1e-12:
                eq_dependencies.add((j, 0))
                
        equations.append(eq_dependencies)
        
    algebraic_eqs = []
    diff_eqs = []
    unassigned_eqs= []
    
    for i, deps in enumerate(equations):
        if len(deps) == 0:
            unassigned_eqs.append(i)
            continue
            
        has_yp = any(order == 1 for _, order in deps)
        if has_yp:
            diff_eqs.append(i)
        else:
            algebraic_eqs.append(i)
            
    # identify variables whose derivatives never appear in any equation (identity)
    algebraic_vars = []
    for j in range(n_vars):
        derivative_found = False
        for i in range(n_vars):
            if (j, 1) in equations[i]:
                derivative_found = True
                break
        if not derivative_found:
            algebraic_vars.append(j)
            
    reduction_fn = REDUCTION_ALGORITHMS[algo_key]
    differentiation_orders, dummy_vars = reduction_fn(equations)
    is_singular = not _has_perfect_matching(equations, n_vars)
    
    return DAEStructure(
        equations=equations,
        n_vars=n_vars,
        algebraic_equations=algebraic_eqs,
        differential_equations=diff_eqs,
        algebraic_variables=algebraic_vars,
        unassigned_equations=unassigned_eqs,
        differentiation_orders=differentiation_orders,
        is_structurally_singular=is_singular,
        reduction_algorithm=algo_key,
        dummy_derivatives=dummy_vars
    )

class IndexReducedDAE:
    """
    A Wrapper that dynamically transforms a high-index DAE
    into a stabilized Index-1 DAE using automatic differentiation and 
    Baumgarte stabilization.
    """
    def __init__(self, F: Callable, structure: DAEStructure, alpha: float = 10.0, index = 2):
        self.F = F
        self.differentiation_orders = structure.differentiation_orders
        self.high_index_eqs = [i for i, order in enumerate(self.differentiation_orders) if order > 0]
        self.alpha = alpha
        self.index = index

    def __call__(self, t: float, y: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
        if len(self.high_index_eqs) == 0:
            return self.F(t, y, yp)

        r = self.F(t, y, yp)
        r_new = r.clone()
        
        # compute dF/dt
        t_tensor = torch.tensor(t, dtype=y.dtype, device=y.device, requires_grad=True)
        t_tangent = torch.ones_like(t_tensor)
        _, dF_dt = func.jvp(lambda t_val, y_val: self.F(t_val, y_val, yp), (t_tensor, y), (t_tangent, yp))
        
        # compute second derivative d2F/dt2, using dF_dt
        if self.index == 3:    
            def dF_dt_func(t_val, y_val):
                _, JVP = func.jvp(lambda t_inner, y_inner: self.F(t_inner, y_inner, yp), (t_val, y_val), (torch.ones_like(t_val), yp))
                return JVP

            _, d2F_dt2 = func.jvp(dF_dt_func, (t_tensor, y), (t_tangent, yp))

        # fix numerical drif errors
        for idx in self.high_index_eqs:
            order = self.differentiation_orders[idx]
            
            if order == 1: # index 2
                r_new[idx] = dF_dt[idx] + self.alpha * r[idx]
                
            elif order == 2: # index 3
                # second-order Baumgarte stabilization
                r_new[..., idx] = d2F_dt2[..., idx] + 2.0 * self.alpha * dF_dt[..., idx] + (self.alpha ** 2) * r[..., idx]
                
        return r_new

class DummyDerivativeDAE:
    """
    A PyTorch-native wrapper that implements the Mattsson-Söderlind 
    Dummy Derivative method. It augments the DAE system with dummy 
    derivatives to reduce a high-index DAE to a stable Index-1 DAE.
    """
    def __init__(self, F: Callable, structure: DAEStructure):
        self.F = F
        self.n_vars = structure.n_vars
        self.differentiation_orders = structure.differentiation_orders
        self.dummy_derivatives = structure.dummy_derivatives
        self.high_index_eqs = [i for i, order in enumerate(self.differentiation_orders) if order > 0]
        self.K = len(self.dummy_derivatives)

    def __call__(self, t: float, Z: torch.Tensor, Zp: torch.Tensor) -> torch.Tensor:
        # extract the y, yp, and u
        y = Z[..., :self.n_vars]
        yp_dynamic = Zp[..., :self.n_vars]
        u = Z[..., self.n_vars:]
        
        # swap values in yp with algebric constants (dummy derivatives)
        yp_bar = yp_dynamic.clone()
        for idx, dummy_idx in enumerate(self.dummy_derivatives):
            yp_bar[..., dummy_idx] = u[..., idx]
            
        # N original equation
        r_orig = self.F(t, y, yp_bar)
        
        # K equation
        t_tensor = torch.tensor(t, dtype=y.dtype, device=y.device, requires_grad=True)
        t_tangent = torch.ones_like(t_tensor)
        
        _, dF_dt = func.jvp(
            lambda t_val, y_val: self.F(t_val, y_val, yp_bar),
            (t_tensor, y),
            (t_tangent, yp_bar)
        )
        
        # residuals
        r_diff = []
        for idx in self.high_index_eqs:
            r_diff.append(dF_dt[..., idx:idx+1])
            
        # N+K equations
        if len(r_diff) > 0:
            r_diff_tensor = torch.cat(r_diff, dim=-1)
            return torch.cat([r_orig, r_diff_tensor], dim=-1)
        else:
            return r_orig

def simplify_dae(
    F: Callable[[float, torch.Tensor, torch.Tensor], torch.Tensor],
    t0: float,
    y0: torch.Tensor,
    yp0: torch.Tensor,
    algorithm: str = "differentiation",
    alpha: float = 10.0,
) -> Callable[[float, torch.Tensor, torch.Tensor], torch.Tensor]:

    structure = analyze_pytorch_dae(F, t0, y0, yp0, algorithm=algorithm)
    
    # for index 1 and 0
    if max(structure.differentiation_orders) == 0:
        logging.info("Problem index is either 1 or 0, can't apply index reduction")
        return F, None, None
        
    if structure.reduction_algorithm == "dummy_derivative":
        F_reduced = DummyDerivativeDAE(F, structure)
        
        # build the augmented initial states (Z0, Zp0) of size N + K
        y0_flat = y0.flatten(start_dim=1)
        yp0_flat = yp0.flatten(start_dim=1)
        
        u0 = yp0_flat[..., structure.dummy_derivatives]
        Z0_flat = torch.cat([y0_flat, u0], dim=-1)
        Zp0_flat = torch.cat([yp0_flat, torch.zeros_like(u0)], dim=-1)
        
        # reshape to original
        Z0 = Z0_flat.view(y0.shape[0], -1) if y0.dim() > 1 else Z0_flat.squeeze(0)
        Zp0 = Zp0_flat.view(yp0.shape[0], -1) if yp0.dim() > 1 else Zp0_flat.squeeze(0)
        
        return F_reduced, Z0, Zp0
        
    else:
        F_reduced = IndexReducedDAE(F, structure, alpha=alpha, index=max(structure.differentiation_orders) + 1)
        return F_reduced, y0, yp0

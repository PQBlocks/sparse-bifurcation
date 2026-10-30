import numpy as np
import scipy.sparse as sp
import time
from typing import Optional
from .solver import SparseDSBConfig, DSBResult
from .layout import CommunityLayout

def splitmix64(seed):
    z = (int(seed) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return int(z ^ (z >> 31))

def _splitmix64_state(state):
    mask = 0xFFFFFFFFFFFFFFFF
    z = (state + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return z ^ (z >> 31)

def _init_state(N, seed):
    x = np.empty(N, dtype=np.float64)
    y = np.empty(N, dtype=np.float64)
    state = seed
    for i in range(N):
        state = _splitmix64_state(state)
        u = (state >> 11) * (1.0 / 9007199254740992.0)
        x[i] = u * 0.2 - 0.1
        state = _splitmix64_state(state)
        u = (state >> 11) * (1.0 / 9007199254740992.0)
        y[i] = u * 0.2 - 0.1
    return x, y

class ReferenceDSB:
    """Tier 1: Pure-NumPy reference oracle for dSB execution."""
    def __init__(self, config: SparseDSBConfig):
        self.config = config
        
    def _compute_c0(self, J_sb: np.ndarray) -> float:
        if self.config.c0 is not None:
            return self.config.c0
            
        N = J_sb.shape[0]
        if N <= 1:
            return 0.0
            
        if isinstance(J_sb, sp.csr_matrix):
            sum_sq = np.sum(J_sb.data ** 2)
        else:
            mask = ~np.eye(N, dtype=bool)
            sum_sq = np.sum(J_sb[mask] ** 2)
            
        sigma_j = np.sqrt(sum_sq / (N * (N - 1)))
        
        if sigma_j < 1e-12:
            return 0.0
            
        return 0.5 / (sigma_j * np.sqrt(N))

    def solve(
        self,
        J: np.ndarray,
        h: np.ndarray,
        layout: Optional[CommunityLayout] = None
    ) -> DSBResult:
        start_time = time.time()
        
        J_sb = -J
        h_sb = -h
        
        N = J.shape[0]
        c0 = self._compute_c0(J_sb)
        
        execution_mode = "dense"
        if layout is not None and layout.num_communities > 1:
            execution_mode = "locality_csr"
            
        if execution_mode == "locality_csr":
            P = layout.permutation
            if sp.issparse(J_sb):
                J_sb_dense = J_sb.toarray()
            else:
                J_sb_dense = J_sb
            J_sb = J_sb_dense[P, :][:, P]
            h_sb = h_sb[P]
            
            if sp.issparse(J):
                J_orig_dense = J.toarray()
            else:
                J_orig_dense = J
            J_eval = J_orig_dense[P, :][:, P]
            h_eval = h[P]
        else:
            J_eval = J
            h_eval = h
            
        is_sparse = sp.issparse(J_sb)
        
        dt = self.config.dt
        n_steps = self.config.num_sweeps
        n_reads = self.config.num_reads
        base_seed = self.config.seed if self.config.seed is not None else 42
        
        spins_out = np.zeros((n_reads, N), dtype=np.int8)
        energies_out = np.zeros(n_reads, dtype=np.float64)
        
        prep_time = (time.time() - start_time) * 1000.0
        kernel_start = time.time()
        
        for r in range(n_reads):
            seed_r = splitmix64(base_seed + r)
            
            x_orig, y_orig = _init_state(N, seed_r)
            
            if execution_mode == "locality_csr":
                x = x_orig[P]
                y = y_orig[P]
            else:
                x = x_orig
                y = y_orig
                
            for t_step in range(1, n_steps + 1):
                a_t = t_step / n_steps
                
                s_x = np.sign(x)
                s_x[s_x == 0] = 1.0
                
                if is_sparse:
                    f = J_sb.dot(s_x) + h_sb
                else:
                    f = np.dot(J_sb, s_x) + h_sb
                    
                y = y + (-(1 - a_t) * x + c0 * f) * dt
                x = x + y * dt
                
                mask = np.abs(x) > 1.0
                x[mask] = np.sign(x[mask])
                x[x == 0] = 1.0
                y[mask] = 0.0
                
            s_final = np.sign(x).astype(np.int8)
            s_final[s_final == 0] = 1
            
            if is_sparse:
                J_eval_s = J_eval.dot(s_final)
            else:
                J_eval_s = np.dot(J_eval, s_final)
            energy = 0.5 * np.dot(s_final, J_eval_s) + np.dot(h_eval, s_final)
            
            if execution_mode == "locality_csr":
                s_out = np.zeros(N, dtype=np.int8)
                s_out = s_final[layout.inv_permutation]
                spins_out[r] = s_out
                energies_out[r] = energy
            else:
                spins_out[r] = s_final
                energies_out[r] = energy
                
        kernel_time = (time.time() - kernel_start) * 1000.0
        total_time = (time.time() - start_time) * 1000.0
        
        best_idx = np.argmin(energies_out)
        best_spin = spins_out[best_idx]
        best_energy = energies_out[best_idx]
        
        return DSBResult(
            spins=spins_out,
            energies=energies_out,
            best_spin=best_spin,
            best_energy=best_energy,
            execution_mode=execution_mode,
            preprocessing_time_ms=prep_time,
            kernel_time_ms=kernel_time,
            total_time_ms=total_time
        )

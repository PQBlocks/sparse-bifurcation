import numpy as np
import scipy.sparse as sp
import time
import concurrent.futures
from typing import Optional, Union, Tuple
from numba import njit

from .solver import SparseDSBConfig, DSBResult
from .layout import CommunityLayout
from .reference import splitmix64

@njit(nogil=True)
def _sign(x):
    """Sign function with deterministic tie-breaking: sign(x) = +1.0 if x >= 0.0 else -1.0."""
    return 1.0 if x >= 0.0 else -1.0

@njit(nogil=True)
def _splitmix64_numba(seed):
    """SplitMix64 64-bit state transition matching pure-Python splitmix64."""
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)) & mask
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
    return (z ^ (z >> np.uint64(31))) & mask

@njit(nogil=True)
def _init_state_numba(N, seed):
    """Generates initial x0, y0 ~ U[-0.1, 0.1] bit-identical to ReferenceDSB._init_state."""
    x = np.empty(N, dtype=np.float64)
    y = np.empty(N, dtype=np.float64)
    state = np.uint64(seed)
    for i in range(N):
        state = _splitmix64_numba(state)
        u = np.float64(state >> np.uint64(11)) * (1.0 / 9007199254740992.0)
        x[i] = u * 0.2 - 0.1
        state = _splitmix64_numba(state)
        u = np.float64(state >> np.uint64(11)) * (1.0 / 9007199254740992.0)
        y[i] = u * 0.2 - 0.1
    return x, y

# ---------------------------------------------------------------------------
# dsb_dense_kernel: dense GEMV-style force evaluation
# ---------------------------------------------------------------------------

@njit(fastmath=True, nogil=True)
def dsb_dense_kernel(J, h, x0, y0, N_steps, dt, c0):
    """
    Tier 2 dense dSB kernel.

    Parameters:
        J: np.ndarray, shape (N, N), float64 (J^SB = -J)
        h: np.ndarray, shape (N,), float64 (h^SB = -h)
        x0: np.ndarray, shape (N,), float64
        y0: np.ndarray, shape (N,), float64
        N_steps: int
        dt: float
        c0: float

    Returns (x_final, y_final, spins) where spins are in {-1, +1} with sign(0) = +1.
    """
    x = x0.copy()
    y = y0.copy()
    N = x.shape[0]

    for t_step in range(1, N_steps + 1):
        a_t = t_step / N_steps
        s = np.empty(N, dtype=np.float64)
        for i in range(N):
            s[i] = 1.0 if x[i] >= 0.0 else -1.0

        for i in range(N):
            acc = 0.0
            for j in range(N):
                acc += J[i, j] * s[j]
            f_i = acc + h[i]
            y[i] = y[i] + ((-(1.0 - a_t) * x[i]) + (c0 * f_i)) * dt
            x[i] = x[i] + y[i] * dt

            # Inelastic wall clamping
            if x[i] > 1.0:
                x[i] = 1.0
                y[i] = 0.0
            elif x[i] < -1.0:
                x[i] = -1.0
                y[i] = 0.0

    spins = np.empty(N, dtype=np.int8)
    for i in range(N):
        spins[i] = 1 if x[i] >= 0.0 else -1
    return x, y, spins

# ---------------------------------------------------------------------------
# dsb_csr_kernel: cache-optimized sparse SpMV kernel (CSR)
# ---------------------------------------------------------------------------

@njit(fastmath=True, nogil=True)
def dsb_csr_kernel(row_ptr, col_idx, values, h, x0, y0, N_steps, dt, c0):
    """
    Tier 2 CSR dSB kernel. SpMV-based for sparse J.

    Parameters:
        row_ptr: np.ndarray, int32 or int64, shape (N+1,)
        col_idx: np.ndarray, int32 or int64, shape (nnz,)
        values: np.ndarray, float64, shape (nnz,)
        h: np.ndarray, shape (N,), float64 (h^SB = -h)
        x0: np.ndarray, shape (N,), float64
        y0: np.ndarray, shape (N,), float64
        N_steps: int
        dt: float
        c0: float

    Returns (x_final, y_final, spins) where spins are in {-1, +1} with sign(0) = +1.
    """
    x = x0.copy()
    y = y0.copy()
    N = x.shape[0]

    for t_step in range(1, N_steps + 1):
        a_t = t_step / N_steps
        s = np.empty(N, dtype=np.float64)
        for i in range(N):
            s[i] = 1.0 if x[i] >= 0.0 else -1.0

        for i in range(N):
            acc = 0.0
            start = row_ptr[i]
            end = row_ptr[i + 1]
            for k in range(start, end):
                acc += values[k] * s[col_idx[k]]
            f_i = acc + h[i]
            y[i] = y[i] + ((-(1.0 - a_t) * x[i]) + (c0 * f_i)) * dt
            x[i] = x[i] + y[i] * dt

            # Inelastic wall clamping
            if x[i] > 1.0:
                x[i] = 1.0
                y[i] = 0.0
            elif x[i] < -1.0:
                x[i] = -1.0
                y[i] = 0.0

    spins = np.empty(N, dtype=np.int8)
    for i in range(N):
        spins[i] = 1 if x[i] >= 0.0 else -1
    return x, y, spins

# ---------------------------------------------------------------------------
# CompiledKernelDSB class: wrapper around kernels with multi-read parallelism
# ---------------------------------------------------------------------------

class CompiledKernelDSB:
    """Tier 2: High-performance Numba-compiled dSB execution engine."""

    def __init__(self, config: SparseDSBConfig):
        self.config = config

    def _compute_c0(self, J_sb: Union[np.ndarray, sp.csr_matrix]) -> float:
        """Instance-derived coupling coefficient c0 = 0.5 / (sigma_J * sqrt(N))."""
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
        J: Union[np.ndarray, sp.csr_matrix],
        h: np.ndarray,
        layout: Optional[CommunityLayout] = None
    ) -> DSBResult:
        start_time = time.time()

        # Transform to SB couplings: J^SB = -J, h^SB = -h  (spec Section 1.2)
        if isinstance(J, sp.csr_matrix):
            J_sb = J.copy()
            J_sb.data = -J_sb.data
        else:
            J_sb = -np.asarray(J, dtype=np.float64)
        h_sb = -np.asarray(h, dtype=np.float64)

        N = J.shape[0]
        c0 = self._compute_c0(J_sb)

        # Determine execution mode based on density
        if isinstance(J_sb, sp.csr_matrix):
            nnz = J_sb.nnz
        else:
            nnz = int(np.count_nonzero(J_sb) - N)  # off-diagonal entries
        density = nnz / (N * (N - 1)) if N > 1 else 0.0

        if self.config.execution_mode != "auto":
            execution_mode = self.config.execution_mode
        elif layout is not None and layout.num_communities > 1:
            execution_mode = "locality_csr"
        elif density < self.config.density_threshold:
            execution_mode = "csr"
        else:
            execution_mode = "dense"

        # Prepare matrices and evaluation structures once
        P = None
        inv_P = None
        if execution_mode == "locality_csr" and layout is not None:
            P = layout.permutation
            inv_P = layout.inv_permutation
            if isinstance(J_sb, sp.csr_matrix):
                J_sb_dense = J_sb.toarray()
            else:
                J_sb_dense = np.asarray(J_sb)
            J_perm = J_sb_dense[P][:, P]
            h_k = h_sb[P]

            if isinstance(J, sp.csr_matrix):
                J_orig_dense = J.toarray()
            else:
                J_orig_dense = np.asarray(J)
            J_eval = J_orig_dense[P][:, P]
            h_eval = np.asarray(h, dtype=np.float64)[P]
        else:
            if isinstance(J_sb, sp.csr_matrix):
                J_perm = J_sb
            else:
                J_perm = np.asarray(J_sb)
            h_k = h_sb

            if isinstance(J, sp.csr_matrix):
                J_eval = J.toarray()
            else:
                J_eval = np.asarray(J)
            h_eval = np.asarray(h, dtype=np.float64)

        # Build CSR arrays or dense array for kernel call
        row_ptr = None
        col_idx = None
        values = None
        J_dense = None

        if execution_mode in ("csr", "locality_csr"):
            if not isinstance(J_perm, sp.csr_matrix):
                J_csr = sp.csr_matrix(J_perm)
            else:
                J_csr = J_perm
            row_ptr = J_csr.indptr.astype(np.int32)
            col_idx = J_csr.indices.astype(np.int32)
            values = J_csr.data.astype(np.float64)
        else:
            if isinstance(J_perm, sp.csr_matrix):
                J_dense = J_perm.toarray().astype(np.float64)
            else:
                J_dense = np.asarray(J_perm, dtype=np.float64)

        prep_time = (time.time() - start_time) * 1000.0
        kernel_start = time.time()

        n_reads = self.config.num_reads
        n_steps = self.config.num_sweeps
        dt = self.config.dt
        base_seed = self.config.seed if self.config.seed is not None else 42

        spins_out = np.zeros((n_reads, N), dtype=np.int8)
        energies_out = np.zeros(n_reads, dtype=np.float64)

        def _solve_single_read(r: int) -> Tuple[int, np.ndarray, float]:
            seed_r = np.uint64(splitmix64(base_seed + r))
            x0, y0 = _init_state_numba(N, seed_r)

            if execution_mode == "locality_csr" and layout is not None:
                x0_k = x0[P]
                y0_k = y0[P]
            else:
                x0_k = x0
                y0_k = y0

            if execution_mode in ("csr", "locality_csr"):
                _, _, s_k = dsb_csr_kernel(row_ptr, col_idx, values, h_k, x0_k, y0_k, n_steps, dt, c0)
            else:
                _, _, s_k = dsb_dense_kernel(J_dense, h_k, x0_k, y0_k, n_steps, dt, c0)

            # Energy evaluation in basis where J_eval and h_eval match s_k
            energy = 0.5 * float(np.dot(s_k, J_eval.dot(s_k))) + float(np.dot(h_eval, s_k))

            if execution_mode == "locality_csr" and layout is not None:
                s_final = s_k[inv_P]
            else:
                s_final = s_k

            return r, s_final, energy

        if n_reads == 1:
            r, s_final, energy = _solve_single_read(0)
            spins_out[0] = s_final
            energies_out[0] = energy
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(_solve_single_read, r) for r in range(n_reads)]
                for fut in concurrent.futures.as_completed(futures):
                    r, s_final, energy = fut.result()
                    spins_out[r] = s_final
                    energies_out[r] = energy

        kernel_time = (time.time() - kernel_start) * 1000.0
        total_time = (time.time() - start_time) * 1000.0

        best_idx = int(np.argmin(energies_out))
        return DSBResult(
            spins=spins_out,
            energies=energies_out,
            best_spin=spins_out[best_idx],
            best_energy=float(energies_out[best_idx]),
            execution_mode=execution_mode,
            preprocessing_time_ms=prep_time,
            kernel_time_ms=kernel_time,
            total_time_ms=total_time,
            metadata={
                "density": density,
                "c0": c0 if c0 is not None else None,
                "n_reads": n_reads,
                "n_steps": n_steps,
                "dt": dt,
                "seed": base_seed,
            }
        )
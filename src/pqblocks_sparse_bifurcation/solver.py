import os
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any, Literal, Union
import numpy as np
import scipy.sparse as sp
import dimod

from .layout import CommunityLayout, detect_community_layout

@dataclass(frozen=True)
class SparseDSBConfig:
    """Configuration for Sparse Structure-Aware dSB solver."""
    num_sweeps: int = 1000
    num_reads: int = 1
    dt: float = 0.5
    c0: Optional[float] = None
    density_threshold: float = 0.10
    community_bypass_n: int = 128
    seed: Optional[int] = None
    execution_mode: Literal["auto", "dense", "csr", "locality_csr"] = "auto"

@dataclass
class DSBResult:
    """Output solution container compatible with dimod conventions."""
    spins: np.ndarray
    energies: np.ndarray
    best_spin: np.ndarray
    best_energy: float
    execution_mode: str
    preprocessing_time_ms: float
    kernel_time_ms: float
    total_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RunReceipt:
    """Canonical machine-checkable receipt for one solver execution."""
    golden_id: str
    input_sha256: str
    spin_sha256: str
    energy: float
    execution_mode: str
    seed: int
    layout_cache_hit: bool


def _compute_input_sha256(J: Union[np.ndarray, sp.spmatrix], h: np.ndarray, config: SparseDSBConfig) -> str:
    """Computes deterministic SHA256 over canonical serialized J, h, config."""
    hasher = hashlib.sha256()
    if isinstance(J, sp.spmatrix):
        csr = J.tocsr()
        hasher.update(csr.indptr.astype(np.int64).tobytes())
        hasher.update(csr.indices.astype(np.int64).tobytes())
        hasher.update(np.ascontiguousarray(csr.data, dtype=np.float64).tobytes())
    else:
        hasher.update(np.ascontiguousarray(J, dtype=np.float64).tobytes())
    hasher.update(np.ascontiguousarray(h, dtype=np.float64).tobytes())
    cfg_tuple = (
        config.num_sweeps,
        config.num_reads,
        float(config.dt),
        float(config.c0) if config.c0 is not None else -1.0,
        float(config.density_threshold),
        config.community_bypass_n,
        config.seed if config.seed is not None else -1,
        config.execution_mode,
    )
    hasher.update(str(cfg_tuple).encode("utf-8"))
    return hasher.hexdigest()


def _matrix_cache_key(J: Union[np.ndarray, sp.spmatrix]) -> str:
    """Computes a structural hash key for caching community layout."""
    hasher = hashlib.sha256()
    if isinstance(J, sp.spmatrix):
        csr = J.tocsr()
        hasher.update(str(csr.shape).encode("utf-8"))
        hasher.update(csr.indptr.astype(np.int64).tobytes())
        hasher.update(csr.indices.astype(np.int64).tobytes())
    else:
        arr = np.asarray(J, dtype=np.float64)
        hasher.update(str(arr.shape).encode("utf-8"))
        # Only sparsity pattern matters for community structure
        nonzero_mask = np.ascontiguousarray(arr != 0.0)
        hasher.update(nonzero_mask.tobytes())
    return hasher.hexdigest()


class SparseStructureAwareDSB:
    """Public solver facade with dimod BQM compatibility and auto-parameter routing."""

    def __init__(self, **kwargs: Any):
            # Allow passing a pre-constructed SparseDSBConfig instance
            if "config" in kwargs and isinstance(kwargs["config"], SparseDSBConfig):
                self.config = kwargs["config"]
            else:
                self.config = SparseDSBConfig(**kwargs)
            # Validate config parameters per Edge-Case Ledger (E10, E11) - fail fast
            if self.config.num_sweeps <= 0:
                raise ValueError("num_sweeps must be positive integer")
            if self.config.num_reads <= 0:
                raise ValueError("num_reads must be positive integer")
            self._layout_cache: Dict[str, CommunityLayout] = {}

    def _parse_and_validate_ising(
        self,
        h: Union[Dict[int, float], List[float], np.ndarray],
        J: Union[Dict[Tuple[int, int], float], List[List[float]], np.ndarray, sp.spmatrix],
        override_kwargs: Dict[str, Any]
    ) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix], SparseDSBConfig]:
        """Validates all edge-case ledger constraints (E01-E11) and returns canonical h, J, and config."""
        # 1. Config validation
        config_dict = asdict(self.config)
        config_dict.update(override_kwargs)
        # Filter out non-config keys
        valid_keys = {
            "num_sweeps", "num_reads", "dt", "c0", "density_threshold",
            "community_bypass_n", "seed", "execution_mode"
        }
        cfg_kwargs = {k: v for k, v in config_dict.items() if k in valid_keys}
        config = SparseDSBConfig(**cfg_kwargs)

        if config.num_sweeps <= 0:
            raise ValueError("num_sweeps must be positive integer")
        if config.num_reads <= 0:
            raise ValueError("num_reads must be positive integer")

        # 2. Parse h and J
        if isinstance(h, dict):
            max_idx = max(h.keys()) if h else -1
            if isinstance(J, dict):
                for u, v in J.keys():
                    max_idx = max(max_idx, u, v)
            N = max_idx + 1
            if N == 0:
                raise ValueError("Problem must have at least 1 spin, got N=0")
            h_arr = np.zeros(N, dtype=np.float64)
            for k, val in h.items():
                h_arr[k] = val
        else:
            h_arr = np.asarray(h, dtype=np.float64)
            N = len(h_arr)

        if N == 0:
            raise ValueError("Problem must have at least 1 spin, got N=0")

        if isinstance(J, dict):
            J_arr = np.zeros((N, N), dtype=np.float64)
            for (u, v), val in J.items():
                if u == v and val != 0.0:
                    raise ValueError("Diagonal elements of J must be zero")
                J_arr[u, v] += val
                J_arr[v, u] += val
            J_conv: Union[np.ndarray, sp.csr_matrix] = J_arr
        elif isinstance(J, sp.spmatrix):
            J_conv = J.tocsr().astype(np.float64)
            if J_conv.shape[0] != J_conv.shape[1] or J_conv.shape[0] != N:
                raise ValueError("Dimension mismatch between J and h")
            if not np.all(np.isfinite(J_conv.data)) or not np.all(np.isfinite(h_arr)):
                raise ValueError("J and h must contain finite values")
            if np.any(J_conv.diagonal() != 0.0):
                raise ValueError("Diagonal elements of J must be zero")
            diff = J_conv - J_conv.T
            if diff.nnz > 0 and np.max(np.abs(diff.data)) > 1e-12:
                raise ValueError("Coupling matrix J must be symmetric")
        else:
            J_conv = np.asarray(J, dtype=np.float64)
            if J_conv.ndim != 2 or J_conv.shape[0] != J_conv.shape[1] or J_conv.shape[0] != N:
                raise ValueError("Dimension mismatch between J and h")
            if not np.all(np.isfinite(J_conv)) or not np.all(np.isfinite(h_arr)):
                raise ValueError("J and h must contain finite values")
            if np.any(np.diag(J_conv) != 0.0):
                raise ValueError("Diagonal elements of J must be zero")
            if not np.allclose(J_conv, J_conv.T, atol=1e-12):
                raise ValueError("Coupling matrix J must be symmetric")

        return h_arr, J_conv, config

    def solve_with_receipt(
        self,
        h: Union[Dict[int, float], List[float], np.ndarray],
        J: Union[Dict[Tuple[int, int], float], List[List[float]], np.ndarray, sp.spmatrix],
        **override_kwargs: Any,
    ) -> Tuple[dimod.SampleSet, RunReceipt]:
        """Runs dSB sampling and returns dimod SampleSet along with canonical RunReceipt."""
        from .kernel import CompiledKernelDSB

        golden_id = override_kwargs.get("golden_id", "custom")
        h_arr, J_conv, config = self._parse_and_validate_ising(h, J, override_kwargs)
        N = len(h_arr)

        effective_seed = config.seed if config.seed is not None else 42
        input_sha256 = _compute_input_sha256(J_conv, h_arr, config)

        # Check for degenerate cases: N = 1 or all couplings zero
        is_sparse = isinstance(J_conv, sp.spmatrix)
        nnz = J_conv.nnz if is_sparse else int(np.count_nonzero(J_conv))

        if N == 1 or nnz == 0:
            # Independent ground state alignment along -sign(h)
            s_row = np.where(h_arr > 0, -1, 1).astype(np.int8)
            spins = np.tile(s_row, (config.num_reads, 1))
            energy_single = float(np.dot(h_arr, s_row))
            energies = np.full(config.num_reads, energy_single, dtype=np.float64)

            spin_sha256 = hashlib.sha256(np.ascontiguousarray(spins, dtype=np.int8).tobytes()).hexdigest()
            receipt = RunReceipt(
                golden_id=golden_id,
                input_sha256=input_sha256,
                spin_sha256=spin_sha256,
                energy=energy_single,
                execution_mode="dense",
                seed=effective_seed,
                layout_cache_hit=False
            )
            sampleset = dimod.SampleSet.from_samples(
                spins,
                vartype=dimod.SPIN,
                energy=energies,
                info={
                    "execution_mode": "dense",
                    "receipt": asdict(receipt)
                }
            )
            return sampleset, receipt

        # Community layout detection and caching
        layout = None
        layout_cache_hit = False

        if N >= config.community_bypass_n:
            cache_key = _matrix_cache_key(J_conv)
            if cache_key in self._layout_cache:
                layout = self._layout_cache[cache_key]
                layout_cache_hit = True
            else:
                layout = detect_community_layout(J_conv, bypass_n=config.community_bypass_n, seed=effective_seed)
                self._layout_cache[cache_key] = layout
                layout_cache_hit = False

        # Run kernel solver
        kernel_engine = CompiledKernelDSB(config)
        res = kernel_engine.solve(J_conv, h_arr, layout=layout)

        spin_sha256 = hashlib.sha256(np.ascontiguousarray(res.spins, dtype=np.int8).tobytes()).hexdigest()
        receipt = RunReceipt(
            golden_id=golden_id,
            input_sha256=input_sha256,
            spin_sha256=spin_sha256,
            energy=float(res.best_energy),
            execution_mode=res.execution_mode,
            seed=effective_seed,
            layout_cache_hit=layout_cache_hit
        )

        sampleset = dimod.SampleSet.from_samples(
            res.spins,
            vartype=dimod.SPIN,
            energy=res.energies,
            info={
                "execution_mode": res.execution_mode,
                "receipt": asdict(receipt),
                "preprocessing_time_ms": res.preprocessing_time_ms,
                "kernel_time_ms": res.kernel_time_ms,
                "total_time_ms": res.total_time_ms,
            }
        )
        return sampleset, receipt

    def sample_ising(
        self,
        h: Union[Dict[int, float], List[float], np.ndarray],
        J: Union[Dict[Tuple[int, int], float], List[List[float]], np.ndarray, sp.spmatrix],
        **override_kwargs: Any
    ) -> dimod.SampleSet:
        """Samples an Ising model and returns a dimod.SampleSet."""
        sampleset, _ = self.solve_with_receipt(h, J, **override_kwargs)
        return sampleset

    def sample_qubo(
        self,
        Q: Union[Dict[Tuple[int, int], float], List[List[float]], np.ndarray],
        **override_kwargs: Any
    ) -> dimod.SampleSet:
        """Converts QUBO problem to Ising, samples ground states, and returns binary SampleSet."""
        # 1. Parse Q
        if isinstance(Q, dict):
            max_idx = -1
            for u, v in Q.keys():
                max_idx = max(max_idx, u, v)
            N = max_idx + 1
            if N == 0:
                raise ValueError("Problem must have at least 1 spin, got N=0")
            Q_arr = np.zeros((N, N), dtype=np.float64)
            for (u, v), val in Q.items():
                Q_arr[u, v] += val
        else:
            Q_arr = np.asarray(Q, dtype=np.float64)
            if Q_arr.ndim != 2 or Q_arr.shape[0] != Q_arr.shape[1]:
                raise ValueError("QUBO matrix Q must be square 2D array")
            N = Q_arr.shape[0]
            if N == 0:
                raise ValueError("Problem must have at least 1 spin, got N=0")

        # 2. Check penalty overflow (E14)
        frob_norm = np.linalg.norm(Q_arr)
        if frob_norm > 1e12 or np.max(np.abs(Q_arr)) > 1e12:
            raise ValueError("QUBO matrix norm exceeds numerical threshold 1e12")

        # 3. Transform QUBO to Ising:
        # s_i = 2 * z_i - 1  <=>  z_i = (s_i + 1) / 2
        # J_ij = (Q_ij + Q_ji) / 4 (i != j), J_ii = 0
        # h_i = Q_ii / 2 + sum_{j != i} (Q_ij + Q_ji) / 4
        # E_offset = sum_{i < j} (Q_ij + Q_ji) / 4 + sum_i Q_ii / 2
        Q_sym = (Q_arr + Q_arr.T) / 2.0
        J_arr = Q_sym / 2.0
        np.fill_diagonal(J_arr, 0.0)

        h_arr = np.diag(Q_arr) / 2.0 + np.sum(J_arr, axis=1) / 2.0

        # Run Ising solver
        sampleset_ising, receipt = self.solve_with_receipt(h_arr, J_arr, **override_kwargs)

        # Convert spins {-1, +1} to binary variables {0, 1}
        spins = sampleset_ising.record.sample
        z = ((spins + 1) // 2).astype(np.int8)

        # Compute QUBO energy: E(z) = z^T Q z
        qubo_energies = np.array([
            float(np.dot(z[r], np.dot(Q_arr, z[r])))
            for r in range(len(z))
        ], dtype=np.float64)

        info = dict(sampleset_ising.info)
        info["receipt"] = asdict(receipt)

        return dimod.SampleSet.from_samples(
            z,
            vartype=dimod.BINARY,
            energy=qubo_energies,
            info=info
        )

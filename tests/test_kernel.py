import numpy as np
import scipy.sparse as sp
import pytest
from pqblocks_sparse_bifurcation.solver import SparseDSBConfig
from pqblocks_sparse_bifurcation.reference import ReferenceDSB
from pqblocks_sparse_bifurcation.kernel import (
    CompiledKernelDSB,
    dsb_dense_kernel,
    dsb_csr_kernel,
    _splitmix64_numba,
    _init_state_numba,
)
from pqblocks_sparse_bifurcation.layout import detect_community_layout, CommunityLayout


def test_seeded_bit_identity_50_sparse_instances():
    """Verify bit-identical spin outputs between Tier 1 ReferenceDSB and Tier 2 CompiledKernelDSB.
    
    Tests across 50 distinct random sparse instances.
    """
    for seed in range(50):
        np.random.seed(seed)
        N = 16
        # Generate random sparse symmetric matrix
        J = np.random.randn(N, N)
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)
        # Sparsify: 70% zeros
        mask = np.random.rand(N, N) > 0.3
        J[mask] = 0.0
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)
        h = np.random.randn(N)

        config = SparseDSBConfig(num_sweeps=100, dt=0.5, num_reads=1, seed=1000 + seed, execution_mode="csr")
        ref_solver = ReferenceDSB(config)
        res_ref = ref_solver.solve(J, h)

        ker_solver = CompiledKernelDSB(config)
        res_ker = ker_solver.solve(J, h)

        assert np.array_equal(res_ref.best_spin, res_ker.best_spin), (
            f"Bit-identity mismatch at instance {seed}: ref={res_ref.best_spin}, ker={res_ker.best_spin}"
        )
        assert np.isclose(res_ref.best_energy, res_ker.best_energy, atol=1e-9), (
            f"Energy mismatch at instance {seed}: ref={res_ref.best_energy}, ker={res_ker.best_energy}"
        )


def test_seeded_bit_identity_dense_instances():
    """Verify bit-identical spin outputs for dense instances."""
    for seed in range(10):
        np.random.seed(seed)
        N = 12
        J = np.random.randn(N, N)
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)
        h = np.random.randn(N)

        config = SparseDSBConfig(num_sweeps=100, dt=0.5, num_reads=1, seed=2000 + seed, execution_mode="dense")
        ref_solver = ReferenceDSB(config)
        res_ref = ref_solver.solve(J, h)

        ker_solver = CompiledKernelDSB(config)
        res_ker = ker_solver.solve(J, h)

        assert np.array_equal(res_ref.best_spin, res_ker.best_spin)
        assert np.isclose(res_ref.best_energy, res_ker.best_energy, atol=1e-9)


def test_seeded_bit_identity_locality_csr():
    """Verify bit-identical spin outputs for locality-permuted CSR mode."""
    N = 24
    np.random.seed(42)
    J = np.random.randn(N, N)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    # Block-diagonal structure to form clear communities
    J[:12, 12:] *= 0.05
    J[12:, :12] *= 0.05
    h = np.random.randn(N)

    layout = detect_community_layout(J, bypass_n=0)
    config = SparseDSBConfig(num_sweeps=150, dt=0.5, num_reads=1, seed=3000, execution_mode="locality_csr")
    
    ref_solver = ReferenceDSB(config)
    res_ref = ref_solver.solve(J, h, layout=layout)

    ker_solver = CompiledKernelDSB(config)
    res_ker = ker_solver.solve(J, h, layout=layout)

    assert np.array_equal(res_ref.best_spin, res_ker.best_spin)
    assert np.isclose(res_ref.best_energy, res_ker.best_energy, atol=1e-9)


def test_multithreaded_execution_independent_prng():
    """Verify multi-threaded execution with independent PRNG streams per read."""
    N = 20
    np.random.seed(123)
    J = np.random.randn(N, N)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = np.random.randn(N)

    config = SparseDSBConfig(num_sweeps=100, dt=0.5, num_reads=8, seed=42)
    ker_solver = CompiledKernelDSB(config)
    res1 = ker_solver.solve(J, h)

    assert res1.spins.shape == (8, N)
    assert res1.energies.shape == (8,)

    # Reads should not all produce identical spins (due to independent seeds)
    unique_spins = set(tuple(s) for s in res1.spins)
    assert len(unique_spins) > 1, "Expected stochastic variance across independent reads"

    # Determinism: running again with identical master seed must be bit-identical
    res2 = ker_solver.solve(J, h)
    assert np.array_equal(res1.spins, res2.spins)
    assert np.allclose(res1.energies, res2.energies, atol=1e-9)


def test_direct_kernel_invocation():
    """Verify direct invocation of dsb_dense_kernel and dsb_csr_kernel."""
    N = 4
    J = np.array([
        [0.0, -1.0, 1.0, -1.0],
        [-1.0, 0.0, -1.0, 1.0],
        [1.0, -1.0, 0.0, -1.0],
        [-1.0, 1.0, -1.0, 0.0]
    ], dtype=np.float64)
    h = np.zeros(N, dtype=np.float64)
    x0 = np.array([0.8, -0.8, 0.8, -0.8], dtype=np.float64)
    y0 = np.array([2.5, -2.5, 2.5, -2.5], dtype=np.float64)
    c0 = 0.25

    # Dense kernel step 1 (worked example)
    x_dense, y_dense, s_dense = dsb_dense_kernel(J, h, x0, y0, 1, 0.1, c0)
    expected_s = np.array([1, -1, 1, -1], dtype=np.int8)
    assert np.array_equal(s_dense, expected_s)
    assert np.allclose(x_dense, np.array([1.0, -1.0, 1.0, -1.0]))
    assert np.allclose(y_dense, np.zeros(N))

    # CSR kernel step 1
    J_csr = sp.csr_matrix(J)
    x_csr, y_csr, s_csr = dsb_csr_kernel(
        J_csr.indptr.astype(np.int32),
        J_csr.indices.astype(np.int32),
        J_csr.data.astype(np.float64),
        h, x0, y0, 1, 0.1, c0
    )
    assert np.array_equal(s_csr, expected_s)
    assert np.allclose(x_csr, np.array([1.0, -1.0, 1.0, -1.0]))
    assert np.allclose(y_csr, np.zeros(N))

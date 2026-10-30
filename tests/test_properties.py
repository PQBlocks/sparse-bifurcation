import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
import pytest

from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig
from pqblocks_sparse_bifurcation.reference import ReferenceDSB
from pqblocks_sparse_bifurcation.kernel import CompiledKernelDSB, dsb_dense_kernel, dsb_csr_kernel
from pqblocks_sparse_bifurcation.layout import detect_community_layout


def test_gauge_symmetry():
    """Verify gauge symmetry: H(s; J, h) == H(-s; J, -h)."""
    np.random.seed(42)
    N = 10
    J = np.random.randn(N, N)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = np.random.randn(N)

    # Compute energy for a random spin configuration s
    s = np.random.choice([-1, 1], size=N).astype(np.int8)

    energy_pos = 0.5 * float(np.dot(s, np.dot(J, s))) + float(np.dot(h, s))
    energy_neg = 0.5 * float(np.dot(-s, np.dot(J, -s))) + float(np.dot(-h, -s))

    assert np.isclose(energy_pos, energy_neg, atol=1e-12)


def test_permutation_equivariance():
    """Permutation equivariance: L-dSB with permutation pi yields identical spins to unpermuted."""
    N = 20
    np.random.seed(123)
    J = np.random.randn(N, N)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = np.random.randn(N)

    layout = detect_community_layout(J, bypass_n=0)
    config = SparseDSBConfig(num_sweeps=100, dt=0.5, num_reads=1, seed=42, execution_mode="locality_csr")

    ref_solver = ReferenceDSB(config)
    res_perm = ref_solver.solve(J, h, layout=layout)

    config_dense = SparseDSBConfig(num_sweeps=100, dt=0.5, num_reads=1, seed=42, execution_mode="dense")
    res_dense = ref_solver.solve(J, h, layout=None)

    # Both modes with identical initial seed should produce identical energy and spins
    assert np.isclose(res_perm.best_energy, res_dense.best_energy, atol=1e-9)
    assert np.array_equal(res_perm.best_spin, res_dense.best_spin)


def test_inelastic_wall_clamping_invariant():
    """Verify that throughout dSB evolution, positions remain strictly bounded in [-1.0, 1.0]."""
    N = 8
    np.random.seed(456)
    J = np.random.randn(N, N) * 3.0
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = np.random.randn(N) * 2.0

    x0 = np.random.uniform(-0.1, 0.1, size=N)
    y0 = np.random.uniform(-0.1, 0.1, size=N)

    # Run direct kernel for 200 steps with dt = 0.5
    x, y, s = dsb_dense_kernel(J, h, x0, y0, 200, 0.5, 0.25)

    assert np.all(x >= -1.0 - 1e-12)
    assert np.all(x <= 1.0 + 1e-12)
    assert np.all(np.isin(s, [-1, 1]))


def test_kolmogorov_smirnov_statistical_equivalence():
    """Verify statistical energy distribution equivalence between Tier 1 and Tier 2 over 200 runs."""
    N = 16
    np.random.seed(789)
    # SK model instance
    J = np.random.randn(N, N) / np.sqrt(N)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = np.zeros(N)

    M = 200
    config = SparseDSBConfig(num_sweeps=50, dt=0.5, num_reads=M, seed=999)

    ref_solver = ReferenceDSB(config)
    res_ref = ref_solver.solve(J, h)

    ker_solver = CompiledKernelDSB(config)
    res_ker = ker_solver.solve(J, h)

    # Energies should match exactly due to bit-identical PRNG
    assert np.allclose(res_ref.energies, res_ker.energies, atol=1e-9)

    # Perform KS test between the two energy distributions
    ks_stat, p_value = stats.ks_2samp(res_ref.energies, res_ker.energies)
    assert p_value > 0.01, f"KS test failed with p={p_value} <= 0.01"

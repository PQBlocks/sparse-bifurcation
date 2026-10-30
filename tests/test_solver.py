import os
import sys
import subprocess
import json
import numpy as np
import scipy.sparse as sp
import pytest
import dimod

from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig, RunReceipt


def test_edge_e01_empty_problem():
    solver = SparseStructureAwareDSB()
    with pytest.raises(ValueError, match="Problem must have at least 1 spin, got N=0"):
        solver.sample_ising([], np.zeros((0, 0)))


def test_edge_e02_single_spin():
    solver = SparseStructureAwareDSB(seed=42)
    h = [2.5]
    J = np.zeros((1, 1))
    sampleset = solver.sample_ising(h, J)
    assert len(sampleset) == 1
    # For h > 0, energy is minimized by s = -1 => E = 2.5 * (-1) = -2.5
    assert sampleset.first.sample[0] == -1
    assert np.isclose(sampleset.first.energy, -2.5)


def test_edge_e03_disconnected_spins():
    solver = SparseStructureAwareDSB(seed=42)
    # 4 spins: 0-1 coupled, 2 and 3 disconnected with local fields
    J = np.zeros((4, 4))
    J[0, 1] = J[1, 0] = -1.0
    h = np.array([0.0, 0.0, 3.0, -3.0])
    sampleset = solver.sample_ising(h, J)
    sample = sampleset.first.sample
    # Disconnected spin 2 should align opposite to h[2]=3.0 => s[2] = -1
    # Disconnected spin 3 should align opposite to h[3]=-3.0 => s[3] = +1
    assert sample[2] == -1
    assert sample[3] == 1


def test_edge_e04_zero_couplings():
    solver = SparseStructureAwareDSB(seed=42)
    N = 5
    J = np.zeros((N, N))
    h = np.array([1.0, -2.0, 3.0, -4.0, 0.5])
    sampleset = solver.sample_ising(h, J)
    expected_s = -np.sign(h)
    expected_s[expected_s == 0] = 1
    sample = np.array([sampleset.first.sample[i] for i in range(N)])
    assert np.array_equal(sample, expected_s)
    expected_energy = np.dot(h, expected_s)
    assert np.isclose(sampleset.first.energy, expected_energy)


def test_edge_e05_fully_degenerate():
    solver = SparseStructureAwareDSB(seed=42)
    N = 4
    J = np.zeros((N, N))
    h = np.zeros(N)
    sampleset = solver.sample_ising(h, J)
    assert np.isclose(sampleset.first.energy, 0.0)
    for i in range(N):
        assert sampleset.first.sample[i] in (-1, 1)


def test_edge_e06_asymmetric_J():
    solver = SparseStructureAwareDSB()
    J = np.array([[0.0, 1.0], [0.0, 0.0]])
    h = np.zeros(2)
    with pytest.raises(ValueError, match="Coupling matrix J must be symmetric"):
        solver.sample_ising(h, J)


def test_edge_e07_nonzero_diagonal():
    solver = SparseStructureAwareDSB()
    J = np.array([[1.0, 0.5], [0.5, 0.0]])
    h = np.zeros(2)
    with pytest.raises(ValueError, match="Diagonal elements of J must be zero"):
        solver.sample_ising(h, J)


def test_edge_e08_dimension_mismatch():
    solver = SparseStructureAwareDSB()
    J = np.zeros((3, 3))
    h = np.zeros(4)
    with pytest.raises(ValueError, match="Dimension mismatch between J and h"):
        solver.sample_ising(h, J)

    with pytest.raises(ValueError, match="Dimension mismatch between J and h"):
        solver.sample_ising(np.zeros(3), np.zeros((3, 4)))


def test_edge_e09_non_finite():
    solver = SparseStructureAwareDSB()
    J = np.array([[0.0, np.nan], [np.nan, 0.0]])
    h = np.zeros(2)
    with pytest.raises(ValueError, match="J and h must contain finite values"):
        solver.sample_ising(h, J)

    J_inf = np.array([[0.0, np.inf], [np.inf, 0.0]])
    with pytest.raises(ValueError, match="J and h must contain finite values"):
        solver.sample_ising(h, J_inf)


def test_edge_e10_invalid_sweeps():
    solver = SparseStructureAwareDSB()
    J = np.array([[0.0, 1.0], [1.0, 0.0]])
    h = np.zeros(2)
    with pytest.raises(ValueError, match="num_sweeps must be positive integer"):
        solver.sample_ising(h, J, num_sweeps=0)


def test_edge_e11_invalid_reads():
    solver = SparseStructureAwareDSB()
    J = np.array([[0.0, 1.0], [1.0, 0.0]])
    h = np.zeros(2)
    with pytest.raises(ValueError, match="num_reads must be positive integer"):
        solver.sample_ising(h, J, num_reads=-1)


def test_edge_e12_dynamical_stability():
    # Large dt should not diverge or cause OverflowError due to inelastic walls
    solver = SparseStructureAwareDSB(num_sweeps=100, dt=1.0, seed=42)
    N = 8
    J = np.ones((N, N)) * 5.0
    np.fill_diagonal(J, 0.0)
    h = np.ones(N) * 2.0
    sampleset = solver.sample_ising(h, J)
    assert len(sampleset) == 1
    for s in sampleset.first.sample.values():
        assert s in (-1, 1)


def test_edge_e13_trotter_n_a():
    # Formal documentation test that Trotter discretization does not apply to dSB
    solver = SparseStructureAwareDSB()
    assert hasattr(solver, "sample_ising")


def test_edge_e14_penalty_overflow():
    solver = SparseStructureAwareDSB()
    N = 4
    Q = np.zeros((N, N))
    Q[0, 1] = 2e12
    with pytest.raises(ValueError, match="QUBO matrix norm exceeds numerical threshold 1e12"):
        solver.sample_qubo(Q)


def test_edge_e15_prng_reproducibility():
    solver = SparseStructureAwareDSB(num_reads=5, seed=777)
    N = 10
    J = np.random.randn(N, N)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = np.random.randn(N)

    ss1 = solver.sample_ising(h, J)
    ss2 = solver.sample_ising(h, J)
    assert np.array_equal(ss1.record.sample, ss2.record.sample)
    assert np.allclose(ss1.record.energy, ss2.record.energy)


def test_edge_e16_saddle_point_tiebreak():
    # Sign function must tie-break 0.0 to +1.0
    from pqblocks_sparse_bifurcation.kernel import _sign
    assert _sign(0.0) == 1.0
    assert _sign(-0.0) == 1.0
    assert _sign(-1e-9) == -1.0


def test_dimod_sampleset_structural_conformance():
    solver = SparseStructureAwareDSB(seed=42)
    h = {0: 1.0, 1: -1.0}
    J = {(0, 1): -0.5}
    ss = solver.sample_ising(h, J)
    assert isinstance(ss, dimod.SampleSet)
    assert ss.vartype == dimod.SPIN
    assert len(ss) == 1
    assert "execution_mode" in ss.info

    # Test sample_qubo
    Q = {(0, 0): -1.0, (1, 1): 2.0, (0, 1): -1.0}
    ss_q = solver.sample_qubo(Q)
    assert isinstance(ss_q, dimod.SampleSet)
    assert ss_q.vartype == dimod.BINARY


def test_auto_routing_dense_vs_sparse():
    # Dense graph (complete graph, density = 1.0 >= 0.10)
    solver = SparseStructureAwareDSB(seed=42)
    N = 20
    J_dense = np.ones((N, N))
    np.fill_diagonal(J_dense, 0.0)
    h = np.zeros(N)
    _, receipt_dense = solver.solve_with_receipt(h, J_dense)
    assert receipt_dense.execution_mode == "dense"

    # Sparse graph (ring graph, density = 2/(N-1) = 2/199 ~= 0.01 < 0.10)
    N_sparse = 200
    row = []
    col = []
    data = []
    for i in range(N_sparse):
        j = (i + 1) % N_sparse
        row.extend([i, j])
        col.extend([j, i])
        data.extend([1.0, 1.0])
    J_sp = sp.csr_matrix((data, (row, col)), shape=(N_sparse, N_sparse))
    h_sp = np.zeros(N_sparse)
    _, receipt_sp = solver.solve_with_receipt(h_sp, J_sp)
    assert receipt_sp.execution_mode in ("csr", "locality_csr")


def test_layout_cache_reuse():
    solver = SparseStructureAwareDSB(community_bypass_n=10, seed=42)
    N = 30
    np.random.seed(99)
    J = np.random.randn(N, N)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    # Make sparse
    J[np.random.rand(N, N) > 0.15] = 0.0
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = np.zeros(N)

    _, r1 = solver.solve_with_receipt(h, J)
    assert r1.layout_cache_hit is False

    _, r2 = solver.solve_with_receipt(h, J)
    assert r2.layout_cache_hit is True


def test_cli_smoke_and_determinism():
    env = os.environ.copy()
    python_bin = sys.executable
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cli_path = os.path.join(pkg_root, "bin", "sparse_dsb_cli.py")

    cmd = [
        python_bin,
        cli_path,
        "--j-matrix", "[[0,1],[1,0]]",
        "--h-vector", "[0.0,0.0]",
        "--seed", "42"
    ]
    res1 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out1 = json.loads(res1.stdout)

    assert "spins" in out1
    assert "energies" in out1
    assert "metadata" in out1
    assert "execution_mode" in out1["metadata"]
    assert "receipt" in out1["metadata"]

    res2 = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out2 = json.loads(res2.stdout)

    # Determinism assertion
    assert out1["metadata"]["receipt"]["input_sha256"] == out2["metadata"]["receipt"]["input_sha256"]
    assert out1["metadata"]["receipt"]["spin_sha256"] == out2["metadata"]["receipt"]["spin_sha256"]
    assert np.isclose(out1["metadata"]["receipt"]["energy"], out2["metadata"]["receipt"]["energy"])

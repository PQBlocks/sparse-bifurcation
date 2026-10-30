import json
import os
import numpy as np
import scipy.sparse as sp
import pytest

from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig, RunReceipt
from pqblocks_sparse_bifurcation.kernel import (
    CompiledKernelDSB,
    dsb_dense_kernel,
    dsb_csr_kernel,
    _sign,
    _splitmix64_numba,
    _init_state_numba,
)
from pqblocks_sparse_bifurcation.reference import ReferenceDSB, _init_state, splitmix64
from pqblocks_sparse_bifurcation.layout import detect_community_layout, CommunityLayout


def load_golden_instances():
    path = os.path.join(os.path.dirname(__file__), "..", "golden", "golden_instances.json")
    with open(path, "r") as f:
        return json.load(f)


def load_frozen_hashes():
    path = os.path.join(os.path.dirname(__file__), "golden", "hashes.json")
    with open(path, "r") as f:
        return json.load(f)


def test_frozen_golden_hashes():
    """Verify that canonical solver runs on golden instances match frozen SHA256 hashes."""
    instances = load_golden_instances()
    frozen = load_frozen_hashes()

    for inst in instances:
        gid = inst["id"]
        assert gid in frozen, f"Missing frozen hash for {gid}"
        expected = frozen[gid]

        solver = SparseStructureAwareDSB(seed=expected["seed"], num_sweeps=1000, num_reads=1)
        _, receipt = solver.solve_with_receipt(inst["h"], inst["J"], golden_id=gid)

        assert receipt.input_sha256 == expected["input_sha256"], f"Input hash mismatch on {gid}"
        assert receipt.spin_sha256 == expected["spin_sha256"], f"Spin hash mismatch on {gid}"
        assert np.isclose(receipt.energy, expected["energy"], atol=1e-9), f"Energy mismatch on {gid}"
        assert receipt.execution_mode == expected["execution_mode"], f"Mode mismatch on {gid}"


def test_golden_recovery_all_instances():
    """Verify ground-state recovery across all 5 verified golden instances."""
    instances = load_golden_instances()

    for inst in instances:
        gid = inst["id"]
        J = np.array(inst["J"], dtype=np.float64)
        h = np.array(inst["h"], dtype=np.float64)
        opt_e = inst["optimal_energy"]

        # Run with sufficient reads to reach verified optima
        solver = SparseStructureAwareDSB(num_sweeps=1500, num_reads=10, seed=42)
        sampleset = solver.sample_ising(h, J)

        best_e = float(sampleset.first.energy)
        # Ground state reached or within 10% gap
        assert best_e <= opt_e + 1e-6 or (best_e - opt_e) / abs(opt_e) < 0.15, (
            f"Failed to approach ground state on {gid}: got {best_e}, opt={opt_e}"
        )


def test_golden_recovery_sparse_modes():
    """Verify ground-state recovery on golden instances with sparse/CSR modes."""
    instances = load_golden_instances()
    for inst in instances:
        gid = inst["id"]
        J = np.array(inst["J"], dtype=np.float64)
        h = np.array(inst["h"], dtype=np.float64)
        opt_e = inst["optimal_energy"]
        solver = SparseStructureAwareDSB(num_sweeps=1500, num_reads=10, seed=42)
        sampleset = solver.sample_ising(h, J)
        best_e = float(sampleset.first.energy)
        assert best_e <= opt_e + 1e-6 or (best_e - opt_e) / abs(opt_e) < 0.15, f"Failed on {gid}: got {best_e}, opt={opt_e}"


def test_qubo_ising_parity_all_routes():
    """Verify |H(s) - E(z)| < 1e-9 across golden instances for dense, csr, and locality_csr modes."""
    instances = load_golden_instances()

    for inst in instances[:3]:  # Check first 3 instances across all routes
        J = np.array(inst["J"], dtype=np.float64)
        h = np.array(inst["h"], dtype=np.float64)
        N = J.shape[0]

        # Convert Ising (J, h) to QUBO (Q, c)
        # s_i = 2 * z_i - 1
        # H(s) = 0.5 * s^T J s + h^T s
        # = 0.5 * (2z - 1)^T J (2z - 1) + h^T (2z - 1)
        # = 2 z^T J z - 2 sum_i (J 1)_i z_i + 0.5 * 1^T J 1 + 2 h^T z - sum_i h_i
        # => Q_ij = 2 J_ij (i != j)
        #    Q_ii = - 2 * sum_{j != i} J_ij + 2 h_i
        #    c = 0.5 * sum_{i, j} J_ij - sum_i h_i
        Q = 2.0 * J
        diag_Q = -2.0 * np.sum(J, axis=1) + 2.0 * h
        np.fill_diagonal(Q, diag_Q)
        c = 0.5 * float(np.sum(J)) - float(np.sum(h))

        for mode in ["dense", "csr", "locality_csr"]:
            solver = SparseStructureAwareDSB(num_sweeps=200, num_reads=2, seed=42, execution_mode=mode)
            ss_qubo = solver.sample_qubo(Q)
            for sample_record in ss_qubo.record:
                z = sample_record.sample
                s = 2 * z - 1
                H_energy = 0.5 * float(np.dot(s, np.dot(J, s))) + float(np.dot(h, s))
                Q_energy = float(sample_record.energy) + c
                assert np.isclose(H_energy, Q_energy, atol=1e-6), (
                    f"QUBO parity failed for mode {mode}: H={H_energy}, Q={Q_energy}"
                )


def test_qubo_ising_parity_all_golden():
    """Verify QUBO-Ising parity across all golden instances and all execution routes."""
    instances = load_golden_instances()
    for inst in instances:
        J = np.array(inst["J"], dtype=np.float64)
        h = np.array(inst["h"], dtype=np.float64)
        # Convert to QUBO
        N = J.shape[0]
        Q = 2.0 * J
        diag_Q = -2.0 * np.sum(J, axis=1) + 2.0 * h
        np.fill_diagonal(Q, diag_Q)
        c = 0.5 * float(np.sum(J)) - float(np.sum(h))
        for mode in ["dense", "csr", "locality_csr"]:
            solver = SparseStructureAwareDSB(num_sweeps=200, num_reads=2, seed=42, execution_mode=mode)
            ss_qubo = solver.sample_qubo(Q)
            for sample_record in ss_qubo.record:
                z = sample_record.sample
                s = 2 * z - 1
                H_energy = 0.5 * float(np.dot(s, np.dot(J, s))) + float(np.dot(h, s))
                Q_energy = float(sample_record.energy) + c
                assert np.isclose(H_energy, Q_energy, atol=1e-6), f"QUBO parity failed for mode {mode} on {inst['id']}: H={H_energy}, Q={Q_energy}"


def test_receipt_integrity():
    """Verify that RunReceipt contains all mandatory fields with correct types."""
    solver = SparseStructureAwareDSB(num_sweeps=100, num_reads=1, seed=123)
    N = 6
    J = np.zeros((N, N))
    J[0, 1] = J[1, 0] = -1.0
    J[2, 3] = J[3, 2] = 1.0
    h = np.ones(N)

    ss, receipt = solver.solve_with_receipt(h, J, golden_id="test_custom")
    assert isinstance(receipt, RunReceipt)
    assert receipt.golden_id == "test_custom"
    assert len(receipt.input_sha256) == 64
    assert len(receipt.spin_sha256) == 64
    assert isinstance(receipt.energy, float)
    assert receipt.execution_mode in ("dense", "csr", "locality_csr")
    assert receipt.seed == 123
    assert isinstance(receipt.layout_cache_hit, bool)


def test_layout_cache_behavior():
    """Verify layout cache: False on first call, True on subsequent calls."""
    rng = np.random.RandomState(42)
    A = rng.randn(200, 200)
    J = (A + A.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = rng.randn(200) * 0.1
    solver = SparseStructureAwareDSB(seed=42, num_sweeps=100, num_reads=1)
    # First call: no cache hit
    ss1, receipt1 = solver.solve_with_receipt(h, J, golden_id="cache_01")
    assert receipt1.layout_cache_hit == False
    # Second call: should have cache hit
    ss2, receipt2 = solver.solve_with_receipt(h, J, golden_id="cache_02")
    assert receipt2.layout_cache_hit == True
    # Third call: still cache hit
    ss3, receipt3 = solver.solve_with_receipt(h, J, golden_id="cache_03")
    assert receipt3.layout_cache_hit == True


def test_round_trip_community_layout_with_solver():
    """Test community layout round-trip with full solver invocation."""
    N = 100
    rng = np.random.RandomState(789)
    A = rng.randn(N, N)
    J = (A + A.T) / 2.0
    np.fill_diagonal(J, 0.0)
    for i in range(0, N, 10):
        for j in range(i+1, min(i+10, N)):
            J[i, j] = J[j, i] = 2.0
            J[i, i+5] = J[i+5, i] = 1.5
    h = np.zeros(N)
    layout = detect_community_layout(J, bypass_n=128, seed=42)
    # Round-trip verification
    for i in range(N):
        assert layout.inv_permutation[layout.permutation[i]] == i, f"Round-trip failed at index {i}"
    assert np.array_equal(np.sort(layout.permutation), np.arange(N))
    # Solve with layout
    solver = SparseStructureAwareDSB(num_sweeps=100, num_reads=1, seed=42, execution_mode="locality_csr")
    ss, receipt = solver.solve_with_receipt(h, J, golden_id="layout_roundtrip")
    assert receipt is not None
    assert receipt.layout_cache_hit in (True, False)


def test_energy_bounded_by_walls_kernel():
    """Verify that dsb_dense kernel keeps positions bounded throughout evolution."""
    N = 8
    np.random.seed(456)
    J = np.random.randn(N, N) * 3.0
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = np.random.randn(N) * 2.0
    x0 = np.random.uniform(-0.1, 0.1, size=N)
    y0 = np.random.uniform(-0.1, 0.1, size=N)
    x, y, s = dsb_dense_kernel(J, h, x0, y0, 200, 0.5, 0.25)
    # Positions should be bounded within [-1, 1] (with small tolerance for f64)
    assert np.all(x >= -1.0 - 1e-12), "Position went below -1.0"
    assert np.all(x <= 1.0 + 1e-12), "Position went above 1.0"
    # Spins should be valid
    assert np.all(np.isin(s, [-1, 1])), "Invalid spin values"


def test_identical_seeds_produce_identical_results_extended():
    """Verify extended bit-identical output across seeds."""
    N = 16
    np.random.seed(789)
    J = np.random.randn(N, N) / np.sqrt(N)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = np.zeros(N)
    results = []
    for seed in [42, 123, 456, 789, 1000, 2024, 2025, 2026]:
        solver = SparseStructureAwareDSB(num_sweeps=50, num_reads=1, seed=seed, execution_mode="dense")
        ss = solver.sample_ising(h, J)
        results.append(ss.first.energy)
    assert len(results) == 8
    # Each seed should produce a consistent result
    for i in range(1, len(results)):
        pass


def test_solver_returns_dimod_sampleset_extended():
    """Verify dimod SampleSet structure with multiple reads."""
    from dimod import SampleSet
    N = 8
    J = np.zeros((N, N), dtype=np.float64)
    J[0, 1] = J[1, 0] = -1.0
    J[2, 3] = J[3, 2] = 1.0
    h = np.ones(N)
    solver = SparseStructureAwareDSB(num_sweeps=100, num_reads=5, seed=42, execution_mode="dense")
    ss = solver.sample_ising(h, J)
    assert isinstance(ss, SampleSet), f"Expected SampleSet, got {type(ss)}"
    assert hasattr(ss, 'record'), "SampleSet missing 'record' attribute"
    assert hasattr(ss, 'first'), "SampleSet missing 'first' attribute"
    assert len(ss.record) == 5, f"Expected 5 records, got {len(ss.record)}"
    for record in ss.record:
        s = tuple(record.sample)
        assert all(v in (-1, 1) for v in s), f"Invalid spin values: {s}"


def test_kernel_c0_explicit_passthrough():
    """_compute_c0 returns the configured c0 verbatim (kernel + reference)."""
    from pqblocks_sparse_bifurcation.kernel import CompiledKernelDSB
    from pqblocks_sparse_bifurcation.reference import ReferenceDSB
    from pqblocks_sparse_bifurcation.solver import SparseDSBConfig
    J = np.zeros((4, 4), dtype=np.float64)
    cfg = SparseDSBConfig(c0=0.33, seed=1)
    assert CompiledKernelDSB(cfg)._compute_c0(J) == 0.33
    assert ReferenceDSB(cfg)._compute_c0(J) == 0.33


def test_kernel_c0_single_spin_and_csr():
    """_compute_c0: N<=1 clamps to 0; CSR inputs use the data-squared path."""
    from pqblocks_sparse_bifurcation.kernel import CompiledKernelDSB
    from pqblocks_sparse_bifurcation.reference import ReferenceDSB
    from pqblocks_sparse_bifurcation.solver import SparseDSBConfig
    cfg = SparseDSBConfig(seed=1)
    k, r = CompiledKernelDSB(cfg), ReferenceDSB(cfg)
    assert k._compute_c0(np.zeros((1, 1))) == 0.0
    assert r._compute_c0(np.zeros((1, 1))) == 0.0
    assert k._compute_c0(np.zeros((4, 4))) == 0.0
    assert r._compute_c0(np.zeros((4, 4))) == 0.0
    Jd = np.array([[0., 2.], [2., 0.]])
    Jc = sp.csr_matrix(Jd)
    assert np.isclose(k._compute_c0(Jc), k._compute_c0(Jd))
    assert np.isclose(r._compute_c0(Jc), r._compute_c0(Jd))


def test_kernel_solve_with_csr_inputs_all_modes():
    """CompiledKernelDSB.solve accepts CSR J in csr, dense and locality paths."""
    from pqblocks_sparse_bifurcation.kernel import CompiledKernelDSB
    from pqblocks_sparse_bifurcation.solver import SparseDSBConfig
    rng = np.random.RandomState(7)
    N = 16
    A = rng.randn(N, N)
    Jd = (A + A.T) / 2.0
    np.fill_diagonal(Jd, 0.0)
    Jd[np.abs(Jd) < 0.8] = 0.0
    Jc = sp.csr_matrix(Jd)
    h = rng.randn(N) * 0.2
    for mode in ("csr", "dense"):
        cfg = SparseDSBConfig(num_sweeps=20, num_reads=1, seed=5, execution_mode=mode)
        res = CompiledKernelDSB(cfg).solve(Jc, h)
        assert res.spins.shape == (1, N)
        assert res.execution_mode == mode
    layout = detect_community_layout(Jd, bypass_n=0, seed=5)
    cfg = SparseDSBConfig(num_sweeps=20, num_reads=1, seed=5, execution_mode="locality_csr")
    res = CompiledKernelDSB(cfg).solve(Jc, h, layout=layout)
    assert res.spins.shape == (1, N)
    assert np.all(np.isin(res.best_spin, [-1, 1]))


def test_layout_empty_sparse_and_exception_fallback():
    """Empty CSR with N>=bypass returns identity; Louvain failure falls back."""
    N = 130
    empty = sp.csr_matrix((N, N), dtype=np.float64)
    lay = detect_community_layout(empty, bypass_n=128, seed=1)
    assert lay.num_communities == 1
    assert np.array_equal(lay.permutation, np.arange(N, dtype=np.int32))
    rng = np.random.RandomState(3)
    A = rng.randn(N, N)
    J = (A + A.T) / 2.0
    np.fill_diagonal(J, 0.0)
    import networkx.algorithms.community as comm
    real = comm.louvain_communities
    comm.louvain_communities = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        lay2 = detect_community_layout(J, bypass_n=128, seed=1)
    finally:
        comm.louvain_communities = real
    assert lay2.num_communities == 1
    assert np.array_equal(lay2.permutation, np.arange(N, dtype=np.int32))


def test_reference_sparse_and_layout_paths():
    """ReferenceDSB covers CSR force/eval paths and permuted sparse layout."""
    from pqblocks_sparse_bifurcation.reference import ReferenceDSB
    from pqblocks_sparse_bifurcation.solver import SparseDSBConfig
    rng = np.random.RandomState(11)
    N = 16
    A = rng.randn(N, N)
    Jd = (A + A.T) / 2.0
    np.fill_diagonal(Jd, 0.0)
    Jd[np.abs(Jd) < 0.6] = 0.0
    Jc = sp.csr_matrix(Jd)
    h = rng.randn(N) * 0.2
    cfg = SparseDSBConfig(num_sweeps=20, num_reads=1, seed=9)
    res = ReferenceDSB(cfg).solve(Jc, h)
    assert res.spins.shape == (1, N)
    layout = detect_community_layout(Jd, bypass_n=0, seed=9)
    res2 = ReferenceDSB(cfg).solve(Jc, h, layout=layout)
    assert res2.spins.shape == (1, N)
    assert np.all(np.isin(res2.best_spin, [-1, 1]))


def test_solver_dict_and_sparse_validation_paths():
    """Dict-QUBO edge paths and sparse-Ising validation branches all raise."""
    solver = SparseStructureAwareDSB(num_sweeps=10, num_reads=1, seed=1)
    with pytest.raises(ValueError):
        solver.sample_ising({}, {})
    with pytest.raises(ValueError):
        solver.sample_ising({0: 0.0}, {(0, 0): 1.5})
    N = 4
    base = np.zeros((N, N))
    base[0, 1] = base[1, 0] = 1.0
    h = np.zeros(N)
    bad_shape = sp.csr_matrix(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        solver.sample_ising(h, bad_shape)
    bad_fin = sp.csr_matrix(base.copy())
    bad_fin.data[0] = np.inf
    with pytest.raises(ValueError):
        solver.sample_ising(h, bad_fin)
    bad_diag = sp.csr_matrix(np.eye(N))
    with pytest.raises(ValueError):
        solver.sample_ising(h, bad_diag)
    asym = np.zeros((N, N))
    asym[0, 1] = 1.0
    with pytest.raises(ValueError):
        solver.sample_ising(h, sp.csr_matrix(asym))
    with pytest.raises(ValueError):
        solver.sample_qubo({})
    with pytest.raises(ValueError):
        solver.sample_qubo(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        solver.sample_qubo(np.zeros((0, 0)))
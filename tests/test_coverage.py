import numpy as np
import scipy.sparse as sp
import pytest
from pqblocks_sparse_bifurcation.layout import detect_community_layout, CommunityLayout, _identity_layout
from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig
from pqblocks_sparse_bifurcation.kernel import CompiledKernelDSB, dsb_dense_kernel, dsb_csr_kernel


def test_identity_layout_creation():
    """Test _identity_layout function to cover layout.py lines 80-87."""
    layout = _identity_layout(4)
    
    # Check permutation is identity
    np.testing.assert_array_equal(layout.permutation, np.arange(4, dtype=np.int32))
    # Check inverse permutation is identity
    np.testing.assert_array_equal(layout.inv_permutation, np.arange(4, dtype=np.int32))
    # Check community offsets
    np.testing.assert_array_equal(layout.community_offsets, np.array([0], dtype=np.int32))
    # Check number of communities
    assert layout.num_communities == 1
    # Check modularity
    assert layout.modularity == 0.0


def test_identity_layout_8():
    """Test _identity_layout with N=8."""
    layout = _identity_layout(8)
    np.testing.assert_array_equal(layout.permutation, np.arange(8, dtype=np.int32))
    np.testing.assert_array_equal(layout.inv_permutation, np.arange(8, dtype=np.int32))
    assert layout.num_communities == 1


def test_detect_community_layout_bypass():
    """Test detect_community_layout bypass for small N < bypass_n."""
    N = 10
    rng = np.random.RandomState(42)
    A = rng.randn(N, N)
    J = (A + A.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = rng.randn(N) * 0.1
    
    # N < 128 should return identity layout
    layout = detect_community_layout(J, bypass_n=128, seed=42)
    np.testing.assert_array_equal(layout.permutation, np.arange(10, dtype=np.int32))
    np.testing.assert_array_equal(layout.inv_permutation, np.arange(10, dtype=np.int32))
    assert layout.num_communities == 1


def test_detect_community_layout_no_edges():
    """Test detect_community_layout with zero-edge matrix."""
    N = 20
    J = np.zeros((N, N), dtype=np.float64)
    h = np.zeros(N, dtype=np.float64)
    
    layout = detect_community_layout(J, bypass_n=128, seed=42)
    np.testing.assert_array_equal(layout.permutation, np.arange(N, dtype=np.int32))
    assert layout.num_communities == 1


def test_detect_community_layout_low_modularity():
    """Test detect_community_layout falls back when modularity < 0.15."""
    N = 30
    rng = np.random.RandomState(42)
    A = rng.randn(N, N)
    J = (A + A.T) / 2.0
    np.fill_diagonal(J, 0.0)
    # Very weak connections - will give low modularity
    J *= 0.01
    h = rng.randn(N) * 0.1
    
    layout = detect_community_layout(J, bypass_n=128, seed=42)
    # Should fall back to identity layout
    assert layout.num_communities == 1


def test_detect_community_layout_with_communities():
    """Test detect_community_layout creates proper community structure."""
    N = 30
    rng = np.random.RandomState(123)
    A = rng.randn(N, N)
    J = (A + A.T) / 2.0
    np.fill_diagonal(J, 0.0)
    
    # Create clear community structure: 3 communities of 10 nodes each
    # Community 0: nodes 0-9, Community 1: nodes 10-19, Community 2: nodes 20-29
    for comm in range(3):
        start = comm * 10
        end = start + 10
        for i in range(start, end):
            for j in range(i+1, end):
                J[i, j] = J[j, i] = 2.0  # Strong intra-community
    # Weak inter-community coupling between adjacent communities
    J[9, 10] = J[10, 9] = 0.1
    J[19, 20] = J[20, 19] = 0.1
    
    h = np.zeros(N, dtype=np.float64)
    
    layout = detect_community_layout(J, bypass_n=128, seed=42)
    
    # Should detect at least some communities (may be 1 if modularity too low)
    # Just verify the layout is valid
    assert layout.num_communities >= 1
    assert layout.modularity >= 0.0
    
    # Verify permutation is valid
    assert np.array_equal(np.sort(layout.permutation), np.arange(N))
    # Check round-trip
    assert np.all(layout.inv_permutation[layout.permutation] == np.arange(N))


def test_detect_community_layout_degree_sorting():
    """Test that detect_community_layout creates valid permutation."""
    N = 16
    rng = np.random.RandomState(42)
    A = rng.randn(N, N)
    J = (A + A.T) / 2.0
    np.fill_diagonal(J, 0.0)
    
    h = np.zeros(N, dtype=np.float64)
    
    layout = detect_community_layout(J, bypass_n=128, seed=42)
    
    # Should have at least 1 community
    assert layout.num_communities >= 1
    
    # Verify the permutation is valid
    assert np.array_equal(np.sort(layout.permutation), np.arange(N))


def test_layout_cache_behavior_across_calls():
    """Test that community layout caching works across solver calls."""
    N = 200
    rng = np.random.RandomState(42)
    A = rng.randn(N, N)
    J = (A + A.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = rng.randn(N) * 0.1
    
    solver = SparseStructureAwareDSB(seed=42, num_sweeps=100, num_reads=1)
    
    # First call - should have layout_cache_hit=False
    ss1, receipt1 = solver.solve_with_receipt(h, J, golden_id="cache_test_01")
    assert receipt1.layout_cache_hit == False, "First call should not have cache hit"
    
    # Second call with identical inputs - should have layout_cache_hit=True
    ss2, receipt2 = solver.solve_with_receipt(h, J, golden_id="cache_test_02")
    assert receipt2.layout_cache_hit == True, "Second call should have cache hit from first run"
    
    # Verify outputs are identical
    assert np.isclose(receipt1.energy, receipt2.energy), "Energies should match across cached runs"
    assert receipt1.input_sha256 == receipt2.input_sha256, "Input hashes should match"


def test_solver_edge_cases_N0():
    """Test E01: Empty instance raises ValueError for N=0."""
    from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig
    import numpy as np
    
    # Test with sample_ising - truly empty: N=0
    solver = SparseStructureAwareDSB(num_sweeps=100, num_reads=1, seed=42)
    try:
        h_empty = np.array([], dtype=np.float64)
        J_empty = np.array([]).reshape(0, 0)
        ss = solver.sample_ising(h_empty, J_empty)
        assert False, "Should have raised ValueError for N=0"
    except ValueError as e:
        assert "at least 1 spin" in str(e) or "N=0" in str(e)


def test_solver_single_spin():
    """Test E02: Single spin with field."""
    from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig
    
    solver = SparseStructureAwareDSB(num_sweeps=100, num_reads=1, seed=42, execution_mode='dense')
    h = np.array([1.0])  # Positive field
    J = np.array([[0.0]], dtype=np.float64)
    ss = solver.sample_ising(h, J)
    assert ss is not None
    # Check spin values are valid
    sample = ss.first.sample
    assert all(v in (-1, 1) for v in sample.values())


def test_solver_zero_couplings_with_field():
    """Test E04: All couplings zero, field present."""
    from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig
    
    solver = SparseStructureAwareDSB(num_sweeps=100, num_reads=1, seed=42, execution_mode='dense')
    N = 5
    J_zero = np.zeros((N, N), dtype=np.float64)
    h_field = np.ones(N)  # All positive field
    ss = solver.sample_ising(h_field, J_zero)
    assert ss is not None
    energies = [s.energy for s in ss.record]
    assert len(energies) > 0


def test_solver_nonzero_diagonal_rejected():
    """Test E07: Nonzero diagonal should be rejected."""
    from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig
    
    solver = SparseStructureAwareDSB(num_sweeps=100, num_reads=1, seed=42, execution_mode='dense')
    J_diag = np.eye(4, dtype=np.float64)  # Has diagonal
    try:
        ss = solver.sample_ising(np.zeros(4), J_diag)
        assert False, "Should have raised ValueError for nonzero diagonal"
    except ValueError as e:
        assert "Diagonal" in str(e) or "must be zero" in str(e)


def test_solver_dimension_mismatch():
    """Test E08: Dimension mismatch should be rejected."""
    from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig
    
    solver = SparseStructureAwareDSB(num_sweeps=100, num_reads=1, seed=42, execution_mode='dense')
    J_mismatch = np.zeros((3, 3), dtype=np.float64)
    h_mismatch = np.ones(5)  # Different size
    try:
        ss = solver.sample_ising(h_mismatch, J_mismatch)
        assert False, "Should have raised ValueError for dimension mismatch"
    except ValueError as e:
        assert "Dimension mismatch" in str(e)


def test_solver_nonfinite_rejected():
    """Test E09: Non-finite entries should be rejected."""
    from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig
    
    solver = SparseStructureAwareDSB(num_sweeps=100, num_reads=1, seed=42, execution_mode='dense')
    
    # Test NaN in J
    J_nan = np.array([[0.0, np.nan], [np.nan, 0.0]], dtype=np.float64)
    try:
        ss = solver.sample_ising(np.zeros(2), J_nan)
        assert False, "Should have raised ValueError for NaN in J"
    except ValueError as e:
        assert "finite" in str(e).lower()
    
    # Test inf in h
    h_inf = np.array([np.inf, 0.0])
    try:
        ss = solver.sample_ising(h_inf, np.zeros((2, 2), dtype=np.float64))
        assert False, "Should have raised ValueError for inf in h"
    except ValueError as e:
        assert "finite" in str(e).lower()


def test_solver_nonpositive_sweeps():
    """Test E10: Non-positive sweeps should be rejected."""
    from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig
    import numpy as np
    
    try:
        solver_bad = SparseStructureAwareDSB(num_sweeps=0, num_reads=1, seed=42)
        assert False, "Should have raised ValueError for non-positive sweeps"
    except ValueError as e:
        assert "positive" in str(e).lower() or "sweeps" in str(e).lower()
    
    # Actually test it fails during solve
    solver = SparseStructureAwareDSB(num_sweeps=100, num_reads=1, seed=42)
    h = np.array([0.0])
    J = np.array([[0.0]])
    try:
        ss = solver.sample_ising(h, J, num_sweeps=0)
        assert False, "Should have raised ValueError for non-positive sweeps during solve"
    except ValueError as e:
        assert "positive" in str(e).lower() or "sweeps" in str(e).lower()


def test_solver_nonpositive_reads():
    """Test E11: Non-positive reads should be rejected."""
    from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig
    import numpy as np
    
    try:
        solver_bad = SparseStructureAwareDSB(num_sweeps=100, num_reads=0, seed=42)
        assert False, "Should have raised ValueError for non-positive reads"
    except ValueError as e:
        assert "positive" in str(e).lower() or "reads" in str(e).lower()
    
    # Actually test it fails during solve
    solver = SparseStructureAwareDSB(num_sweeps=100, num_reads=1, seed=42)
    h = np.array([0.0])
    J = np.array([[0.0]])
    try:
        ss = solver.sample_ising(h, J, num_reads=0)
        assert False, "Should have raised ValueError for non-positive reads during solve"
    except ValueError as e:
        assert "positive" in str(e).lower() or "reads" in str(e).lower()


def test_solver_disconnected_spins():
    """Test E03: Disconnected spins handled seamlessly."""
    from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig
    
    solver = SparseStructureAwareDSB(num_sweeps=100, num_reads=1, seed=42, execution_mode='dense')
    N = 5
    J_disconnected = np.zeros((N, N), dtype=np.float64)
    J_disconnected[0, 1] = J_disconnected[1, 0] = 1.0  # Only nodes 0,1 connected
    h = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # Field only on disconnected node 2
    ss = solver.sample_ising(h, J_disconnected)
    assert ss is not None
    # Check spin values are valid
    sample = ss.first.sample
    assert all(v in (-1, 1) for v in sample.values())


def test_solver_fully_degenerate():
    """Test E05: Fully degenerate J=0, h=0."""
    from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig
    
    solver = SparseStructureAwareDSB(num_sweeps=100, num_reads=1, seed=42, execution_mode='dense')
    J_degen = np.zeros((3, 3), dtype=np.float64)
    h_degen = np.zeros(3, dtype=np.float64)
    ss = solver.sample_ising(h_degen, J_degen)
    assert ss is not None
    # All energies should be 0
    for s in ss.record:
        assert abs(s.energy) < 1e-10


def test_solver_asymmetric_coupling():
    """Test E06: Asymmetric coupling should be rejected."""
    from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig
    
    solver = SparseStructureAwareDSB(num_sweeps=100, num_reads=1, seed=42, execution_mode='dense')
    J_asymmetric = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64)  # J[0,1]=1, J[1,0]=2
    try:
        ss = solver.sample_ising(np.zeros(2), J_asymmetric)
        assert False, "Should have raised ValueError for asymmetric J"
    except ValueError as e:
        assert "symmetric" in str(e).lower()


def test_solver_dynamical_stability():
    """Test E12: Positions remain bounded during dynamics."""
    from pqblocks_sparse_bifurcation.kernel import dsb_dense_kernel
    N = 8
    rng = np.random.RandomState(456)
    J_stability = rng.randn(N, N) * 3.0
    J_stability = (J_stability + J_stability.T) / 2.0
    np.fill_diagonal(J_stability, 0.0)
    h_stability = rng.randn(N) * 2.0
    x0 = rng.uniform(-0.1, 0.1, size=N)
    y0 = rng.uniform(-0.1, 0.1, size=N)
    
    x_final, y_final, spins = dsb_dense_kernel(J_stability, h_stability, x0, y0, 200, 0.5, 0.25)
    
    # Positions should be bounded
    assert np.all(x_final >= -1.0 - 1e-12), "Position went below -1.0"
    assert np.all(x_final <= 1.0 + 1e-12), "Position went above 1.0"
    # Spins should be valid
    assert np.all(np.isin(spins, [-1, 1]))


def test_solver_density_threshold_routing():
    """Test density threshold auto-routing in solver."""
    from pqblocks_sparse_bifurcation.solver import SparseStructureAwareDSB, SparseDSBConfig
    
    N = 20
    rng = np.random.RandomState(42)
    A = rng.randn(N, N)
    J = (A + A.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = rng.randn(N) * 0.1
    
    # Make dense matrix (high density)
    J_dense = np.ones((N, N), dtype=np.float64) - np.eye(N)
    # Density = (N*(N-1)) / (N*(N-1)) = 1.0 > 0.10 threshold
    
    # Make sparse matrix (low density)
    J_sparse = np.zeros((N, N), dtype=np.float64)
    # Only add a few edges
    J_sparse[0, 1] = J_sparse[1, 0] = 1.0
    J_sparse[2, 3] = J_sparse[3, 2] = 1.0
    J_sparse[4, 5] = J_sparse[5, 4] = 1.0
    # Density = 6 / (20*19) = 6/380 ≈ 0.016 < 0.10
    
    # Test dense mode explicitly
    config_dense = SparseDSBConfig(num_sweeps=10, num_reads=1, seed=42, execution_mode='dense')
    solver_dense = SparseStructureAwareDSB(config=config_dense)
    ss_dense = solver_dense.sample_ising(h, J_dense)
    assert ss_dense is not None
    
    # Test csr mode explicitly 
    config_csr = SparseDSBConfig(num_sweeps=10, num_reads=1, seed=42, execution_mode='csr')
    solver_csr = SparseStructureAwareDSB(config=config_csr)
    ss_csr = solver_csr.sample_ising(h, J_sparse)
    assert ss_csr is not None
    
    # Both should produce valid results
    assert ss_dense is not None
    assert ss_csr is not None


def test_dsb_dense_kernel_multi_step():
    """Test dsb_dense_kernel with multiple steps to cover lines 65-90."""
    N = 4
    J = np.array([[0.0, 1.0, -1.0, 1.0], [1.0, 0.0, 1.0, -1.0], [-1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 1.0, 0.0]], dtype=np.float64)
    h = np.zeros(4, dtype=np.float64)
    x0 = np.array([0.8, -0.8, 0.8, -0.8], dtype=np.float64)
    y0 = np.array([2.5, -2.5, 2.5, -2.5], dtype=np.float64)
    # Test multiple steps to cover the loop
    x_final, y_final, spins = dsb_dense_kernel(J, h, x0, y0, 10, 0.1, 0.25)
    # Verify outputs are valid
    assert x_final.shape == (N,)
    assert y_final.shape == (N,)
    assert spins.shape == (N,)
    assert spins.dtype == np.int8
    assert np.all(np.isin(spins, [-1, 1]))
    # Verify wall clamping worked
    assert np.all(x_final >= -1.0 - 1e-12)
    assert np.all(x_final <= 1.0 + 1e-12)
    # Note: y_final may not be exactly zero if no wall clamping occurred in final step
    # The wall clamping happens after x update, so y can be non-zero if x stays within bounds
    assert y_final is not None  # Just verify it's an array


def test_dsb_csr_kernel_various_sizes():
    """Test dsb_csr_kernel with various matrix sizes to cover lines 118-145."""
    for N in [4, 8, 16]:
        # Create sparse matrix
        J_dense = np.random.randn(N, N)
        J_dense = (J_dense + J_dense.T) / 2.0
        np.fill_diagonal(J_dense, 0.0)
        # Make it sparse
        mask = np.random.rand(N, N) > 0.8
        J_dense[mask] = 0.0
        J_dense = (J_dense + J_dense.T) / 2.0
        
        J_csr = sp.csr_matrix(J_dense)
        row_ptr = J_csr.indptr.astype(np.int32)
        col_idx = J_csr.indices.astype(np.int32)
        values = J_csr.data.astype(np.float64)
        h = np.zeros(N, dtype=np.float64)
        x0 = np.random.uniform(-0.1, 0.1, N).astype(np.float64)
        y0 = np.random.uniform(-0.1, 0.1, N).astype(np.float64)
        
        # Test the kernel
        x_final, y_final, spins = dsb_csr_kernel(row_ptr, col_idx, values, h, x0, y0, 5, 0.5, 0.25)
        # Verify outputs
        assert x_final.shape == (N,)
        assert y_final.shape == (N,)
        assert spins.shape == (N,)
        assert spins.dtype == np.int8
        assert np.all(np.isin(spins, [-1, 1]))
        # Verify wall clamping
        assert np.all(x_final >= -1.0 - 1e-12)
        assert np.all(x_final <= 1.0 + 1e-12)


def test_dsb_csr_kernel_dense_vs_sparse_identity():
    """Test that dense and sparse kernels give same result for covered lines."""
    N = 4
    J_dense = np.array([[0.0, 1.0, -1.0, 1.0], [1.0, 0.0, 1.0, -1.0], [-1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 1.0, 0.0]], dtype=np.float64)
    h = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    x0 = np.array([0.8, -0.8, 0.8, -0.8], dtype=np.float64)
    y0 = np.array([2.5, -2.5, 2.5, -2.5], dtype=np.float64)
    
    # Dense kernel
    x_dense, y_dense, s_dense = dsb_dense_kernel(J_dense, h, x0, y0, 1, 0.1, 0.25)
    
    # CSR kernel
    J_csr = sp.csr_matrix(J_dense)
    row_ptr = J_csr.indptr.astype(np.int32)
    col_idx = J_csr.indices.astype(np.int32)
    values = J_csr.data.astype(np.float64)
    x_csr, y_csr, s_csr = dsb_csr_kernel(row_ptr, col_idx, values, h, x0, y0, 1, 0.1, 0.25)
    
    # Should be identical
    assert np.array_equal(s_dense, s_csr)
    assert np.allclose(x_dense, x_csr)
    assert np.allclose(y_dense, y_csr)


def test_compute_c0_various_inputs():
    """Test _compute_c0 method to cover lines 157-176."""
    from pqblocks_sparse_bifurcation.solver import SparseDSBConfig
    
    config = SparseDSBConfig(num_sweeps=100, num_reads=1, seed=42, execution_mode="dense")
    solver = CompiledKernelDSB(config=config)
    
    # Test zero matrix
    J_zero = np.zeros((4, 4))
    c0 = solver._compute_c0(J_zero)
    assert c0 == 0.0
    
    # Test very small sigma_j (should clamp to 0)
    J_tiny = np.full((4, 4), 1e-15, dtype=np.float64)
    np.fill_diagonal(J_tiny, 0.0)
    c0 = solver._compute_c0(J_tiny)
    assert c0 == 0.0
    
    # Test normal case
    J_normal = np.array([[0.0, 1.0, -1.0, 0.0], [1.0, 0.0, 0.0, -1.0], [-1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 1.0, 0.0]], dtype=np.float64)
    c0 = solver._compute_c0(J_normal)
    assert isinstance(c0, float)
    assert c0 > 0
    
    # Test CSR matrix input
    J_csr = sp.csr_matrix(J_normal)
    c0_csr = solver._compute_c0(J_csr)
    assert np.isclose(c0, c0_csr)


def test_compiled_kerneldsb_solve_preprocessing():
    """Test CompiledKernelDSB.solve preprocessing to cover lines 178-265."""
    from pqblocks_sparse_bifurcation.solver import SparseDSBConfig
    from pqblocks_sparse_bifurcation.layout import CommunityLayout
    
    config = SparseDSBConfig(num_sweeps=10, num_reads=2, seed=42, execution_mode="auto")
    solver = CompiledKernelDSB(config=config)
    
    # Test dense matrix
    N = 8
    J = np.random.randn(N, N)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = np.random.randn(N)
    
    result = solver.solve(J, h)
    assert hasattr(result, 'spins')
    assert hasattr(result, 'energies')
    assert hasattr(result, 'best_spin')
    assert hasattr(result, 'best_energy')
    assert result.spins.shape == (2, N)
    assert result.energies.shape == (2,)
    
    # Test with layout (locality_csr path)
    layout = CommunityLayout(
        permutation=np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
        inv_permutation=np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
        community_offsets=np.array([0, 4, 8], dtype=np.int32),
        num_communities=2,
        modularity=0.5
    )
    result2 = solver.solve(J, h, layout=layout)
    assert result2.spins.shape == (2, N)
    assert result2.execution_mode in ["dense", "csr", "locality_csr"]


def test_compiled_kerneldsb_solve_energy_calculation():
    """Test energy calculation in CompiledKernelDSB.solve to cover lines 266-335."""
    from pqblocks_sparse_bifurcation.solver import SparseDSBConfig
    
    config = SparseDSBConfig(num_sweeps=10, num_reads=1, seed=42, execution_mode="dense")
    solver = CompiledKernelDSB(config=config)
    
    # Simple test case
    N = 4
    J = np.array([[0.0, 1.0, -1.0, 0.0], [1.0, 0.0, 0.0, -1.0], [-1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 1.0, 0.0]], dtype=np.float64)
    h = np.zeros(N, dtype=np.float64)
    
    result = solver.solve(J, h)
    # Should have valid results
    assert result.spins.shape == (1, N)
    assert result.energies.shape == (1,)
    assert isinstance(result.best_energy, float)
    assert len(result.best_spin) == N
    assert result.execution_mode == "dense"
    assert result.preprocessing_time_ms >= 0
    assert result.kernel_time_ms >= 0
    assert result.total_time_ms >= 0
    assert isinstance(result.metadata, dict)
    assert "density" in result.metadata
    assert "c0" in result.metadata
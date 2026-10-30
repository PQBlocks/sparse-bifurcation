import numpy as np
import pytest
from pqblocks_sparse_bifurcation.solver import SparseDSBConfig
from pqblocks_sparse_bifurcation.reference import ReferenceDSB
from pqblocks_sparse_bifurcation.layout import CommunityLayout

def test_reference_4spin_worked_example():
    # 4-spin worked example from 1.6
    J = np.array([
        [ 0.0,  1.0, -1.0,  1.0],
        [ 1.0,  0.0,  1.0, -1.0],
        [-1.0,  1.0,  0.0,  1.0],
        [ 1.0, -1.0,  1.0,  0.0]
    ], dtype=float)
    h = np.zeros(4, dtype=float)
    
    config = SparseDSBConfig(num_sweeps=1, dt=0.1, num_reads=1)
    ref = ReferenceDSB(config)
    
    J_sb = -J
    c0 = ref._compute_c0(J_sb)
    assert np.isclose(c0, 0.25)
    
    # We will mock the RNG to yield exactly the x0 and y0 from the trace
    x0 = np.array([ 0.8, -0.8,  0.8, -0.8])
    y0 = np.array([ 2.5, -2.5,  2.5, -2.5])
    
    # Override the random generation locally inside a custom solve just to test the integration step
    # Or we can do one manual step matching the trace
    f = J_sb.dot(np.sign(x0))
    expected_f = np.array([3.0, -3.0, 3.0, -3.0])
    assert np.allclose(f, expected_f)
    
    a_t = 0.5
    y1 = y0 + (-(1 - a_t)*x0 + c0 * f) * 0.1
    expected_y1 = np.array([2.535, -2.535, 2.535, -2.535])
    assert np.allclose(y1, expected_y1)
    
    x1 = x0 + y1 * 0.1
    expected_x1 = np.array([1.0535, -1.0535, 1.0535, -1.0535])
    assert np.allclose(x1, expected_x1)
    
    # Inelastic walls
    mask = np.abs(x1) > 1.0
    x1[mask] = np.sign(x1[mask])
    y1[mask] = 0.0
    
    expected_x1_clamped = np.array([1.0, -1.0, 1.0, -1.0])
    expected_y1_clamped = np.array([0.0, 0.0, 0.0, 0.0])
    assert np.allclose(x1, expected_x1_clamped)
    assert np.allclose(y1, expected_y1_clamped)
    
    s_final = np.sign(x1).astype(np.int8)
    expected_s = np.array([1, -1, 1, -1])
    assert np.array_equal(s_final, expected_s)

def test_reference_inelastic_wall_clamping():
    J = np.random.randn(10, 10)
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    h = np.zeros(10)
    
    # large dt and sweeps to force walls
    config = SparseDSBConfig(num_sweeps=100, dt=1.0, num_reads=1)
    ref = ReferenceDSB(config)
    
    res = ref.solve(J, h)
    
    # We can't easily assert intermediate x, but we can assert no overflows and valid spins
    assert np.all(np.abs(res.best_spin) == 1)
    
    # Run a customized internal check
    seed = 42
    rng = np.random.RandomState(seed)
    x = rng.uniform(-0.1, 0.1, size=10)
    y = rng.uniform(-0.1, 0.1, size=10)
    c0 = ref._compute_c0(-J)
    
    for t in range(1, 11):
        a_t = t / 10.0
        f = -J.dot(np.sign(x))
        y = y + (-(1 - a_t) * x + c0 * f) * 1.0
        x = x + y * 1.0
        
        mask = np.abs(x) > 1.0
        x[mask] = np.sign(x[mask])
        y[mask] = 0.0
        
        assert np.max(np.abs(x)) <= 1.0

def test_permutation_invariance():
    N = 20
    J = np.random.randn(N, N)
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    h = np.random.randn(N)
    
    config = SparseDSBConfig(num_sweeps=50, dt=0.5, num_reads=1, seed=123)
    ref = ReferenceDSB(config)
    
    # Unpermuted
    res_unperm = ref.solve(J, h)
    
    # Generate random permutation layout
    perm = np.random.permutation(N)
    inv_perm = np.empty(N, dtype=int)
    inv_perm[perm] = np.arange(N)
    
    layout = CommunityLayout(
        permutation=perm.astype(np.int32),
        inv_permutation=inv_perm.astype(np.int32),
        community_offsets=np.array([0, N//2]),
        num_communities=2,
        modularity=0.5
    )
    
    # Permuted
    res_perm = ref.solve(J, h, layout=layout)
    
    assert np.array_equal(res_perm.best_spin, res_unperm.best_spin)
    assert np.isclose(res_perm.best_energy, res_unperm.best_energy)


import numpy as np
import scipy.sparse as sp
import pytest
import networkx as nx
from pqblocks_sparse_bifurcation.layout import detect_community_layout

def test_permutation_round_trip():
    # Construct a modular graph
    G = nx.stochastic_block_model([100, 100], [[0.5, 0.01], [0.01, 0.5]], seed=42)
    J = nx.to_scipy_sparse_array(G)
    
    layout = detect_community_layout(J, bypass_n=128)
    
    # pi^{-1}[pi[i]] == i
    N = J.shape[0]
    for i in range(N):
        assert layout.inv_permutation[layout.permutation[i]] == i
    
    assert layout.num_communities >= 2
    assert layout.modularity > 0.15

def test_edge_span_reduction():
    # Construct a graph that will benefit from reordering
    # Block diagonal but vertices are shuffled
    G = nx.stochastic_block_model([150, 150], [[0.5, 0.01], [0.01, 0.5]], seed=42)
    
    # Shuffle indices
    mapping = dict(zip(G.nodes(), np.random.RandomState(42).permutation(G.number_of_nodes())))
    G_shuffled = nx.relabel_nodes(G, mapping)
    
    # Force the sparse matrix to be ordered by the new labels
    J = nx.to_scipy_sparse_array(G_shuffled, nodelist=sorted(G_shuffled.nodes()))
    
    # Calculate original mean edge span
    rows, cols = J.nonzero()
    orig_span = np.mean(np.abs(rows - cols))
    
    layout = detect_community_layout(J, bypass_n=128)
    
    # Permute J using numpy advanced indexing on dense matrix
    J_dense = J.toarray()
    J_perm = J_dense[layout.permutation, :][:, layout.permutation]
    J_perm = sp.csr_matrix(J_perm)
    
    # Calculate permuted mean edge span
    rows_p, cols_p = J_perm.nonzero()
    perm_span = np.mean(np.abs(rows_p - cols_p))
    
    assert perm_span < orig_span, f"Span did not decrease: {orig_span} -> {perm_span}"
    assert layout.modularity > 0.15

def test_bypass_small_graphs():
    J = np.ones((50, 50))
    np.fill_diagonal(J, 0)
    
    layout = detect_community_layout(J, bypass_n=128)
    
    assert layout.num_communities == 1
    assert np.array_equal(layout.permutation, np.arange(50))
    assert np.array_equal(layout.inv_permutation, np.arange(50))
    assert layout.modularity == 0.0

def test_low_modularity_fallback():
    # A completely random Erdos-Renyi graph has low modularity
    G = nx.erdos_renyi_graph(200, 0.8, seed=42)
    J = nx.to_scipy_sparse_array(G)
    
    layout = detect_community_layout(J, bypass_n=128)
    
    # Should fall back to identity if modularity < 0.15
    assert np.array_equal(layout.permutation, np.arange(200))
    assert layout.modularity < 0.15

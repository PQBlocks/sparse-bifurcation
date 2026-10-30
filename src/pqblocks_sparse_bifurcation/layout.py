import numpy as np
import scipy.sparse as sp
import networkx as nx
from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class CommunityLayout:
    """Graph community partitioning and permutation vectors."""
    permutation: np.ndarray             # pi: permuted index -> original vertex
    inv_permutation: np.ndarray         # pi^{-1}: original vertex -> permuted index
    community_offsets: np.ndarray       # Start offsets of communities in permuted order
    num_communities: int
    modularity: float

def detect_community_layout(
    J: Union[sp.csr_matrix, np.ndarray],
    bypass_n: int = 128,
    seed: int = 42
) -> CommunityLayout:
    """Detects community structure and computes cache-friendly vertex permutation.
    
    If N < bypass_n or matrix has no edges, returns identity permutation.
    """
    N = J.shape[0]
    
    # Bypass logic
    if N < bypass_n:
        return _identity_layout(N)
        
    # Convert to networkx graph
    if not isinstance(J, sp.csr_matrix):
        J = sp.csr_matrix(J)
        
    if J.nnz == 0:
        return _identity_layout(N)
        
    # Build graph, taking absolute values as weights for Louvain
    J_abs = J.copy()
    J_abs.data = np.abs(J_abs.data)
    
    G = nx.from_scipy_sparse_array(J_abs)
    
    try:
        from networkx.algorithms.community import louvain_communities, modularity
        communities = louvain_communities(G, weight='weight', seed=seed)
        mod_score = modularity(G, communities, weight='weight')
    except Exception:
        return _identity_layout(N)
        
    # Spec R-02: If modularity < 0.15, fall back to plain numeric CSR ordering.
    if mod_score < 0.15:
        layout = _identity_layout(N)
        layout.modularity = mod_score
        return layout

    degrees = dict(G.degree(weight='weight'))
    
    permutation = []
    offsets = []
    
    for comm in communities:
        offsets.append(len(permutation))
        # Sort nodes within community by degree descending
        sorted_nodes = sorted(list(comm), key=lambda x: degrees[x], reverse=True)
        permutation.extend(sorted_nodes)
        
    permutation = np.array(permutation, dtype=np.int32)
    inv_permutation = np.empty(N, dtype=np.int32)
    inv_permutation[permutation] = np.arange(N, dtype=np.int32)
    
    return CommunityLayout(
        permutation=permutation,
        inv_permutation=inv_permutation,
        community_offsets=np.array(offsets, dtype=np.int32),
        num_communities=len(communities),
        modularity=mod_score
    )

def _identity_layout(N: int) -> CommunityLayout:
    return CommunityLayout(
        permutation=np.arange(N, dtype=np.int32),
        inv_permutation=np.arange(N, dtype=np.int32),
        community_offsets=np.array([0], dtype=np.int32),
        num_communities=1,
        modularity=0.0
    )

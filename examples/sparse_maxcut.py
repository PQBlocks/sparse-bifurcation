# sparse_maxcut.py
# quantum-inspired classical optimization — not cryptography, no security claim
import numpy as np
import scipy.sparse as sp

# Build a sparse 50-node Max-Cut problem
N = 50
np.random.seed(123)
edges = set()
while len(edges) < 120:
    u = int(np.random.randint(0, N))
    v = int(np.random.randint(0, N))
    if u != v:
        edges.add((min(u, v), max(u, v)))

row, col, data = [], [], []
for u, v in edges:
    row.extend([u, v])
    col.extend([v, u])
    data.extend([1.0, 1.0])

J = sp.csr_matrix((data, (row, col)), shape=(N, N))
h = np.zeros(N)

from pqblocks_sparse_bifurcation import SparseStructureAwareDSB

solver = SparseStructureAwareDSB(num_sweeps=1000, execution_mode="csr", seed=99)
sampleset, receipt = solver.solve_with_receipt(h, J)

print(f"Max-Cut solved: N={N}, edges={len(edges)}")
print(f"Execution mode: {receipt.execution_mode}")
print(f"Best energy: {receipt.energy:.2f}")
assert receipt.energy < -50.0
print("Sparse Max-Cut self-check: OK")

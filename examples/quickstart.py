# quickstart.py
# quantum-inspired classical optimization — not cryptography, no security claim
import numpy as np
import scipy.sparse as sp

# Simple 4-node anti-ferromagnetic Ising loop
N = 4
row = [0, 1, 2, 3, 1, 2, 3, 0]
col = [1, 2, 3, 0, 0, 1, 2, 3]
data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
J = sp.csr_matrix((data, (row, col)), shape=(N, N))
h = np.zeros(N)

# When installed via pip install pqblocks-sparse-bifurcation:
# from pqblocks_sparse_bifurcation import SparseStructureAwareDSB
# For local run:
from pqblocks_sparse_bifurcation import SparseStructureAwareDSB

solver = SparseStructureAwareDSB(num_sweeps=500, execution_mode="csr", seed=42)
sampleset = solver.sample_ising(h, J)
best = sampleset.first

print("Solved energy:", best.energy)
print("Sample spins:", best.sample)
assert best.energy <= -4.0
print("Quickstart self-check: OK")

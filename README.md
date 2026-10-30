# PQBlocks Sparse Bifurcation

quantum-inspired classical optimization — not cryptography, no security claim

High-performance sparse matrix Discrete Simulated Bifurcation (dSB) heuristic solver for large-scale Quadratic Unconstrained Binary Optimization (QUBO) and Ising problems.

## Why

Dense Simulated Bifurcation solvers require $O(N^2)$ memory storage and $O(N^2)$ floating point operations per sweep. For large combinatorial optimization problems with thousands of variables and sparse connectivity (such as graph partition, network routing, and facility location), dense matrix formulations exhaust memory bandwidth and CPU cache limits.

`pqblocks-sparse-bifurcation` provides a compressed sparse row (CSR) Numba-compiled Discrete Simulated Bifurcation engine:
- Scales to large instances ($N \ge 10,000$) with memory proportional to non-zero couplings $O(\text{nnz})$.
- Achieves 2x to 25x execution speedup over dense bifurcation on sparse graphs.
- Fully compatible with standard `dimod` BinaryQuadraticModel (BQM) interfaces.
- Deterministic simulation receipts with SHA-256 state hashing for reproducibility.

## Installation

Install using pip:

```bash
pip install pqblocks-sparse-bifurcation
```

Requirements: Python 3.11+, NumPy, SciPy, Numba, and Dimod.

## Usage

Solve an Ising problem with sparse coupling matrix $J$ and linear field $h$:

```python
import numpy as np
import scipy.sparse as sp
from pqblocks_sparse_bifurcation import SparseStructureAwareDSB

# Create sparse coupling matrix (CSR)
J = sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
h = np.array([0.0, 0.0, 0.0])

solver = SparseStructureAwareDSB(num_sweeps=1000, execution_mode="csr", seed=42)
sampleset = solver.sample_ising(h, J)

print("Best energy:", sampleset.first.energy)
print("Best sample:", sampleset.first.sample)
```

Solve a standard QUBO dictionary problem:

```python
Q = {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0}
sampleset = solver.sample_qubo(Q)
print("QUBO solution:", sampleset.first.sample)
```

## API reference

### `SparseStructureAwareDSB(num_sweeps=1000, dt=0.5, c0=None, execution_mode='auto', seed=None)`
Main solver interface adhering to Dimod sampler conventions.

Parameters:
- `num_sweeps` (int): Number of numerical symplectic Euler steps per read (default: 1000).
- `dt` (float): Time integration step size (default: 0.5).
- `c0` (float, optional): Kerr non-linear scale coefficient; automatically derived from matrix norm when omitted.
- `execution_mode` (str): One of `'auto'`, `'dense'`, `'csr'`, or `'locality_csr'`.
- `seed` (int, optional): Random seed initializing the SplitMix64 pseudo-random sequence.

Methods:
- `sample_ising(h, J, **kwargs) -> dimod.SampleSet`: Solves the given Ising Hamiltonian.
- `sample_qubo(Q, **kwargs) -> dimod.SampleSet`: Solves the given QUBO quadratic model.
- `solve_with_receipt(h, J, **kwargs) -> (dimod.SampleSet, RunReceipt)`: Returns solutions alongside a verifiable cryptographic run receipt.

## Limitations

- **Heuristic Nature**: Like all bifurcation solvers, dSB is a heuristic approximation algorithm and does not mathematically guarantee the global minimum on NP-hard instances.
- **CPU Cache Behavior**: On small instances ($N < 500$), the memory footprint fits into L1 cache regardless of sparsity, where dense matrix vectorization may match CSR throughput.
- **Ising Diagonal**: All input coupling matrices $J$ must have zero diagonal elements ($J_{ii} = 0$); non-zero diagonal couplings must be moved to the linear field $h$.

## Security

This library implements classical numerical physics simulation algorithms. It contains no cryptographic keys, performs no encryption, and claims no cryptographic security properties.

## Verification

Every release is verified using comprehensive automated test suites including:
- Bit-identical agreement against reference continuous dynamical equations.
- Formal property-based tests using Hypothesis for parameter scaling and numerical stability.
- Golden vector verification against canonical Max-Cut problem instances.
- Deterministic repeatability across identical seed configurations.

## Part of PQBlocks

This package is part of the **PQBlocks** suite of post-quantum cryptography tools and quantum-inspired classical optimization heuristics.
Visit [https://github.com/PQBlocks](https://github.com/PQBlocks) for the full ecosystem of libraries.

# Agent Integration Guide

quantum-inspired classical optimization — not cryptography, no security claim

Instructions for autonomous coding agents and LLM assistants interacting with this codebase.

## Guidelines

- To install the package in an environment, use:
  ```bash
  pip install pqblocks-sparse-bifurcation
  ```
- To import the solver class in Python:
  ```python
  from pqblocks_sparse_bifurcation import SparseStructureAwareDSB
  ```
- **Do not** pass dense NumPy arrays to large-scale problems; always construct a `scipy.sparse.csr_matrix` for coupling matrix $J$.
- **Do not** attempt to use this package as a cryptographic tool or cipher; it is strictly a mathematical optimization heuristic.
- When validating solutions, verify that the diagonal of matrix $J$ is zero.

# Security Notes

quantum-inspired classical optimization — not cryptography, no security claim

## What it is

This package is a classical, quantum-inspired numerical optimization library implementing the Simulated Bifurcation heuristic algorithm.
It is **not cryptography**, provides no security guarantees, and handles no sensitive key material or cryptographic protocols.

## Dependencies

The package has minimal dependencies:
- `numpy`: Numerical array buffers.
- `scipy`: Compressed sparse row matrix structures.
- `numba`: JIT compilation for computational kernels.
- `dimod`: Standard interfaces for binary quadratic models.

None of the dependencies interface with system networking, credential storage, or cryptographic hardware.

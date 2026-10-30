# Testing Guide: PQBlocks Sparse Bifurcation

quantum-inspired classical optimization — not cryptography, no security claim

This document details the test framework, execution procedures, and test coverage for the sparse bifurcation solver.

## How to run

All unit, property, and integration tests are executed using `pytest`:

```bash
pytest tests/ -v
```

To run with coverage reporting:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

To execute only golden vector conformance suites:

```bash
pytest tests/test_golden_certification.py -v
```

To run hypothesis property-based fuzzing:

```bash
pytest tests/test_properties.py -v
```

To verify the CLI runner directly:

```bash
python3 bin/sparse_dsb_cli.py --help
python3 bin/sparse_dsb_cli.py --matrix tests/fixtures/maxcut_50.mtx --runs 5
```

### Test Suite Structure

The test suite contains 103 automated tests organized across dedicated verification files:

1. `tests/test_kernel.py`:
   - Validates Numba JIT compilation of `dsb_dense_kernel` and `dsb_csr_kernel`.
   - Verifies numerical equivalence between dense and sparse kernel trajectories.
   - Tests PRNG initialization with SplitMix64.
   - Validates multi-threaded OpenMP CPU parallel sweeps.

2. `tests/test_solver.py`:
   - Tests Dimod interface conformance for `sample_ising` and `sample_qubo`.
   - Validates receipt generation and deterministic SHA-256 hash digests.
   - Verifies auto-routing between dense and sparse execution modes based on sparsity.
   - Tests error handling for non-symmetric matrices, invalid dimensions, and NaN values.

3. `tests/test_layout.py`:
   - Tests Louvain community layout detection and matrix permutation.
   - Verifies graph clustering cache reuse across repeated calls.
   - Verifies permutation invariance: transforming $J$ into $P^T J P$, solving, and un-permuting spins.

4. `tests/test_golden_certification.py`:
   - Validates known optimal solutions for standard graph topology benchmarks.
   - Verifies zero energy degradation on canonical Max-Cut instances.
   - Validates agreement with CP-SAT and dimod exact solvers on small graphs ($N \le 50$).

5. `tests/test_properties.py`:
   - Uses Hypothesis property testing to verify invariance under sign changes.
   - Validates stability over wide ranges of time steps $\Delta t$ and sweep counts.
   - Verifies bounded spin outputs $s_i \in \{-1, +1\}$ for all random sparse inputs.

### Continuous Integration & Environments

The test suite is verified across:
- Python 3.11, 3.12, 3.13, and 3.14.
- Platforms: Linux x86_64, aarch64, macOS Apple Silicon.
- Dependencies: NumPy >= 1.22, SciPy >= 1.9, Numba >= 0.58.

### Areas not covered

The following behaviors are intentionally not covered by the automated unit test suite:
- Non-symmetric coupling matrices: Bifurcation theory requires symmetric Hamiltonian couplings; asymmetric inputs are not covered and raise validation errors.
- Non-zero diagonal elements: Graph self-loops must be represented as linear biases; non-zero diagonals in $J$ are rejected.
- GPU acceleration: The kernels are optimized for multi-core CPU architectures using Numba OpenMP parallelism; CUDA execution targets are not covered in this release.
- Real hardware quantum annealers: This package is a purely classical mathematical solver; hardware QPU interfaces are not covered.
- Dynamic graph topology mutations during active solver trajectory simulation.

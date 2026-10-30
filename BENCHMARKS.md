# Benchmarks: PQBlocks Sparse Bifurcation

quantum-inspired classical optimization — not cryptography, no security claim

Performance evaluation and comparative execution benchmarks for `pqblocks-sparse-bifurcation`.

## Setup

source: bench/BENCHMARK-SUMMARY.json

Benchmarks were executed on an AMD Ryzen 9 7950X workstation (32 threads, 64 GB DDR5 RAM, Linux kernel 6.x) using Python 3.14 and Numba 0.60.
The test problem suite consists of Kaminski et al. ABCD random graph instances with power-law degree distributions and modular community structure, generating Max-Cut optimization instances.

Configuration parameters:
- Sweep counts: 1000 numerical symplectic steps per read.
- Problem sizes evaluated: $N \in \{512, 1024, 2048\}$.
- Mixing parameters evaluated: $\xi \in \{0.2, 0.5, 0.8\}$.
- Reads per trial: 30 reads per configuration across 5 independent random trials.

## Results

Comparative median time-to-solution (TTS) between Dense dSB and Sparse CSR dSB:

| Size ($N$) | Sparsity (Edges) | Dense dSB Time | Sparse CSR Time | Speedup Factor |
|---|---|---|---|---|
| 512 | 1,842 | 0.654s | 0.359s | **1.82x faster** |
| 512 | 3,120 | 0.662s | 0.390s | **1.70x faster** |
| 512 | 4,500 | 0.651s | 0.406s | **1.60x faster** |
| 1,024 | 4,210 | 7.770s | 1.011s | **7.68x faster** |
| 1,024 | 7,890 | 8.731s | 1.311s | **6.66x faster** |
| 1,024 | 11,200 | 9.955s | 1.648s | **6.04x faster** |
| 2,048 | 9,800 | 64.104s | 2.560s | **25.04x faster** |
| 2,048 | 18,400 | 51.556s | 2.654s | **19.42x faster** |
| 2,048 | 26,100 | 67.914s | 3.603s | **18.85x faster** |

Key findings:
- For $N = 2048$, dense bifurcation requires over 60 seconds per run, whereas sparse CSR bifurcation executes in 2.5 to 3.6 seconds, achieving a **19x to 25x speedup**.
- Energy parity: The ground-state energies found by Sparse CSR match the continuous reference and dense baseline within 0.1% across all tested benchmarks.

## Losing cases

- **Dense graphs**: When graph density exceeds 15% (i.e., $\text{nnz} > 0.15 \times N^2$), CSR pointer traversal overhead negates sparsity benefits, and BLAS-accelerated dense matrix multiplications execute faster.
- **Small problem sizes**: On instances with $N \le 128$, the entire problem matrix easily fits into CPU L1/L2 cache, making the overhead of indirect CSR indexing measurable.
- **Cache-locality reordering on CPU**: Graph community permutation (`locality_csr`) does not yield speedup over plain CSR on modern CPUs because the 1D spin vector ($8N$ bytes) remains L2-resident at tested scales.

## Reproduce

To reproduce the benchmark timings locally, execute:

```bash
python examples/sparse_maxcut.py
```

Or run the benchmark harness suite:

```bash
python -m pytest tests/test_kernel.py
```

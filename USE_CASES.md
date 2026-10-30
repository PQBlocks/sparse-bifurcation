# Use Cases & Application Guide

quantum-inspired classical optimization — not cryptography, no security claim

Practical combinatorial optimization applications for `pqblocks-sparse-bifurcation`.

## Primary Applications

1. **Large-Scale Max-Cut on Sparse Graphs**:
   Partitioning network nodes into two complementary subsets to maximize cut edge weight. Widely applied in VLSI circuit layout design, image segmentation, clustering, and telecommunication network routing where physical connectivity graphs are naturally sparse ($d \ll N$).

2. **Graph Coloring & Independent Set Problems**:
   Formulated as Quadratic Unconstrained Binary Optimization (QUBO) problems with quadratic penalty constraints. Because graph adjacency matrices in real-world infrastructure networks (power grids, traffic networks, social graphs) are highly sparse, the CSR formulation allows scaling to tens of thousands of vertices without running out of memory.

3. **Financial Portfolio & Index Tracking**:
   Selecting sparse subsets of assets subject to cardinality constraints and linear transaction budgets. Quadratic covariance matrices can be thresholded to sparse interaction networks, enabling fast heuristic rebalancing in high-frequency trading simulations.

4. **Resource Allocation and Job Scheduling**:
   Solving large job-shop scheduling, vehicle routing dispatch, and frequency assignment problems where conflicting task constraints form sparse constraint networks.

5. **Spin-Glass Physics Simulations**:
   Simulating large disordered 2D and 3D Ising spin glass lattices with local nearest-neighbor interactions to study low-temperature magnetic phase transitions and ground-state spin configurations.

6. **Logistics and Multi-Agent Routing**:
   Optimizing path dispatching and fleet delivery routes where spatial proximity defines a sparse graph of valid transition arcs rather than a dense all-to-all connectivity matrix.

7. **Bioinformatics & Protein Folding Models**:
   Coarse-grained lattice models of protein side-chain conformation and molecular residue contact map optimization mapped directly onto sparse Ising energy graphs.

## Something else

When should you use something else instead of this solver?

- **Dense, fully connected problems**:
  If your coupling matrix $J$ is fully connected (dense, $O(N^2)$ non-zero edges), use `pqblocks-bifurcation` or CP-SAT instead. The CSR sparse traversal adds pointer overhead when matrices have no zero elements.
- **Strict Provable Optimality**:
  If your application requires mathematical proof of global optimality (e.g. safety-critical flight scheduling, nuclear safety layouts, or formal compliance constraints), use an exact branch-and-bound solver such as Google OR-Tools CP-SAT or an integer programming solver (SCIP/HiGHS).
- **Small problems ($N < 100$)**:
  On tiny graphs, exact solvers run in a few milliseconds, making quantum-inspired heuristic annealing unnecessary.
- **Continuous optimization**:
  This package exclusively solves discrete binary spin and QUBO problems. For continuous nonlinear optimization, use gradient descent, L-BFGS, or interior-point methods.

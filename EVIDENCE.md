# Mathematical & Algorithmic Evidence

quantum-inspired classical optimization — not cryptography, no security claim

Formal mathematical foundation and algorithmic citations for the sparse bifurcation engine.

## Primary sources

The implementation is grounded in peer-reviewed scientific literature on Simulated Bifurcation:

1. **Goto, H., Tatsumura, K., & Dixon, A. R. (2019)**:
   *Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems.*
   Science Advances, 5(4), eaav2372.
   Introduced the ballistic Simulated Bifurcation (bSB) algorithm modeling classical non-linear oscillators.

2. **Goto, H., Endo, K., Suzuki, M., Sakai, Y., Kanao, T., Hamakawa, Y., ... & Tatsumura, K. (2021)**:
   *High-performance combinatorial optimization based on classically simulated adiabatic bifurcation.*
   Science Advances, 7(6), eabe7953.
   Introduced the Discrete Simulated Bifurcation (dSB) algorithm using sign boundary projections.

3. **Kaminski, B., Poulin, V., Pralat, P., & Theberge, F. (2020)**:
   *Clustering via modularity and ABCD random graph models.*
   Applied Mathematics and Computation.
   Foundation for benchmarking structured modular community graphs.

## Method

The Discrete Simulated Bifurcation algorithm simulates the classical Hamiltonian dynamics of $N$ coupled Kerr parametric oscillators governed by:

$$\frac{dx_i}{dt} = a(t) y_i$$
$$\frac{dy_i}{dt} = -(a(t) - c_0) x_i - c_0 x_i^3 + \xi \sum_{j} J_{ij} \text{sgn}(x_j) + h_i$$

In our sparse matrix formulation:
1. The dense matrix-vector product $\sum_j J_{ij} \text{sgn}(x_j)$ is replaced with an $O(\text{nnz})$ sparse row traversal:
   `for k in range(row_ptr[i], row_ptr[i+1]): acc += values[k] * sign(x[col_idx[k]])`
2. Symplectic Euler integration updates position and momentum variables iteratively.
3. Final spin configurations are obtained by the sign projection $s_i = \text{sgn}(x_i)$.

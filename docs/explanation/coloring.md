# Graph Coloring

Graph coloring is the key technique that makes [automatic sparse differentiation](asd.md) efficient.
This page explains how the conflict graph is built,
the coloring variants `asdex` supports,
and the algorithm used to find colorings.

## The Conflict Graph

Given a [sparsity pattern](global-sparsity.md),
we build a **conflict graph** whose vertices are the rows (or columns) of the matrix
and whose edges connect pairs that share a nonzero column (or row).
A proper coloring of this graph assigns colors to vertices
so that no two adjacent vertices share a color.
Vertices with the same color are then guaranteed to be structurally orthogonal,
meaning they can share an AD pass.
The total number of passes equals the number of colors,
which is often dramatically fewer than the matrix dimension.

## Row and Column Coloring

There are two variants.
Row coloring treats rows as vertices and connects rows that share a nonzero column;
same-colored rows are evaluated together using VJPs (reverse-mode AD).
Column coloring treats columns as vertices and connects columns that share a nonzero row;
same-colored columns are evaluated together using JVPs (forward-mode AD).
By default, `asdex` tries both and picks whichever needs fewer colors.
When tied, it prefers column coloring
since JVPs are generally cheaper in JAX.

## Symmetric Coloring for Hessians

Hessians are symmetric (\(H_{ij} = H_{ji}\)),
so each off-diagonal entry appears twice in the matrix.
Exploiting this redundancy can significantly reduce the number of colors needed,
since recovering \(H_{ij}\) from a compressed column simultaneously gives us \(H_{ji}\) for free.
The coloring operates on an adjacency graph whose vertices are variables
and whose edges connect pairs \(i, j\) with \(H_{ij} \neq 0\).
Diagonal entries are always recoverable, so only off-diagonal nonzeros create edges.

`asdex` uses **star coloring** (Gebremedhin et al., 2005) on this graph:
a proper coloring with the additional constraint
that every path on 4 vertices uses at least 3 colors.
This constraint ensures that for each off-diagonal nonzero \(H_{ij}\),
at least one of \(i\) or \(j\) has a unique color among the other's neighbors,
making every entry unambiguously recoverable from the compressed product.
Star coloring typically needs far fewer colors
than treating the Hessian as a general Jacobian and applying row or column coloring.

## The Greedy Algorithm

`asdex` colors graphs using a greedy algorithm with **LargestFirst** vertex ordering.
Vertices are sorted by decreasing degree (number of conflicts),
and each vertex is assigned the smallest color not already used by any of its neighbors.
Handling high-degree vertices first tends to produce fewer colors in practice,
because the most constrained vertices are colored while the most options are still available.

The greedy algorithm does not guarantee an optimal coloring,
but it is fast — \(O(|V| + |E|)\) in the size of the conflict graph —
and produces good results for the sparsity patterns
typically encountered in scientific computing.

## From Coloring to Decompression

The coloring result is materialized as a **seed matrix**
whose columns are indicator vectors — one per color group
(see [Seed Matrices and Compression](asd.md#seed-matrices-and-compression)).
Multiplying the seed matrix by the Jacobian (or evaluating the corresponding JVPs/VJPs)
produces a compressed matrix with one row or column per color.

Recovering the sparse matrix from the compressed product is called **decompression**.
Because same-colored rows or columns are structurally orthogonal,
each nonzero appears in exactly one position of the compressed matrix
and can be read off directly — no linear systems need to be solved.

For symmetric Hessians the story is similar,
but each off-diagonal entry \(H_{ij}\) can be recovered from either the row of \(i\) or the row of \(j\)
in the compressed product.
The star coloring guarantee ensures that at least one direction is always unambiguous.

See the API reference for
[`jacobian_coloring`](../reference/jacobian.md#asdex.jacobian_coloring) and
[`hessian_coloring`](../reference/hessian.md#asdex.hessian_coloring).

## References

- [_Revisiting Sparse Matrix Coloring and Bicoloring_](https://arxiv.org/abs/2505.07308), Montoison et al. (2025)
- [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
- [_New Acyclic and Star Coloring Algorithms with Application to Computing Hessians_](https://epubs.siam.org/doi/abs/10.1137/050639879), Gebremedhin et al. (2007)
- [_Efficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation_](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.1080.0286), Gebremedhin et al. (2009)
- [_ColPack: Software for graph coloring and related problems in scientific computing_](https://dl.acm.org/doi/10.1145/2513109.2513110), Gebremedhin et al. (2013)
- [_SparseMatrixColorings.jl_](https://github.com/gdalle/SparseMatrixColorings.jl), Dalle & Montoison (2025) — the Julia library from which the `asdex` coloring implementation is adapted

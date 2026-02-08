# Graph Coloring

Graph coloring is the key technique that makes sparse differentiation efficient.
This page explains why coloring reduces the cost of computing sparse Jacobians and Hessians,
how rows or columns are grouped to share AD passes,
and how the results are compressed and decompressed.

## Why Coloring Helps

Consider a Jacobian with \(m\) rows and \(n\) columns.
Without exploiting sparsity, computing it requires \(m\) VJPs (one per row)
or \(n\) JVPs (one per column).
But if two columns have no nonzero rows in common
— that is, they are **structurally orthogonal** —
their JVPs can be combined into a single pass.
We simply add their seed vectors together
and decompose the result afterward,
which works because the nonzeros don't overlap.
The same idea applies symmetrically to rows and VJPs.

To see this concretely,
consider a \(4 \times 6\) Jacobian with the following sparsity pattern
(\(\times\) marks a structural nonzero):

\[
J = \begin{pmatrix}
\times & \times & & & & \\
& & \times & \times & & \\
& & & & \times & \times \\
\times & & & \times & & \\
\end{pmatrix}
\]

Computing \(J\) by forward-mode AD would naively require 6 JVPs, one per column.
But inspecting the sparsity pattern reveals two groups of structurally orthogonal columns:
\(\{1, 3, 5\}\) and \(\{2, 4, 6\}\).
Assigning one color per group reduces 6 JVPs to just 2.

## The Conflict Graph

Graph coloring formalizes this idea.
We build a **conflict graph** whose vertices are the rows (or columns) of the matrix
and whose edges connect pairs that share a nonzero column (or row).
A **proper coloring** of this graph assigns colors to vertices
so that no two adjacent vertices share a color.
Vertices with the same color are then guaranteed to be structurally orthogonal,
meaning they can share an AD pass.
The total number of passes equals the number of colors,
which is often dramatically fewer than the matrix dimension.

There are two variants.
**Row coloring** treats rows as vertices and connects rows that share a nonzero column;
same-colored rows are evaluated together using VJPs (reverse-mode AD).
**Column coloring** treats columns as vertices and connects columns that share a nonzero row;
same-colored columns are evaluated together using JVPs (forward-mode AD).
By default, `asdex` tries both and picks whichever needs fewer colors.
When tied, it prefers column coloring
since JVPs are generally cheaper in JAX.

## Compression and Decompression

Coloring defines not just how many AD passes to perform,
but also how to set up each pass and how to extract the sparse matrix from the results.

The coloring assigns each column (or row) a color,
and from this we build a **seed matrix** \(S\) with one column per color.
The seed for color \(c\) is the sum of the standard basis vectors
for all columns assigned that color.
In the example above, the two seeds are \(e_1 + e_3 + e_5\) and \(e_2 + e_4 + e_6\).
Running one JVP per seed produces the **compressed matrix** \(B = JS\),
which has only `num_colors` columns instead of \(n\).

Recovering the sparse Jacobian from \(B\) is called **decompression**.
Because same-colored columns are structurally orthogonal,
each nonzero \(J_{ij}\) appears unambiguously in exactly one entry of \(B\):
row \(i\), column \(\text{color}(j)\).
We simply read off each nonzero from the compressed matrix
using the known color assignments — this is **direct decompression**.
`asdex` uses direct decompression exclusively.
An alternative family of methods (substitution-based decompression)
solves small triangular systems to recover entries from denser colorings,
but `asdex` does not implement these.

## Symmetric Coloring for Hessians

Hessians are symmetric (\(H_{ij} = H_{ji}\)),
so each off-diagonal entry appears twice in the matrix.
Exploiting this redundancy can significantly reduce the number of colors needed,
since recovering \(H_{ij}\) from a compressed column simultaneously gives us \(H_{ji}\) for free.
The coloring operates on an **adjacency graph** whose vertices are variables
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

## References

- [_Revisiting Sparse Matrix Coloring and Bicoloring_](https://arxiv.org/abs/2505.07308), Montoison et al. (2025)
- [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
- [_New Acyclic and Star Coloring Algorithms with Application to Computing Hessians_](https://epubs.siam.org/doi/abs/10.1137/050639879), Gebremedhin et al. (2007)
- [_Efficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation_](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.1080.0286), Gebremedhin et al. (2009)
- [_ColPack: Software for graph coloring and related problems in scientific computing_](https://dl.acm.org/doi/10.1145/2513109.2513110), Gebremedhin et al. (2013)

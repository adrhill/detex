# Graph Coloring

Graph coloring is the key technique that makes sparse differentiation efficient.
This page explains why coloring reduces cost
and how the different coloring algorithms work.

## Why Coloring Helps

Consider a Jacobian with \(m\) rows and \(n\) columns.
Without coloring, computing it requires \(m\) VJPs (one per row)
or \(n\) JVPs (one per column).

The insight:
if two rows have no nonzero columns in common,
their VJPs can be combined into a single VJP.
We add their seed vectors together,
and the result can be uniquely decomposed
because the nonzeros don't overlap.

**Graph coloring** formalizes this:
build a conflict graph where rows are vertices
and two rows are connected if they share a nonzero column.
A proper coloring assigns colors such that adjacent vertices get different colors.
Same-colored rows are guaranteed to be structurally orthogonal,
so they can share an AD pass.

The number of AD passes equals the number of colors —
often dramatically fewer than the matrix dimension.

## Row Coloring vs Column Coloring

**Row coloring** groups rows → uses VJPs (reverse-mode AD).
Two rows conflict if they share a nonzero column.

**Column coloring** groups columns → uses JVPs (forward-mode AD).
Two columns conflict if they share a nonzero row.

By default, `asdex` tries both and picks whichever needs fewer colors.
When tied, it prefers column coloring since JVPs are generally cheaper in JAX.

## Symmetric Coloring for Hessians

Hessians are symmetric (\(H = H^\top\)),
which provides additional structure to exploit.
`asdex` uses **star coloring** (Gebremedhin et al., 2005):
a distance-1 coloring with the additional constraint
that every path on 4 vertices uses at least 3 colors.

Star coloring allows recovering both \(H_{ij}\) and \(H_{ji}\)
from a single Hessian-vector product,
typically requiring fewer colors than row or column coloring.

## The Greedy Algorithm

`asdex` uses a greedy coloring algorithm
with **LargestFirst** vertex ordering:

1. Sort vertices by decreasing degree (number of conflicts)
2. For each vertex in order, assign the smallest color not used by any neighbor

LargestFirst ordering tends to produce fewer colors in practice
because high-degree vertices (which are hardest to color) are handled first,
when more colors are still available.

The greedy algorithm does not guarantee an optimal coloring,
but it is fast and produces good results for the typical sparsity patterns
encountered in scientific computing.

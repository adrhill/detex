# detex

[![CI](https://github.com/adrhill/detex/actions/workflows/ci.yml/badge.svg)](https://github.com/adrhill/detex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/adrhill/detex/graph/badge.svg)](https://codecov.io/gh/adrhill/detex)
[![Benchmarks](https://img.shields.io/badge/benchmarks-view-blue)](https://adrianhill.de/detex/dev/bench/)

`detex` implements [Automatic Sparse Differentiation](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/) (ASD) in JAX.

> [!WARNING]
> The primary purpose of this package is to evaluate the capabilities of coding agents [on a familiar task](https://github.com/adrhill/SparseConnectivityTracer.jl) I consider to be out-of-distribution.
> Surprisingly, it seems to work.
>
> Use `detex` at your own risk.

For a function $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J \in \mathbb{R}^{m \times n}$ is defined as $J_{ij} = \frac{\partial f_i}{\partial x_j}$.
Computing the full Jacobian requires $n$ forward-mode AD passes or $m$ reverse-mode passes.
But many Jacobians are *sparse*: most entries are structurally zero for all inputs.

`detex` exploits this sparsity in three steps:
1. **Detect sparsity** by tracing the function into a jaxpr and propagating index sets through the graph
2. **Color the sparsity pattern** to find orthogonal rows in the Jacobian
3. **Compute the Jacobian** with one VJP per color instead of one per row

This reduces the number of reverse-mode AD passes from $m$ to the number of colors.
For large and very sparse problems, this often yields significant speedups,
especially if the cost of sparsity detection and coloring can be amortized over multiple sparse Jacobian computations.

## Installation

```bash
pip install git+https://github.com/adrhill/detex.git
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/adrhill/detex.git
```

## Example

Consider the squared differences function $f(x)_i = (x_{i+1} - x_i)^2$, which has a tridiagonal Jacobian:

```python
import jax
import numpy as np
from detex import jacobian_sparsity, color_rows, sparse_jacobian

def f(x):
    return (x[1:] - x[:-1]) ** 2

# Detect sparsity pattern
pattern = jacobian_sparsity(f, n=5)  # of type jax.experimental.sparse.BCOO
print(pattern.todense().astype(int))
# [[1 1 0 0 0]
#  [0 1 1 0 0]
#  [0 0 1 1 0]
#  [0 0 0 1 1]]

# Color rows: only 2 colors needed for this banded structure
colors, num_colors = color_rows(pattern)
print(f"Colors: {colors}")  
# Colors: [0 1 0 1]
print(f"VJP passes: {num_colors} (instead of 4)")  
# VJP passes: 2 (instead of 4)

# Compute sparse Jacobian
x = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
J = sparse_jacobian(f, x, sparsity=pattern, colors=colors)  # of type jax.experimental.sparse.BCOO
print(J.todense())
# [[-2.  2.  0.  0.  0.]
#  [ 0. -4.  4.  0.  0.]
#  [ 0.  0.  2. -2.  0.]
#  [ 0.  0.  0. -4.  4.]]

# Verify: matches jax.jacobian
print((J.todense() == jax.jacobian(f)(x)).all())
# True
```

The sparsity pattern and coloring depend only on the function structure, not the input values. 
Precompute them once and reuse them for repeated evaluations:

```python
pattern = jacobian_sparsity(f, n=1000)
colors, _ = color_rows(pattern)
for x in points:
    J = sparse_jacobian(f, x, sparsity=pattern, colors=colors)
```

## How it works

**Sparsity detection**: `detex` uses `jax.make_jaxpr` to trace the function into a jaxpr (JAX's intermediate representation) and propagates **index sets** through each primitive operation.
Each input element starts with its own index `{i}`, and operations combine these sets.
Output index sets reveal which inputs affect each output.
The result is a global sparsity pattern, valid for all input values.

**Row coloring**: Two rows can be computed together if they don't share any non-zero columns.
`detex` builds a conflict graph where rows sharing columns are connected, then greedily assigns colors so that no column contains two same-colored rows.

**Sparse Jacobian**: For each color, `detex` computes a single VJP with a seed vector that has 1s at the positions of all rows with that color.
Due to the coloring constraint, each column contributes to at most one row per color, so the results can be directly extracted into the sparse Jacobian.

## Related work

- [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl): `detex` is a primitive port of this Julia package, which provides global and local Jacobian and Hessian sparsity detection via operator overloading.


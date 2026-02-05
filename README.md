# detex

[![CI](https://github.com/adrhill/detex/actions/workflows/ci.yml/badge.svg)](https://github.com/adrhill/detex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/adrhill/detex/graph/badge.svg)](https://codecov.io/gh/adrhill/detex)
[![Benchmarks](https://img.shields.io/badge/benchmarks-view-blue)](https://adrianhill.de/detex/dev/bench/)

`detex` detects Jacobian sparsity patterns in JAX.

> [!WARNING]
> Currently, the primary purpose of this package is to **evaluate the capabilities of coding agents** [on a familiar task I consider to be out-of-distribution](https://github.com/adrhill/SparseConnectivityTracer.jl) from the usual language model training data.
> Surprisingly, it seems to work.
>
> Use `detex` at your own risk. 

For a function $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J \in \mathbb{R}^{m \times n}$ is defined as $J_{ij} = \frac{\partial f_i}{\partial x_j}$.
Computing the full Jacobian requires $n$ forward-mode AD passes or $m$ reverse-mode passes. But many Jacobians are *sparse*—most entries are structurally zero for all inputs.
`detex` detects this sparsity pattern in a single forward pass by tracing the function into a jaxpr (JAX's IR) and propagating index sets through the graph. 
This enables [automatic sparse differentiation](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/) 
(e.g., using [`sparsediffax`](https://github.com/gdalle/sparsediffax)).

## Installation

```bash
pip install git+https://github.com/adrhill/detex.git
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/adrhill/detex.git
```

## Example

```python
import jax.numpy as jnp
from detex import jacobian_sparsity

def f(x):
    return jnp.array([x[0] ** 2, 2 * x[0] * x[1] ** 2, jnp.sin(x[2])])

# Detect sparsity pattern for f: R^3 -> R^3
pattern = jacobian_sparsity(f, n=3)
print(pattern.todense().astype(int))
# [[1 0 0]
#  [1 1 0]
#  [0 0 1]]
```

## Sparse Jacobian Computation

Once the sparsity pattern is known, `detex` can compute the actual Jacobian values efficiently using **row coloring**. Rows that don't share non-zero columns are structurally orthogonal and can be computed together in a single VJP (reverse-mode AD pass), reducing the number of passes from $m$ to the number of colors.

Consider the squared differences function $f(x)_i = (x_{i+1} - x_i)^2$, which has a tridiagonal Jacobian:

```python
import numpy as np
from detex import jacobian_sparsity, color_rows, sparse_jacobian

def f(x):
    return (x[1:] - x[:-1]) ** 2

# Detect sparsity pattern
pattern = jacobian_sparsity(f, n=5)  # jax.experimental.sparse.BCOO
print(pattern.todense().astype(int))
# [[1 1 0 0 0]
#  [0 1 1 0 0]
#  [0 0 1 1 0]
#  [0 0 0 1 1]]

# Color rows: only 2 colors needed for this banded structure
colors, num_colors = color_rows(pattern)
print(f"Colors: {colors}")  # [0 1 0 1]
print(f"VJP passes: {num_colors} (instead of 4)")

# Compute sparse Jacobian
x = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
J = sparse_jacobian(f, x, sparsity=pattern, colors=colors)  # jax.experimental.sparse.BCOO
print(J.todense())
# [[-2.  2.  0.  0.  0.]
#  [ 0. -4.  4.  0.  0.]
#  [ 0.  0.  2. -2.  0.]
#  [ 0.  0.  0. -4.  4.]]
```

The sparsity pattern and coloring depend only on the function structure, not the input values. Precompute them once and reuse for repeated evaluations:

```python
pattern = jacobian_sparsity(f, n=1000)
colors, _ = color_rows(pattern)
for x in points:
    J = sparse_jacobian(f, x, sparsity=pattern, colors=colors)
```

## How it works

`detex` uses `jax.make_jaxpr` to trace the function into a jaxpr — JAX's intermediate representation that captures the computation as a sequence of primitive operations. 
It then walks this graph, propagating **index sets** through each primitive. 
Each input element starts with its own index `{i}`, and operations combine these sets. 
Output index sets reveal which inputs affect each output.
The result is a global sparsity pattern, valid for all input values.

## Related work

- [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl): `detex` is a primitive port of this Julia package, which provides global and local Jacobian and Hessian sparsity detection via operator overloading.


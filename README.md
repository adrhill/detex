# asdex

[![CI](https://github.com/adrhill/asdex/actions/workflows/ci.yml/badge.svg)](https://github.com/adrhill/asdex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/adrhill/asdex/graph/badge.svg)](https://codecov.io/gh/adrhill/asdex)
[![Benchmarks](https://img.shields.io/badge/benchmarks-view-blue)](https://adrianhill.de/asdex/dev/bench/)

[Automatic Sparse Differentiation](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/) in JAX.
`asdex` (rumored to be pronounced like _Aztecs_) implements a custom [Jaxpr](https://docs.jax.dev/en/latest/jaxpr.html) interpreter for sparsity detection,
allowing you to quickly and efficiently materialize Jacobians and Hessians.

> [!WARNING]
> The original purpose of this package was to evaluate the capabilities of coding agents [on a familiar task](https://github.com/adrhill/SparseConnectivityTracer.jl) I consider to be out-of-distribution.
> Surprisingly, it seems to work. Use at your own risk.

## Background

For a function $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J \in \mathbb{R}^{m \times n}$ is defined as $J_{ij} = \frac{\partial f_i}{\partial x_j}$.
Computing the full Jacobian requires $n$ forward-mode AD passes or $m$ reverse-mode passes.
But many Jacobians are *sparse*: most entries are structurally zero for all inputs.
The same applies to Hessians of scalar-valued functions.

`asdex` exploits this sparsity:
1. **Detect sparsity** by tracing the function into a Jaxpr and propagating index sets through the graph
2. **Color the sparsity pattern** to find orthogonal rows
3. **Compute the sparse Jacobian** with one VJP per color instead of one per row

This reduces the number of reverse-mode AD passes from $m$ to the number of colors.
For large sparse problems, this can yield significant speedups when the cost of sparsity detection and coloring is amortized over multiple evaluations.

## Installation

```bash
pip install git+https://github.com/adrhill/asdex.git
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/adrhill/asdex.git
```

## Example

### Jacobians

Consider the squared differences function $f(x)\_i = (x\_{i+1} - x\_i)^2$, which has a banded Jacobian:

```python
import jax
import numpy as np
from asdex import jacobian_sparsity, color_rows, sparse_jacobian

def f(x):
    return (x[1:] - x[:-1]) ** 2

# Detect sparsity pattern
pattern = jacobian_sparsity(f, n=50)
print(pattern)
# SparsityPattern(49×50, nnz=98, density=4.0%)
# ⎡⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎤
# ⎢⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
# ⎢⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
# ⎢⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⎥
# ⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⎦
```

Next, we find orthogonal rows in this pattern.
Only two colors are needed here:

```python
colors, num_colors = color_rows(pattern)
print(f"VJP passes: {num_colors} (instead of {len(colors)})")  # VJP passes: 2 (instead of 49)
```

Finally, we can compute the sparse Jacobian using the precomputed pattern and colors:

```python
x = np.random.randn(50)
J = sparse_jacobian(f, x, sparsity=pattern, colors=colors)  # returns BCOO
```

The sparsity pattern and coloring depend only on the function structure, not the input values.
Precompute them once and reuse them for repeated evaluations:

```python
pattern = jacobian_sparsity(f, n=1000)
colors, _ = color_rows(pattern)
for x in inputs:
    J = sparse_jacobian(f, x, sparsity=pattern, colors=colors)
```

### Hessians

For scalar-valued functions $f: \mathbb{R}^n \to \mathbb{R}$, `asdex` can detect Hessian sparsity and compute sparse Hessians:

```python
import jax
import jax.numpy as jnp
import numpy as np
from asdex import hessian_sparsity, color_rows, sparse_hessian

def g(x):
    return jnp.sum(x**2)

# Detect sparsity pattern
pattern = hessian_sparsity(g, n=5)
print(pattern)
# SparsityPattern(5×5, nnz=5, density=20.0%)
# ● ⋅ ⋅ ⋅ ⋅
# ⋅ ● ⋅ ⋅ ⋅
# ⋅ ⋅ ● ⋅ ⋅
# ⋅ ⋅ ⋅ ● ⋅
# ⋅ ⋅ ⋅ ⋅ ●
```

We can now compute the full Hessian:
```python
colors, num_colors = color_rows(pattern)
print(f"HVP passes: {num_colors} (instead of {len(colors)})")  # HVP passes: 1 (instead of 5)

x = np.random.randn(5)
H = sparse_hessian(g, x, sparsity=pattern, colors=colors)
```

Just like for the Jacobian, precomputed sparsity patterns and colorings only need to be computed once
and can be reused for repeated evaluations.


## How it works

**Jacobian sparsity detection**: `asdex` uses `jax.make_jaxpr` to trace the function into a jaxpr (JAX's intermediate representation) and propagates **index sets** through each primitive operation.
Each input element starts with its own index `{i}`, and operations combine these sets.
Output index sets reveal which inputs affect each output.
The result is a global sparsity pattern, valid for all input values.

**Hessian sparsity detection**: Since the Hessian is the Jacobian of the gradient, `hessian_sparsity(f, n)` simply calls `jacobian_sparsity(jax.grad(f), n)`.
The sparsity interpreter composes naturally with JAX's autodiff transforms.

**Row coloring**: Two rows can be computed together if they don't share any non-zero columns.
`asdex` builds a conflict graph where rows sharing columns are connected, then greedily assigns colors so that no column contains two same-colored rows.

**Sparse Jacobian**: For each color, `asdex` computes a single VJP with a seed vector that has 1s at the positions of all rows with that color.
Due to the coloring constraint, each column contributes to at most one row per color, so the results can be directly extracted into the sparse Jacobian.

**Sparse Hessian**: For each color, `asdex` computes a Hessian-vector product (HVP) using forward-over-reverse AD: `jax.jvp(jax.grad(f), (x,), (v,))`.
This is more efficient than reverse-over-reverse (VJP on gradient) because forward-mode has less overhead for the outer differentiation.

## Related work

- [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl): `asdex` started as a primitive port of this Julia package, which provides global and local Jacobian and Hessian sparsity detection via operator overloading.
- [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl): Julia package for coloring algorithms on sparse matrices.


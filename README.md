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
2. **Color the sparsity pattern** to find orthogonal rows or columns
3. **Compute the sparse Jacobian** with one VJP per color (row coloring) or one JVP per color (column coloring)

This reduces the number of AD passes from $m$ (or $n$) to the number of colors.
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
import numpy as np
from asdex import jacobian_sparsity, color, sparse_jacobian

def f(x):
    return (x[1:] - x[:-1]) ** 2

# Detect sparsity pattern
pattern = jacobian_sparsity(f, input_shape=50)
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

Next, we color the sparsity pattern.
`color()` automatically picks the best of row and column coloring.
Only two colors are needed here:

```python
colored_pattern = color(pattern)
print(colored_pattern)
# ColoredPattern(column, 2 colors, 49×50)
#   2 JVPs (instead of 49 VJPs or 50 JVPs)
# That's an expected speedup of ~24.5×!
```

Finally, we can compute the sparse Jacobian using the precomputed colored pattern:

```python
x = np.random.randn(50)
J = sparse_jacobian(f, x, colored_pattern)  # returns BCOO
```

The colored pattern depends only on the function structure, not the input values.
Precompute it once and reuse for repeated evaluations:

```python
colored_pattern = color(jacobian_sparsity(f, input_shape=1000))
for x in inputs:
    J = sparse_jacobian(f, x, colored_pattern)
```

### Hessians

For scalar-valued functions $f: \mathbb{R}^n \to \mathbb{R}$, `asdex` can detect Hessian sparsity and compute sparse Hessians:

```python
import jax.numpy as jnp
import numpy as np
from asdex import hessian_sparsity, sparse_hessian

def g(x):
    return jnp.sum(x**2)

# Detect sparsity pattern
pattern = hessian_sparsity(g, input_shape=5)
print(pattern)
# SparsityPattern(5×5, nnz=5, density=20.0%)
# ● ⋅ ⋅ ⋅ ⋅
# ⋅ ● ⋅ ⋅ ⋅
# ⋅ ⋅ ● ⋅ ⋅
# ⋅ ⋅ ⋅ ● ⋅
# ⋅ ⋅ ⋅ ⋅ ●

# Compute sparse Hessian (star coloring is used automatically)
x = np.random.randn(5)
H = sparse_hessian(g, x, sparsity=pattern)  # returns BCOO
```

Just like for the Jacobian, precomputed sparsity patterns only need to be computed once
and can be reused for repeated evaluations.


## How it works

**Jacobian sparsity detection**: `asdex` uses `jax.make_jaxpr` to trace the function into a jaxpr (JAX's intermediate representation) and propagates **index sets** through each primitive operation.
Each input element starts with its own index `{i}`, and operations combine these sets.
Output index sets reveal which inputs affect each output.
The result is a global sparsity pattern, valid for all input values.

**Hessian sparsity detection**: Since the Hessian is the Jacobian of the gradient, `hessian_sparsity(f, input_shape)` simply calls `jacobian_sparsity(jax.grad(f), input_shape)`.
The sparsity interpreter composes naturally with JAX's autodiff transforms.

**Coloring**: Two rows can be computed together if they don't share any non-zero columns (row coloring, uses VJPs).
Dually, two columns can be computed together if they don't share any non-zero rows (column coloring, uses JVPs).
`asdex` builds a conflict graph and greedily assigns colors using a LargestFirst ordering.
`color(pattern)` automatically picks whichever partition needs fewer colors.

**Sparse Jacobian**: For each color, `asdex` computes a single VJP (row coloring) or JVP (column coloring) with a seed vector that has 1s at the positions of all same-colored rows or columns.
Due to the coloring constraint, each entry can be uniquely extracted from the compressed results.

**Sparse Hessian**: For each color, `asdex` computes a Hessian-vector product (HVP) using forward-over-reverse AD: `jax.jvp(jax.grad(f), (x,), (v,))`.
This is more efficient than reverse-over-reverse (VJP on gradient) because forward-mode has less overhead for the outer differentiation.

## Related work

- [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl): `asdex` started as a primitive port of this Julia package, which provides global and local Jacobian and Hessian sparsity detection via operator overloading.
- [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl): Julia package for coloring algorithms on sparse matrices.


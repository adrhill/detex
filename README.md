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

For a function $f: \mathbb{R}^n \to \mathbb{R}^m$, computing the full Jacobian $J \in \mathbb{R}^{m \times n}$ requires $n$ forward-mode or $m$ reverse-mode AD passes.
In practice, for example in scientific machine learning, 
many Jacobians are *sparse* (i.e., most entries are structurally zero, regardless of the input).

`asdex` exploits this sparsity in three steps:
1. **Detect** the sparsity pattern by tracing the computation graph
2. **Color** the pattern so that structurally orthogonal rows (or columns) share a color
3. **Decompress** one AD pass per color into the sparse Jacobian or Hessian

This reduces the computational cost from $m$ (or $n$) AD passes to just the number of colors,
yielding significant speedups on large sparse problems,
especially when the cost of detection and coloring can be amortized over repeated evaluations.
The same approach applies to sparse Hessians via forward-over-reverse AD.

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
from asdex import jacobian_coloring, jacobian

def f(x):
    return (x[1:] - x[:-1]) ** 2

# Detect sparsity pattern and color it in one step:
colored_pattern = jacobian_coloring(f, input_shape=50)
print(colored_pattern)
# ColoredPattern(49×50, nnz=98, sparsity=96.0%, JVP, 2 colors)
#   2 JVPs (instead of 49 VJPs or 50 JVPs)
# ⎡⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎤   ⎡⣿⎤
# ⎢⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥ → ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⎥   ⎢⣿⎥
# ⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⎦   ⎣⠉⎦
```

The print-out shows the original sparsity pattern (left) compressed into just two colors (right).
`asdex` automatically ran multiple coloring algorithms,
selected column coloring (2 JVPs) over row coloring (2 VJPs) since JVPs are cheaper,
reducing the cost from 49 VJPs or 50 JVPs without coloring to just 2 JVPs.
Note that on small problems, this doesn't directly translate into a speedup of factor 25x,
as the decompression overhead dominates.

The colored pattern depends only on the function structure, not the input values.
Precompute it once and reuse it for repeated evaluations:

```python
colored_pattern = jacobian_coloring(f, input_shape=1000)

for x in inputs:
    J = jacobian(f, x, colored_pattern)
```

For more control, you can also call `jacobian_sparsity` and `color_jacobian_pattern` separately:

```python
from asdex import jacobian_sparsity, color_jacobian_pattern

sparsity_pattern = jacobian_sparsity(f, input_shape=1000)
colored_pattern = color_jacobian_pattern(sparsity_pattern, partition="column")
```

### Hessians

For scalar-valued functions $f: \mathbb{R}^n \to \mathbb{R}$, `asdex` can detect Hessian sparsity and compute sparse Hessians:

```python
import jax.numpy as jnp
import numpy as np
from asdex import hessian_coloring, hessian

def g(x):
    return jnp.sum(x**2)

# Detect symmetric sparsity pattern and color it in one step:
colored_pattern = hessian_coloring(g, input_shape=100)
print(colored_pattern)

# Compute sparse Hessian using the precomputed coloring:
for x in inputs:
    H = hessian(g, x, colored_pattern=colored_pattern)
```

## How it works

**Jacobian sparsity detection**: `asdex` uses `jax.make_jaxpr` to trace the function into a jaxpr (JAX's intermediate representation) and propagates **index sets** through each primitive operation.
Each input element starts with its own index `{i}`, and operations combine these sets.
Output index sets reveal which inputs affect each output.
The result is a global sparsity pattern, valid for all input values.

**Hessian sparsity detection**: Since the Hessian is the Jacobian of the gradient, `hessian_sparsity(f, input_shape)` simply calls `jacobian_sparsity(jax.grad(f), input_shape)`.
The sparsity interpreter composes naturally with JAX's autodiff transforms.

**Coloring**: Two rows can be computed together if they don't share any non-zero columns (row coloring, uses VJPs).
Analogously, two columns can be computed together if they don't share any non-zero rows (column coloring, uses JVPs).
`asdex` builds a conflict graph and greedily assigns colors using a LargestFirst ordering.
By default, `color_jacobian_pattern(sparsity_pattern)` runs both row and column coloring
and automatically selects whichever partition needs fewer colors,
choosing the corresponding AD mode (VJPs for row coloring, JVPs for column coloring).

**Sparse Jacobian**: Based on the coloring result, `asdex` automatically uses either reverse mode (VJPs for row coloring) or forward mode (JVPs for column coloring).
For each color, it computes a single VJP or JVP with a seed vector that has 1s at the positions of all same-colored rows or columns.
Due to the coloring constraint, each entry can be uniquely extracted from the compressed results.

**Sparse Hessian**: For each color, `asdex` computes a Hessian-vector product (HVP) using forward-over-reverse AD: `jax.jvp(jax.grad(f), (x,), (v,))`.
This is more efficient than reverse-over-reverse (VJP on gradient) because forward-mode has less overhead for the outer differentiation.

## Related work

- [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl): `asdex` started as a primitive port of this Julia package, which provides global and local Jacobian and Hessian sparsity detection via operator overloading.
- [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl): Julia package for coloring algorithms on sparse matrices.


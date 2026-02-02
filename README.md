# detex

[![CI](https://github.com/adrhill/detex/actions/workflows/ci.yml/badge.svg)](https://github.com/adrhill/detex/actions/workflows/ci.yml)

`detex` detects Jacobian sparsity patterns in JAX.

> [!CAUTION]
> Currently, the primary purpose of this package is to **evaluate the capabilities of coding agents** [on a familiar task I consider to be out-of-distribution](https://github.com/adrhill/SparseConnectivityTracer.jl) from the usual language model training data.
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
print(pattern.toarray().astype(int))
# [[1 0 0]
#  [1 1 0]
#  [0 0 1]]
```

## How it works

`detex` uses `jax.make_jaxpr` to trace the function into a jaxpr — JAX's intermediate representation that captures the computation as a sequence of primitive operations. It then walks this graph, propagating **index sets** through each primitive. Each input element starts with its own index `{i}`, and operations combine these sets (e.g., `z = x * y` means `z`'s indices are the union of `x`'s and `y`'s). Output index sets reveal which inputs affect each output.
The result is a global sparsity pattern, valid for all input values.

## Related work

- [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl): `detex` is a primitive port of this Julia package, which provides global and local Jacobian and Hessian sparsity detection via operator overloading.


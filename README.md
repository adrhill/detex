# detex

`detex` detects Jacobian sparsity patterns in JAX.

> [!WARNING]
> This project is an agentic port of my Julia package [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl).
> Its primary purpose is to **evaluate the capabilities of coding agents** on a familiar task I consider to be out-of-distribution from the usual training data.
> Use `detex` at your own risk. 

___

For a function $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J \in \mathbb{R}^{m \times n}$ is defined as:

$$
J_{ij} = \frac{\partial f_i}{\partial x_j}
$$

Computing the full Jacobian requires $n$ forward-mode AD passes or $m$ reverse-mode passes. But many Jacobians are *sparse*—most entries are structurally zero for all inputs.
`detex` detects this sparsity pattern in a single forward pass by analyzing the computation graph. This enables [automatic sparse differentiation](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/) after graph coloring,
i.e. using [sparsediffax](https://github.com/gdalle/sparsediffax).

## Installation

```bash
pip install detex
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add detex
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

The function

$$f(x) = \begin{bmatrix} x_1^2 \\ 2 x_1 x_2^2 \\ \sin(x_3) \end{bmatrix}$$

has the Jacobian

$$J_f = \begin{bmatrix} 2x_1 & 0 & 0 \\ 2x_2^2 & 4x_1 x_2 & 0 \\ 0 & 0 & \cos(x_3) \end{bmatrix}$$

`detex` detects the corresponding sparsity pattern

$$\begin{bmatrix} 1 & 0 & 0 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

## How it works

Instead of computing actual derivatives, `detex` propagates **index sets** through the computation graph:
- Each input element starts with its own index `{i}`
- Operations combine index sets (e.g., `z = x + y` means `z`'s indices are the union of `x`'s and `y`'s)
- Output index sets reveal which inputs affect each output

This is a global sparsity pattern and valid for all input values.

## See also

- [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl): detex is a very basic port of this package. 

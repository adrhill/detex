# Computing Sparse Hessians

`asdex` computes sparse Hessians for scalar-valued functions
\(f: \mathbb{R}^n \to \mathbb{R}\)
using symmetric coloring and forward-over-reverse AD.

## One-Call API

The simplest way to compute a sparse Hessian:

```python
from asdex import hessian

H = hessian(f, x)
```

This detects sparsity, colors the pattern symmetrically, and decompresses.

!!! warning "Precompute the colored pattern"

    Without a precomputed colored pattern,
    `hessian` re-detects sparsity and re-colors on every call.
    These steps are computationally expensive.
    If you call `hessian` more than once for the same function,
    precompute the colored pattern and reuse it — see below.

## Precomputing the Colored Pattern

When computing Hessians at many different inputs,
precompute the colored pattern once:

```python exec="true" session="hess" source="above"
import jax.numpy as jnp
from asdex import hessian_coloring, hessian

def g(x):
    return jnp.sum(x ** 2)

colored_pattern = hessian_coloring(g, input_shape=100)
```

```python exec="true" session="hess"
print(f"```\n{colored_pattern}\n```")
```

Reuse the colored pattern across evaluations:

```python
for x in inputs:
    H = hessian(g, x, colored_pattern)
```

## Symmetric Coloring

Hessians are symmetric (\(H = H^\top\)),
and `asdex` exploits this with *star coloring*
(Gebremedhin et al., 2005).
Symmetric coloring typically needs fewer colors than row or column coloring,
since both \(H_{ij}\) and \(H_{ji}\) can be recovered from a single coloring.

The convenience functions `hessian_coloring` and `hessian` use symmetric coloring automatically.

## Separate Detection and Coloring

For more control:

```python
from asdex import hessian_sparsity, color_hessian_pattern

sparsity = hessian_sparsity(g, input_shape=100)
colored_pattern = color_hessian_pattern(sparsity)
```

Since the Hessian is the Jacobian of the gradient,
`hessian_sparsity` simply calls `jacobian_sparsity(jax.grad(f), input_shape)`.
The sparsity interpreter composes naturally with JAX's autodiff transforms.

You can also provide a sparsity pattern manually.
Create a `SparsityPattern` from coordinate arrays, a dense matrix, or a JAX BCOO matrix:

```python
import numpy as np
from asdex import SparsityPattern, color_hessian_pattern

# From row and column index arrays:
sparsity = SparsityPattern.from_coordinates(
    rows=[0, 0, 1, 1, 1, 2, 2],
    cols=[0, 1, 0, 1, 2, 1, 2],
    shape=(3, 3),
)

# From a dense boolean or numeric matrix:
dense = np.array([[1, 1, 0],
                  [1, 1, 1],
                  [0, 1, 1]])
sparsity = SparsityPattern.from_dense(dense)

colored_pattern = color_hessian_pattern(sparsity)
```

## Multi-Dimensional Inputs

`asdex` supports multi-dimensional input arrays.
The Hessian is always returned as a 2D matrix
of shape \((n, n)\) where \(n\) is the total number of input elements:

```python exec="true" session="hess-multi" source="above"
import jax.numpy as jnp
from asdex import hessian_coloring

def g(x):
    # x has shape (10, 10)
    return jnp.sum((x[1:, :] - x[:-1, :]) ** 2)

colored_pattern = hessian_coloring(g, input_shape=(10, 10))
```

```python exec="true" session="hess-multi"
print(f"```\n{colored_pattern}\n```")
```

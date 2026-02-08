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
The result is a JAX [BCOO](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) sparse matrix.

!!! warning "Precompute the colored pattern"

    Without a precomputed colored pattern,
    `hessian` re-detects sparsity and re-colors on every call.
    These steps are computationally expensive.
    If you call `hessian` more than once for the same function,
    precompute the colored pattern and reuse it — see below.

## Precomputing the Colored Pattern

When computing Hessians at many different inputs,
precompute the colored pattern once:

```python
from asdex import hessian_coloring, hessian

colored_pattern = hessian_coloring(g, input_shape=100)

for x in inputs:
    H = hessian(g, x, colored_pattern)
```

The colored pattern depends only on the function structure,
not the input values,
so it can be reused across evaluations.

## Symmetric Coloring

Hessians are symmetric (\(H = H^\top\)),
and `asdex` exploits this with *star coloring*
(Gebremedhin et al., 2005).
Symmetric coloring typically needs fewer colors than row or column coloring,
since both \(H_{ij}\) and \(H_{ji}\) can be recovered from a single coloring.

The convenience functions `hessian_coloring` and `hessian` use symmetric coloring automatically:

```python exec="true" session="hess" source="above"
import jax.numpy as jnp
from asdex import hessian_coloring

def g(x):
    return jnp.sum(x ** 2)

colored_pattern = hessian_coloring(g, input_shape=100)
```

```python exec="true" session="hess"
print(f"```\n{colored_pattern}\n```")
```

## Separate Detection and Coloring

For more control, you can split detection and coloring:

```python
from asdex import hessian_sparsity, color_hessian_pattern

sparsity = hessian_sparsity(g, input_shape=100)
colored_pattern = color_hessian_pattern(sparsity)
```

This is useful when you want to inspect the sparsity pattern (`print(sparsity)`)
before deciding on a coloring strategy.

Since the Hessian is the Jacobian of the gradient,
`hessian_sparsity` simply calls `jacobian_sparsity(jax.grad(f), input_shape)`.
The sparsity interpreter composes naturally with JAX's autodiff transforms.

## Manually Providing a Sparsity Pattern

You can provide a sparsity pattern manually if you already know it ahead of time.
Create a `SparsityPattern` from coordinate arrays, a dense matrix, or a JAX BCOO matrix.

From a dense boolean or numeric matrix:

```python exec="true" session="hess" source="above"
import numpy as np
from asdex import SparsityPattern

dense = np.array([[1, 1, 0, 0],
                  [1, 1, 1, 0],
                  [0, 1, 1, 1],
                  [0, 0, 1, 1]])
sparsity = SparsityPattern.from_dense(dense)
```
```python exec="true" session="hess"
print(f"```\n{sparsity}\n```")
```

From row and column index arrays:

```python exec="true" session="hess" source="above"
sparsity = SparsityPattern.from_coordinates(
    rows=[0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
    cols=[0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
    shape=(4, 4),
)
```
```python exec="true" session="hess"
print(f"```\n{sparsity}\n```")
```

From a JAX BCOO sparse matrix:
```python
sparsity = SparsityPattern.from_bcoo(bcoo_matrix)
```

Finally, color the sparsity pattern and compute the Hessian:
```python
from asdex import color_hessian_pattern, hessian

colored_pattern = color_hessian_pattern(sparsity)
H = hessian(f, x, colored_pattern)
```

## Multi-Dimensional Inputs

`asdex` supports multi-dimensional input arrays.
The Hessian is always returned as a 2D matrix
of shape \((n, n)\) where \(n\) is the total number of input elements:

```python exec="true" session="hess-multi" source="above"
import jax.numpy as jnp
from asdex import hessian_coloring

def g(x):
    # x has shape (5, 20)
    return jnp.sum(x ** 3)

colored_pattern = hessian_coloring(g, input_shape=(5, 20))
```

```python exec="true" session="hess-multi"
print(f"```\n{colored_pattern}\n```")
```

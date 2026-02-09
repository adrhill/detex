# Computing Sparse Jacobians

## One-Call API

The simplest way to compute a sparse Jacobian:

```python
from asdex import jacobian

J = jacobian(f)(x)
```

This detects sparsity, colors the pattern, and decompresses — all in one call.
The result is a JAX [BCOO](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) sparse matrix.

!!! warning "Precompute the colored pattern"

    Without a precomputed colored pattern,
    `jacobian` re-detects sparsity and re-colors on every call.
    These steps are computationally expensive.
    If you call `jacobian` more than once for the same function,
    precompute the colored pattern and reuse it — see below.

## Precomputing the Colored Pattern

When computing Jacobians at many different inputs,
precompute the colored pattern once:

```python
from asdex import jacobian_coloring, jacobian

colored_pattern = jacobian_coloring(f, input_shape=1000)

jac_f = jacobian(f, colored_pattern)
for x in inputs:
    J = jac_f(x)
```

The colored pattern depends only on the function structure,
not the input values,
so it can be reused across evaluations.

## Choosing Row vs Column Coloring

By default, `asdex` tries both row and column coloring
and picks whichever needs fewer colors:

```python exec="true" session="jac" source="above"
from asdex import jacobian_coloring

def f(x):
    return (x[1:] - x[:-1]) ** 2

# Automatic selection (default):
colored_pattern = jacobian_coloring(f, input_shape=100)
```

```python exec="true" session="jac"
print(f"```\n{colored_pattern}\n```")
```

You can also force a specific coloring strategy.
Row coloring uses VJPs (reverse-mode AD),
column coloring uses JVPs (forward-mode AD):

```python exec="true" session="jac" source="above"
# Force column coloring (uses JVPs):
colored_pattern = jacobian_coloring(f, input_shape=100, partition="column")

# Force row coloring (uses VJPs):
colored_pattern = jacobian_coloring(f, input_shape=100, partition="row")
```

```python exec="true" session="jac"
print(f"```\n{colored_pattern}\n```")
```

When the number of colors is equal,
`asdex` prefers column coloring since JVPs are generally cheaper in JAX.

## Separate Detection and Coloring

For more control, you can split detection and coloring:

```python
from asdex import jacobian_sparsity, color_jacobian_pattern

sparsity = jacobian_sparsity(f, input_shape=1000)
colored_pattern = color_jacobian_pattern(sparsity, partition="column")
```

This is useful when you want to inspect the sparsity pattern (`print(sparsity)`)
before deciding on a coloring strategy.

## Manually Providing a Sparsity Pattern

You can provide a sparsity pattern manually if you already know it ahead of time.
Create a `SparsityPattern` from coordinate arrays, a dense matrix, or a JAX BCOO matrix.

From a dense boolean or numeric matrix:

```python exec="true" session="jac" source="above"
import numpy as np
from asdex import SparsityPattern

dense = np.array([[1, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 1]])
sparsity = SparsityPattern.from_dense(dense)
```
```python exec="true" session="jac"
print(f"```\n{sparsity}\n```")
```

From row and column index arrays:

```python exec="true" session="jac" source="above"
sparsity = SparsityPattern.from_coordinates(
    rows=[0, 0, 1, 1, 2, 2],
    cols=[0, 1, 1, 2, 2, 3],
    shape=(3, 4),
)
```
```python exec="true" session="jac"
print(f"```\n{sparsity}\n```")
```

From a JAX BCOO sparse matrix:
```python
sparsity = SparsityPattern.from_bcoo(bcoo_matrix)
```

Finally, color the sparsity pattern and compute the Jacobian:
```python
from asdex import color_jacobian_pattern, jacobian

colored_pattern = color_jacobian_pattern(sparsity)
J = jacobian(f, colored_pattern)(x)
```

## Multi-Dimensional Inputs

`asdex` supports multi-dimensional input and output arrays.
The Jacobian is always returned as a 2D matrix
of shape \((m, n)\) where \(n\) is the total number of input elements
and \(m\) is the total number of output elements:

```python exec="true" session="jac-multi" source="above"
from asdex import jacobian_coloring

def f(x):
    # x has shape (10, 10), output has shape (9, 10)
    return x[1:, :] - x[:-1, :]

colored_pattern = jacobian_coloring(f, input_shape=(10, 10))
```

```python exec="true" session="jac-multi"
print(f"```\n{colored_pattern}\n```")
```

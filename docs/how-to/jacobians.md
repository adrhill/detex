# Computing Sparse Jacobians

## One-Call API

The simplest way to compute a sparse Jacobian:

```python
from asdex import jacobian

J = jacobian(f, x)
```

This detects sparsity, colors the pattern, and decompresses — all in one call.
The result is a JAX [BCOO](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) sparse matrix.

## Precomputing the Colored Pattern

When computing Jacobians at many different inputs,
precompute the colored pattern once:

```python
from asdex import jacobian_coloring, jacobian

colored_pattern = jacobian_coloring(f, input_shape=1000)

for x in inputs:
    J = jacobian(f, x, colored_pattern)
```

The colored pattern depends only on the function structure,
not the input values,
so it can be reused across evaluations.

## Choosing Row vs Column Coloring

By default, `asdex` tries both row and column coloring
and picks whichever needs fewer colors:

```python
from asdex import jacobian_coloring

# Automatic selection (default):
colored_pattern = jacobian_coloring(f, input_shape=100)

# Force row coloring (uses VJPs):
colored_pattern = jacobian_coloring(f, input_shape=100, partition="row")

# Force column coloring (uses JVPs):
colored_pattern = jacobian_coloring(f, input_shape=100, partition="column")
```

Row coloring uses VJPs (reverse-mode AD),
column coloring uses JVPs (forward-mode AD).
When the number of colors is equal,
`asdex` prefers column coloring since JVPs are generally cheaper in JAX.

## Separate Detection and Coloring

For more control, you can split detection and coloring:

```python
from asdex import jacobian_sparsity, color_jacobian_pattern

sparsity = jacobian_sparsity(f, input_shape=1000)
colored_pattern = color_jacobian_pattern(sparsity, partition="column")
```

This is useful when you want to inspect the sparsity pattern
before deciding on a coloring strategy.

## Multi-Dimensional Inputs

`asdex` supports multi-dimensional input and output arrays.
The Jacobian is always returned as a 2D matrix
of shape \((m, n)\) where \(n\) is the total number of input elements
and \(m\) is the total number of output elements:

```python
def f(x):
    # x has shape (10, 10), output has shape (9, 10)
    return x[1:, :] - x[:-1, :]

colored_pattern = jacobian_coloring(f, input_shape=(10, 10))
```

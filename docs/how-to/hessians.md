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

## Precomputing the Colored Pattern

When computing Hessians at many different inputs,
precompute the colored pattern once:

```python
from asdex import hessian_coloring, hessian

colored_pattern = hessian_coloring(g, input_shape=100)

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

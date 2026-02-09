# Getting Started

This tutorial walks through the three stages of automatic sparse differentiation:
**detection**, **coloring**, and **decompression**.

## The Problem

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

## Sparse Jacobians

Consider the squared differences function
\(f(x)_i = (x_{i+1} - x_i)^2\),
which has a banded Jacobian.
Detect the sparsity pattern and color it in one step:

```python exec="true" session="gs" source="above"
from asdex import jacobian_coloring

def f(x):
    return (x[1:] - x[:-1]) ** 2

colored_pattern = jacobian_coloring(f, input_shape=50)
```

```python exec="true" session="gs"
print(f"```\n{colored_pattern}\n```")
```

The print-out shows the original sparsity pattern (left) compressed into just two colors (right).
`asdex` automatically ran multiple coloring algorithms,
selected column coloring (2 JVPs) over row coloring (2 VJPs) since JVPs are cheaper,
reducing the cost from 49 VJPs or 50 JVPs without coloring to just 2 JVPs.
Note that on small problems, this doesn't directly translate into a speedup of factor 25x,
as the decompression overhead dominates.

!!! info "Global Sparsity Patterns"

    The detected pattern is a **global** sparsity pattern:
    it depends only on the function's structure, not on any particular input.
    This means it may contain extra nonzeros compared to the sparsity at a specific point,
    but it is guaranteed to be correct everywhere.
    If you encounter overly conservative patterns,
    please [open an issue](https://github.com/adrhill/asdex/issues).
    See [Sparsity Detection](../explanation/sparsity-detection.md) for details.

Now we can compute the sparse Jacobian using the colored pattern:

```python exec="true" session="gs" source="above"
import jax.numpy as jnp
from asdex import jacobian

x = jnp.ones(50)
J = jacobian(f, colored_pattern)(x)
```

The result is a JAX
[BCOO](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) sparse matrix.
We can verify that `asdex` produces the same result as `jax.jacobian`:

```python exec="true" session="gs" source="above"
import jax
import numpy as np

J_dense = jax.jacobian(f)(x)
np.testing.assert_allclose(J.todense(), J_dense, atol=1e-6)
```

On larger problems, the speedup from coloring becomes significant.
Let's benchmark on a 5000-dimensional input
(note that timings may vary as part of the doc-building process):

```python exec="true" session="gs" source="above"
import timeit

n = 5000
colored_pattern = jacobian_coloring(f, input_shape=n)
x = jnp.ones(n)

# Warm up JIT caches
jac_f = jacobian(f, colored_pattern)
_ = jac_f(x)
_ = jax.jacobian(f)(x)

t_asdex = timeit.timeit(lambda: jac_f(x).block_until_ready(), number=10) / 10
t_jax = timeit.timeit(lambda: jax.jacobian(f)(x).block_until_ready(), number=10) / 10
```

```python exec="true" session="gs"
lines = [
    f"asdex.jacobian:  {t_asdex*1000:8.2f} ms",
    f"jax.jacobian:    {t_jax*1000:8.2f} ms",
    f"speedup:         {t_jax/t_asdex:8.1f}x",
]
print("```\n" + "\n".join(lines) + "\n```")
```

!!! tip "Precompute for Repeated Evaluations"

    The colored pattern depends only on the function structure,
    not the input values.
    When computing Jacobians at many different inputs,
    precompute the colored pattern once and reuse it:

    ```python
    colored_pattern = jacobian_coloring(f, input_shape=5000)

    for x in inputs:
        J = jacobian(f, colored_pattern)(x)
    ```

## Sparse Hessians

For scalar-valued functions \(f: \mathbb{R}^n \to \mathbb{R}\),
`asdex` can detect Hessian sparsity and compute sparse Hessians:

```python
from asdex import hessian_coloring, hessian

def g(x):
    return jnp.sum(x ** 2)

hess_g = hessian(g, hessian_coloring(g, input_shape=20))

for x in inputs:
    H = hess_g(x)
```

## Next Steps

- [Computing Sparse Jacobians](../how-to/jacobians.md) — Guide on Jacobian computation
- [Computing Sparse Hessians](../how-to/hessians.md) — Guide on Hessian computation
- [Sparsity Detection](../explanation/sparsity-detection.md) — Explanation how sparsity patterns are detected
- [Graph Coloring](../explanation/coloring.md) — Explanation how coloring reduces cost

# Getting Started

This tutorial walks through the three stages of automatic sparse differentiation:
**detection**, **coloring**, and **decompression**.

## The Problem

For a function \(f: \mathbb{R}^n \to \mathbb{R}^m\),
computing the full Jacobian \(J \in \mathbb{R}^{m \times n}\)
requires \(n\) forward-mode or \(m\) reverse-mode AD passes.
In practice, many Jacobians are *sparse*
— most entries are structurally zero, regardless of the input.

`asdex` exploits this sparsity in three steps:

1. **Detect** the sparsity pattern by tracing the computation graph
2. **Color** the pattern so that structurally orthogonal rows (or columns) share a color
3. **Decompress** one AD pass per color into the sparse Jacobian

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

The print-out shows the original sparsity pattern (left)
compressed into just two colors (right).
`asdex` automatically tried both row and column coloring
and selected column coloring (2 JVPs) since JVPs are cheaper,
reducing the cost from 49 VJPs or 50 JVPs to just 2 JVPs.

Now compute the sparse Jacobian using the colored pattern:

```python exec="true" session="gs" source="above"
import jax.numpy as jnp
from asdex import jacobian

x = jnp.ones(50)
J = jacobian(f, x, colored_pattern)
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
_ = jacobian(f, x, colored_pattern)
_ = jax.jacobian(f)(x)

t_asdex = timeit.timeit(lambda: jacobian(f, x, colored_pattern).block_until_ready(), number=10) / 10
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
        J = jacobian(f, x, colored_pattern)
    ```

## Sparse Hessians

For scalar-valued functions \(f: \mathbb{R}^n \to \mathbb{R}\),
`asdex` can detect Hessian sparsity and compute sparse Hessians:

```python exec="true" session="gs" source="above"
from asdex import hessian_coloring, hessian

def g(x):
    return jnp.sum(x ** 2)

colored_pattern = hessian_coloring(g, input_shape=20)
```

```python exec="true" session="gs"
print(f"```\n{colored_pattern}\n```")
```

Compute the sparse Hessian using the colored pattern:

```python exec="true" session="gs" source="above"
x = jnp.ones(20)
H = hessian(g, x, colored_pattern)
```

## Next Steps

- [Computing Sparse Jacobians](../how-to/jacobians.md) — more Jacobian recipes
- [Computing Sparse Hessians](../how-to/hessians.md) — Hessian computation guide
- [The ASD Pipeline](../explanation/pipeline.md) — deeper explanation of the architecture

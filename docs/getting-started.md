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

## Setup

Let's use the squared differences function
\(f(x)_i = (x_{i+1} - x_i)^2\),
which has a banded Jacobian:

```python
import jax.numpy as jnp
import numpy as np
from asdex import jacobian_sparsity, color_jacobian_pattern, jacobian
```

## Step 1: Detect Sparsity

Detect the sparsity pattern by analyzing the computation graph.
No derivatives are evaluated — the result is valid for all inputs:

```python
def f(x):
    return (x[1:] - x[:-1]) ** 2

sparsity = jacobian_sparsity(f, input_shape=50)
print(sparsity)
# SparsityPattern(49×50, nnz=98, sparsity=96.0%)
```

The sparsity pattern tells us which entries of the Jacobian may be nonzero.
Out of \(49 \times 50 = 2450\) entries, only 98 are structurally nonzero.

## Step 2: Color the Pattern

Graph coloring assigns colors to rows (or columns)
such that same-colored rows don't share any nonzero columns.
This allows computing multiple rows in a single AD pass:

```python
colored_pattern = color_jacobian_pattern(sparsity)
print(colored_pattern)
# ColoredPattern(49×50, nnz=98, sparsity=96.0%, JVP, 2 colors)
```

By default, `asdex` tries both row coloring (VJPs) and column coloring (JVPs),
then picks whichever needs fewer colors.
Here it found that 2 JVPs suffice — down from 49 VJPs or 50 JVPs.

## Step 3: Decompress

Compute the sparse Jacobian using the colored pattern.
Each color corresponds to one AD pass:

```python
x = jnp.ones(50)
J = jacobian(f, x, colored_pattern)
print(J.shape)   # (49, 50)
print(J.nnz)     # 98
```

The result is a JAX
[BCOO](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) sparse matrix.

## The One-Call API

For convenience, you can do all three steps in one call:

```python
from asdex import jacobian_coloring, jacobian

colored_pattern = jacobian_coloring(f, input_shape=50)
J = jacobian(f, x, colored_pattern)
```

Or let `jacobian` handle everything automatically:

```python
J = jacobian(f, x)
```

!!! tip "Precompute for Repeated Evaluations"

    The sparsity pattern depends only on the function structure,
    not the input values.
    When computing Jacobians at many different inputs,
    precompute the colored pattern once and reuse it:

    ```python
    colored_pattern = jacobian_coloring(f, input_shape=1000)

    for x in inputs:
        J = jacobian(f, x, colored_pattern)
    ```

## Next Steps

- [Computing Sparse Jacobians](how-to/jacobians.md) — more Jacobian recipes
- [Computing Sparse Hessians](how-to/hessians.md) — Hessian computation guide
- [The 3-Stage Pipeline](explanation/pipeline.md) — deeper explanation of the architecture

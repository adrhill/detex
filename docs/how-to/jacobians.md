# Computing Sparse Jacobians

`asdex` computes sparse Jacobians for functions
\(f: \mathbb{R}^n \to \mathbb{R}^m\)
using [row or column coloring](../explanation/coloring.md) with forward- or reverse-mode AD.

!!! tip "Verify correctness at least once"

    asdex's [sparsity patterns](../explanation/global-sparsity.md) should always be conservative,
    but a bug in [sparsity detection](../explanation/sparsity-detection.md) could cause missing nonzeros.
    Always verify against vanilla JAX at least once on a new function.
    See [Verifying Results](#verifying-results) below.

## Basic Usage

The simplest way to compute a sparse Jacobian is to pass `input_shape`:

```python
from asdex import jacobian

jac_fn = jacobian(f, input_shape=1000)
J = jac_fn(x)
```

This detects sparsity and colors the pattern once at definition time,
then each call to `jac_fn` only performs the cheap decompression step.
The result is a JAX [BCOO](https://docs.jax.dev/en/latest/jax.experimental.sparse.html) sparse matrix.

The same function can be reused across evaluations at different inputs:

```python
for x in inputs:
    J = jac_fn(x)
```

## Precomputing the Colored Pattern

For more control,
precompute the colored pattern explicitly and pass it to `jacobian`:

```python
from asdex import jacobian_coloring, jacobian

coloring = jacobian_coloring(f, input_shape=1000)
jac_fn = jacobian(f, coloring)

for x in inputs:
    J = jac_fn(x)
```

This is useful when you want to inspect the colored pattern,
save it to disk,
or use a specific coloring mode.

!!! tip

    If your `coloring` looks wrong or overly dense,
    please help out `asdex`'s development by
    [reporting it](https://github.com/adrhill/asdex/issues).
    These reports directly drive improvements
    and are one of the most impactful ways to contribute.

## Saving and Loading Patterns

Save a colored pattern to disk and reload it in a later session:

```python
coloring = jacobian_coloring(f, input_shape=1000)
coloring.save("colored.npz")
```

```python
from asdex import ColoredPattern

coloring = ColoredPattern.load("colored.npz")
jac_fn = jacobian(f, coloring)
```

[`SparsityPattern`](../reference/index.md#asdex.SparsityPattern) supports the same `save`/`load` interface.

## Choosing Row vs Column Coloring

By default, `asdex` tries both row and column coloring
and picks whichever needs fewer colors:

```python exec="true" session="jac" source="above"
from asdex import jacobian_coloring

def f(x):
    return (x[1:] - x[:-1]) ** 2

# Automatic selection (default):
coloring = jacobian_coloring(f, input_shape=100)
```

```python exec="true" session="jac"
print(f"```\n{coloring}\n```")
```

You can also force a specific AD mode.
``"fwd"`` colors columns (uses JVPs, forward-mode AD),
``"rev"`` colors rows (uses VJPs, reverse-mode AD):

```python exec="true" session="jac" source="above"
# Force forward mode (column coloring, uses JVPs):
coloring = jacobian_coloring(f, input_shape=100, mode="fwd")

# Force reverse mode (row coloring, uses VJPs):
coloring = jacobian_coloring(f, input_shape=100, mode="rev")
```

```python exec="true" session="jac"
print(f"```\n{coloring}\n```")
```

When the number of colors is equal,
`asdex` prefers column coloring since JVPs are generally cheaper in JAX.

## Separate Detection and Coloring

For more control, you can split detection and coloring:

```python
from asdex import jacobian_sparsity, color_jacobian_pattern

sparsity = jacobian_sparsity(f, input_shape=1000)
coloring = color_jacobian_pattern(sparsity, mode="fwd")
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
sparsity = SparsityPattern.from_coo(
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

coloring = color_jacobian_pattern(sparsity)
jac_fn = jacobian(f, coloring)
J = jac_fn(x)
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

coloring = jacobian_coloring(f, input_shape=(10, 10))
```

```python exec="true" session="jac-multi"
print(f"```\n{coloring}\n```")
```

## Verifying Results

Use [`check_jacobian_correctness`][asdex.check_jacobian_correctness]
to verify the sparse Jacobian against vanilla JAX.

```python
from asdex import check_jacobian_correctness

check_jacobian_correctness(f, x)
```

Use verification for debugging and initial setup, not in production loops.
A good place to call it is in your test suite.

By default, this uses randomized matrix-vector products (`method="matvec"`)
to check `asdex.jacobian(f, input_shape=...)(x)` against JVPs or VJPs,
automatically picking forward or reverse mode based on the input and output sizes.
This is cheap — O(k) in the number of probes — and scales to large problems.
If the results match, the function returns silently.
If they disagree, it raises a [`VerificationError`][asdex.VerificationError].

You can also pass a pre-computed colored pattern, control the AD mode used for the reference computation, 
set custom tolerances, the number of probes, and the PRNG seed:

```python
check_jacobian_correctness(f, x, coloring=coloring, ad_mode="rev", rtol=1e-5, atol=1e-5, num_probes=50, seed=42)
```

For an exact element-wise comparison against the full dense Jacobian,
use `method="dense"`:

```python
check_jacobian_correctness(f, x, method="dense")
```

!!! warning "Dense computation"

    `method="dense"` materializes the full dense Jacobian,
    which is computationally very expensive for large problems.

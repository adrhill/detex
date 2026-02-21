# Computing Sparse Hessians

`asdex` computes sparse Hessians for scalar-valued functions
\(f: \mathbb{R}^n \to \mathbb{R}\)
using symmetric coloring and forward-over-reverse AD.

!!! tip "Verify correctness at least once"

    asdex's [sparsity patterns](../explanation/global-sparsity.md) should always be conservative,
    but a bug in [sparsity detection](../explanation/sparsity-detection.md) could cause missing nonzeros.
    Always verify against vanilla JAX at least once on a new function.
    See [Verifying Results](#verifying-results) below.

## One-Call API

The simplest way to compute a sparse Hessian:

```python
from asdex import hessian

H = hessian(f)(x)
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
hess_fn = hessian(g, colored_pattern)

for x in inputs:
    H = hess_fn(x)
```

The colored pattern depends only on the function structure,
not the input values,
so it can be reused across evaluations.

!!! tip

    If your `colored_pattern` looks wrong or overly dense,
    please help out `asdex`'s development by
    [reporting it](https://github.com/adrhill/asdex/issues).
    These reports directly drive improvements
    and are one of the most impactful ways to contribute.

## Saving and Loading Patterns

Save a colored pattern to disk and reload it in a later session:

```python
colored_pattern = hessian_coloring(g, input_shape=100)
colored_pattern.save("colored.npz")
```

```python
from asdex import ColoredPattern

colored_pattern = ColoredPattern.load("colored.npz")
hess_fn = hessian(g, colored_pattern)
```

[`SparsityPattern`](../reference/index.md#asdex.SparsityPattern) supports the same `save`/`load` interface.

## Symmetric Coloring

Hessians are symmetric (\(H = H^\top\)),
and `asdex` exploits this with *star coloring*
(Gebremedhin et al., 2005).
Symmetric coloring typically needs fewer colors than row or column coloring,
since both \(H_{ij}\) and \(H_{ji}\) can be recovered from a single coloring.

The convenience functions `hessian_coloring` and `hessian` use symmetric coloring automatically.
Here we use the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function),
a classic optimization benchmark whose Hessian is tridiagonal:

\[f(x) = \sum_{i=1}^{n-1} \left[(1 - x_i)^2 + 100\,(x_{i+1} - x_i^2)^2\right]\]

```python exec="true" session="hess" source="above"
import jax.numpy as jnp
from asdex import hessian_coloring

def rosenbrock(x):
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

colored_pattern = hessian_coloring(rosenbrock, input_shape=100)
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
The [sparsity interpreter](../explanation/sparsity-detection.md) composes naturally with JAX's autodiff transforms.

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
hess_fn = hessian(f, colored_pattern)
H = hess_fn(x)
```

## Choosing an HVP Mode

By default, `hessian` uses forward-over-reverse AD to compute Hessian-vector products.
You can select a different AD composition strategy via the `mode` parameter:

```python
from asdex import hessian

h_for = hessian(f, mode="fwd_over_rev")(x)  # default
h_rof = hessian(f, mode="rev_over_fwd")(x)
h_ror = hessian(f, mode="rev_over_rev")(x)
```

All three modes produce the same mathematical result.
They differ in how JAX's AD primitives are composed:

- **`fwd_over_rev`** (default): `jvp(grad(f), ...)`.
    Generally the fastest under JIT.
- **`rev_over_fwd`**: `grad(lambda p: jvp(f, (p,), (v,))[1])`.
    Can use less memory than forward-over-reverse for functions with many intermediates.
- **`rev_over_rev`**: `grad(lambda y: dot(grad(f)(y), v))`.
    Avoids forward-mode entirely;
    useful when forward-mode is expensive or unsupported.

!!! tip

    When in doubt, stick with the default `"fwd_over_rev"`.
    It is the most widely used and typically the most efficient under `jax.jit`.

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

## Verifying Results

Use [`check_hessian_correctness`][asdex.check_hessian_correctness]
to verify the sparse Hessian against vanilla JAX.

```python
from asdex import check_hessian_correctness

check_hessian_correctness(g, x)
```

Use verification for debugging and initial setup, not in production loops.
A good place to call it is in your test suite.

By default, this uses randomized matrix-vector products (`method="matvec"`)
to check `asdex.hessian(g)(x)` against a forward-over-reverse HVP reference.
This is cheap — O(k) in the number of probes — and scales to large problems.
If the results match, the function returns silently.
If they disagree, it raises a [`VerificationError`][asdex.VerificationError].

You can also pass a pre-computed colored pattern, control the AD mode used for the reference computation,
set custom tolerances, the number of probes, and the PRNG seed:

```python
check_hessian_correctness(g, x, colored_pattern=colored_pattern, mode="rev_over_rev", rtol=1e-5, atol=1e-5, num_probes=50, seed=42)
```

The `mode` parameter accepts the same values as the
[`mode` parameter](#choosing-an-hvp-mode) on `hessian`:
`"fwd_over_rev"`, `"rev_over_fwd"`, and `"rev_over_rev"`.

For an exact element-wise comparison against the full dense Hessian,
use `method="dense"`:

```python
check_hessian_correctness(g, x, method="dense")
```

!!! warning "Dense computation"

    `method="dense"` materializes the full dense Hessian,
    which is computationally very expensive for large problems.

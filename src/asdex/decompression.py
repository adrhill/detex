"""Sparse Jacobian and Hessian computation using coloring and AD.

Row coloring + VJPs: same-colored rows don't share non-zero columns,
so they can be computed together in a single VJP.
Column coloring + JVPs: same-colored columns don't share non-zero rows,
so they can be computed together in a single JVP.
Star coloring + HVPs: exploits Hessian symmetry for fewer colors.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from numpy.typing import ArrayLike, NDArray

from asdex.coloring import color_hessian_pattern, color_jacobian_pattern
from asdex.detection import hessian_sparsity as _detect_hessian_sparsity
from asdex.detection import jacobian_sparsity as _detect_sparsity
from asdex.pattern import ColoredPattern, SparsityPattern


def _decompress(colored_pattern: ColoredPattern, compressed: jax.Array) -> BCOO:
    """Extract sparse entries from compressed gradient rows.

    Uses pre-computed extraction indices on the ``ColoredPattern``
    to vectorize the decompression step
    (no Python loop over nnz entries).

    Args:
        colored_pattern: Colored sparsity pattern with cached indices.
        compressed: JAX array of shape (num_colors, vector_len),
            one row per color.

    Returns:
        Sparse matrix as BCOO in sparsity-pattern order.
    """
    color_idx, elem_idx = colored_pattern._extraction_indices
    data = compressed[jnp.asarray(color_idx), jnp.asarray(elem_idx)]
    return colored_pattern.sparsity.to_bcoo(data=data)


def _decompress_jacobian(
    sparsity: SparsityPattern,
    colors: NDArray[np.int32],
    compressed: jax.Array,
) -> BCOO:
    """Extract Jacobian entries from VJP results (row coloring).

    For each non-zero (i, j) in the sparsity pattern, the value is extracted
    from the gradient corresponding to row i's color. Due to the orthogonality
    property (same-colored rows don't share columns), each gradient entry
    uniquely corresponds to one Jacobian row.

    Kept for the backward-compat ``hessian(f, x, sparsity=..., colors=...)`` path
    which passes raw ``colors`` arrays instead of a ``ColoredPattern``.

    Args:
        sparsity: Sparsity pattern
        colors: Color assignment for each row
        compressed: JAX array of shape (num_colors, vector_len),
            one row per color.

    Returns:
        Sparse Jacobian as BCOO matrix
    """
    rows = sparsity.rows
    cols = sparsity.cols

    color_idx = colors[rows].astype(np.intp)
    elem_idx = cols.astype(np.intp)
    data = compressed[jnp.asarray(color_idx), jnp.asarray(elem_idx)]

    return sparsity.to_bcoo(data=data)


def jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    colored_pattern: ColoredPattern | None = None,
) -> BCOO:
    """Compute sparse Jacobian using coloring and AD.

    Uses row coloring + VJPs or column coloring + JVPs,
    depending on which needs fewer colors.

    Args:
        f: Function taking an array and returning an array.
            Both may be multi-dimensional.
        x: Input point (any shape).
        colored_pattern: Optional pre-computed :class:`ColoredPattern`
            from :func:`color`.
            If None, sparsity is detected and colored automatically.

    Returns:
        Sparse Jacobian matrix of shape (m, n) as BCOO,
        where n = x.size and m = prod(output_shape)
    """
    x = jnp.asarray(x)
    n = x.size

    if colored_pattern is None:
        sparsity = _detect_sparsity(f, x.shape)
        colored_pattern = color_jacobian_pattern(sparsity)

    sparsity = colored_pattern.sparsity
    m = sparsity.m
    out_shape = jax.eval_shape(f, jnp.zeros_like(x)).shape

    # Handle edge case: no outputs
    if m == 0:
        return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(0, n))

    # Handle edge case: all-zero Jacobian
    if sparsity.nse == 0:
        return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(m, n))

    if colored_pattern.mode == "VJP":
        return _jacobian_rows(f, x, colored_pattern, out_shape)
    else:
        return _jacobian_cols(f, x, colored_pattern)


def _jacobian_rows(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    colored_pattern: ColoredPattern,
    out_shape: tuple[int, ...],
) -> BCOO:
    """Compute sparse Jacobian via row coloring + VJPs."""
    seeds = jnp.asarray(colored_pattern._seed_matrix, dtype=x.dtype)
    _, vjp_fn = jax.vjp(f, x)

    def single_vjp(seed: jax.Array) -> jax.Array:
        (grad,) = vjp_fn(seed.reshape(out_shape))
        return grad.ravel()

    return _decompress(colored_pattern, jax.vmap(single_vjp)(seeds))


def _jacobian_cols(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    colored_pattern: ColoredPattern,
) -> BCOO:
    """Compute sparse Jacobian via column coloring + JVPs."""
    seeds = jnp.asarray(colored_pattern._seed_matrix, dtype=x.dtype)

    def single_jvp(seed: jax.Array) -> jax.Array:
        _, jvp_out = jax.jvp(f, (x,), (seed.reshape(x.shape),))
        return jvp_out.ravel()

    return _decompress(colored_pattern, jax.vmap(single_jvp)(seeds))


def hessian(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    colored_pattern: ColoredPattern | None = None,
    sparsity: SparsityPattern | None = None,
    colors: NDArray[np.int32] | None = None,
) -> BCOO:
    """Compute sparse Hessian using coloring and HVPs.

    When ``colored_pattern`` is provided,
    extracts sparsity and colors from it
    and uses star decompression.
    When ``colors`` is provided (backward compatibility with pre-computed
    ``color_rows`` colors), uses the original row-coloring decompression.
    When both are None, uses star coloring for fewer colors
    and symmetric decompression.

    Uses forward-over-reverse Hessian-vector products for efficiency.

    Args:
        f: Scalar-valued function returning a scalar.
            Input may be multi-dimensional.
        x: Input point (any shape).
        colored_pattern: Optional pre-computed :class:`ColoredPattern`
            from :func:`hessian_coloring`.
            If provided, ``sparsity`` and ``colors`` are ignored.
        sparsity: Optional pre-computed Hessian sparsity pattern from
            hessian_sparsity(). If None, detected automatically.
        colors: Optional pre-computed row coloring from color_rows().
            If None, star coloring is computed and used automatically.

    Returns:
        Sparse Hessian matrix of shape (n, n) as BCOO
    """

    x = jnp.asarray(x)
    n = x.size

    if colored_pattern is not None:
        sparsity = colored_pattern.sparsity
        if sparsity.nse == 0:
            return BCOO(
                (jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(n, n)
            )
        grads = _compute_hvps(f, x, colored_pattern)
        return _decompress(colored_pattern, grads)

    if sparsity is None:
        sparsity = _detect_hessian_sparsity(f, x.shape)

    # Handle edge case: all-zero Hessian
    if sparsity.nse == 0:
        return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(n, n))

    if colors is not None:
        # Backward compat: pre-computed row coloring -> row-based decompression
        num_colors = int(colors.max()) + 1 if len(colors) > 0 else 0
        grads = _compute_hvps_legacy(f, x, colors, num_colors)
        return _decompress_jacobian(sparsity, colors, grads)

    # Default: star coloring + symmetric decompression
    cp = color_hessian_pattern(sparsity)
    grads = _compute_hvps(f, x, cp)
    return _decompress(cp, grads)


def _compute_hvps(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    colored_pattern: ColoredPattern,
) -> jax.Array:
    """Compute one HVP per color using pre-computed seed matrix."""
    seeds = jnp.asarray(colored_pattern._seed_matrix, dtype=x.dtype)
    grad_f = jax.grad(f)

    def single_hvp(seed: jax.Array) -> jax.Array:
        _, hvp = jax.jvp(grad_f, (x,), (seed.reshape(x.shape),))
        return hvp.ravel()

    return jax.vmap(single_hvp)(seeds)


def _compute_hvps_legacy(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    colors: NDArray[np.int32],
    num_colors: int,
) -> jax.Array:
    """Compute one HVP per color (backward-compat path with raw colors)."""
    seed_matrix = np.stack([colors == c for c in range(num_colors)])
    seeds = jnp.asarray(seed_matrix, dtype=x.dtype)
    grad_f = jax.grad(f)

    def single_hvp(seed: jax.Array) -> jax.Array:
        _, hvp = jax.jvp(grad_f, (x,), (seed.reshape(x.shape),))
        return hvp.ravel()

    return jax.vmap(single_hvp)(seeds)

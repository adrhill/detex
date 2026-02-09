"""Sparse Jacobian and Hessian computation using coloring and AD."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from numpy.typing import ArrayLike

from asdex.coloring import color_hessian_pattern, color_jacobian_pattern
from asdex.detection import hessian_sparsity as _detect_hessian_sparsity
from asdex.detection import jacobian_sparsity as _detect_sparsity
from asdex.pattern import ColoredPattern

# =========================================================================
# Public API
# =========================================================================


def jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    colored_pattern: ColoredPattern | None = None,
) -> Callable[[ArrayLike], BCOO]:
    """Build a sparse Jacobian function using coloring and AD.

    Uses row coloring + VJPs or column coloring + JVPs,
    depending on which needs fewer colors.

    Args:
        f: Function taking an array and returning an array.
            Input and output may be multi-dimensional.
        colored_pattern: Optional pre-computed [`ColoredPattern`][asdex.ColoredPattern]
            from [`jacobian_coloring`][asdex.jacobian_coloring].
            If None, sparsity is detected and colored automatically on each call.

    Returns:
        A function that takes an input array and returns
        the sparse Jacobian as BCOO of shape ``(m, n)``
        where ``n = x.size`` and ``m = prod(output_shape)``.
    """
    if colored_pattern is not None and not isinstance(colored_pattern, ColoredPattern):
        raise TypeError(
            f"Expected ColoredPattern, got {type(colored_pattern).__name__}. "
            "The API changed: use jacobian(f)(x) instead of jacobian(f, x)."
        )

    def jac_fn(x: ArrayLike) -> BCOO:
        return _eval_jacobian(f, jnp.asarray(x), colored_pattern)

    return jac_fn


def hessian(
    f: Callable[[ArrayLike], ArrayLike],
    colored_pattern: ColoredPattern | None = None,
) -> Callable[[ArrayLike], BCOO]:
    """Build a sparse Hessian function using coloring and HVPs.

    Uses symmetric (star) coloring and forward-over-reverse
    Hessian-vector products for efficiency.

    Args:
        f: Scalar-valued function taking an array.
            Input may be multi-dimensional.
        colored_pattern: Optional pre-computed [`ColoredPattern`][asdex.ColoredPattern]
            from [`hessian_coloring`][asdex.hessian_coloring].
            If None, sparsity is detected and colored automatically on each call.

    Returns:
        A function that takes an input array and returns
        the sparse Hessian as BCOO of shape ``(n, n)``
        where ``n = x.size``.
    """
    if colored_pattern is not None and not isinstance(colored_pattern, ColoredPattern):
        raise TypeError(
            f"Expected ColoredPattern, got {type(colored_pattern).__name__}. "
            "The API changed: use hessian(f)(x) instead of hessian(f, x)."
        )

    def hess_fn(x: ArrayLike) -> BCOO:
        return _eval_hessian(f, jnp.asarray(x), colored_pattern)

    return hess_fn


# =========================================================================
# Internal evaluation logic
# =========================================================================


def _eval_jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    colored_pattern: ColoredPattern | None,
) -> BCOO:
    """Evaluate the sparse Jacobian of f at x."""
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
    if sparsity.nnz == 0:
        return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(m, n))

    if colored_pattern.mode == "VJP":
        return _jacobian_rows(f, x, colored_pattern, out_shape)
    return _jacobian_cols(f, x, colored_pattern)


def _eval_hessian(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    colored_pattern: ColoredPattern | None,
) -> BCOO:
    """Evaluate the sparse Hessian of f at x."""
    n = x.size

    if colored_pattern is None:
        sparsity = _detect_hessian_sparsity(f, x.shape)
        colored_pattern = color_hessian_pattern(sparsity)

    sparsity = colored_pattern.sparsity

    # Handle edge case: all-zero Hessian
    if sparsity.nnz == 0:
        return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(n, n))

    grads = _compute_hvps(f, x, colored_pattern)
    return _decompress(colored_pattern, grads)


# =========================================================================
# Private helpers: Jacobian
# =========================================================================


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


# =========================================================================
# Private helpers: Hessian
# =========================================================================


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


# =========================================================================
# Private helpers: decompression
# =========================================================================


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

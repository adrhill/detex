"""Sparse Jacobian and Hessian computation using coloring and AD."""

from collections.abc import Callable
from typing import assert_never

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from numpy.typing import ArrayLike

from asdex.coloring import hessian_coloring as _hessian_coloring
from asdex.coloring import jacobian_coloring as _jacobian_coloring
from asdex.detection import _ensure_scalar
from asdex.modes import HessianMode, JacobianMode
from asdex.pattern import ColoredPattern

# Public API


def jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    input_shape: int | tuple[int, ...],
    *,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
) -> Callable[[ArrayLike], BCOO]:
    """Detect sparsity, color, and return a function computing sparse Jacobians.

    Combines [`jacobian_coloring`][asdex.jacobian_coloring]
    and [`jacobian_from_coloring`][asdex.jacobian_from_coloring]
    in one call.

    Args:
        f: Function taking an array and returning an array.
            Input and output may be multi-dimensional.
        input_shape: Shape of the input array.
        mode: AD mode.
            ``"fwd"`` uses JVPs (forward-mode AD),
            ``"rev"`` uses VJPs (reverse-mode AD).
            ``None`` picks whichever of fwd/rev needs fewer colors.
        symmetric: Whether to use symmetric (star) coloring.
            Requires a square Jacobian.

    Returns:
        A function that takes an input array and returns
            the sparse Jacobian as BCOO of shape ``(m, n)``
            where ``n = x.size`` and ``m = prod(output_shape)``.
    """
    coloring = _jacobian_coloring(f, input_shape, mode=mode, symmetric=symmetric)
    return jacobian_from_coloring(f, coloring)


def hessian(
    f: Callable[[ArrayLike], ArrayLike],
    input_shape: int | tuple[int, ...],
    *,
    mode: HessianMode | None = None,
    symmetric: bool = True,
) -> Callable[[ArrayLike], BCOO]:
    """Detect sparsity, color, and return a function computing sparse Hessians.

    Combines [`hessian_coloring`][asdex.hessian_coloring]
    and [`hessian_from_coloring`][asdex.hessian_from_coloring]
    in one call.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function taking an array.
            Input may be multi-dimensional.
        input_shape: Shape of the input array.
        mode: AD composition strategy for Hessian-vector products.
            ``"fwd_over_rev"`` uses forward-over-reverse,
            ``"rev_over_fwd"`` uses reverse-over-forward,
            ``"rev_over_rev"`` uses reverse-over-reverse.
            Defaults to ``"fwd_over_rev"``.
        symmetric: Whether to use symmetric (star) coloring.
            Defaults to True (exploits H = H^T for fewer colors).

    Returns:
        A function that takes an input array and returns
            the sparse Hessian as BCOO of shape ``(n, n)``
            where ``n = x.size``.
    """
    coloring = _hessian_coloring(f, input_shape, mode=mode, symmetric=symmetric)
    return hessian_from_coloring(f, coloring)


def jacobian_from_coloring(
    f: Callable[[ArrayLike], ArrayLike],
    coloring: ColoredPattern,
) -> Callable[[ArrayLike], BCOO]:
    """Build a sparse Jacobian function from a pre-computed coloring.

    Uses row coloring + VJPs or column coloring + JVPs,
    depending on which needs fewer colors.

    Args:
        f: Function taking an array and returning an array.
            Input and output may be multi-dimensional.
        coloring: Pre-computed [`ColoredPattern`][asdex.ColoredPattern]
            from [`jacobian_coloring`][asdex.jacobian_coloring].

    Returns:
        A function that takes an input array and returns
            the sparse Jacobian as BCOO of shape ``(m, n)``
            where ``n = x.size`` and ``m = prod(output_shape)``.
    """

    def jac_fn(x: ArrayLike) -> BCOO:
        return _eval_jacobian(f, jnp.asarray(x), coloring)

    return jac_fn


def hessian_from_coloring(
    f: Callable[[ArrayLike], ArrayLike],
    coloring: ColoredPattern,
) -> Callable[[ArrayLike], BCOO]:
    """Build a sparse Hessian function from a pre-computed coloring.

    Uses symmetric (star) coloring and Hessian-vector products by default.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function taking an array.
            Input may be multi-dimensional.
        coloring: Pre-computed [`ColoredPattern`][asdex.ColoredPattern]
            from [`hessian_coloring`][asdex.hessian_coloring].

    Returns:
        A function that takes an input array and returns
            the sparse Hessian as BCOO of shape ``(n, n)``
            where ``n = x.size``.
    """

    def hess_fn(x: ArrayLike) -> BCOO:
        return _eval_hessian(f, jnp.asarray(x), coloring)

    return hess_fn


# Internal evaluation logic


def _eval_jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
) -> BCOO:
    """Evaluate the sparse Jacobian of f at x."""
    n = x.size

    expected = coloring.sparsity.input_shape
    if x.shape != expected:
        raise ValueError(
            f"Input shape {x.shape} does not match the colored pattern, "
            f"which expects shape {expected}."
        )

    sparsity = coloring.sparsity
    m = sparsity.m
    out_shape = jax.eval_shape(f, jnp.zeros_like(x)).shape

    # Handle edge case: no outputs
    if m == 0:
        return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(0, n))

    # Handle edge case: all-zero Jacobian
    if sparsity.nnz == 0:
        return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(m, n))

    match coloring.mode:
        case "rev":
            return _jacobian_rows(f, x, coloring, out_shape)
        case "fwd":
            return _jacobian_cols(f, x, coloring)
        case _ as unreachable:
            assert_never(unreachable)  # type: ignore[type-assertion-failure]


def _eval_hessian(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
) -> BCOO:
    """Evaluate the sparse Hessian of f at x.

    If ``f`` returns a squeezable shape like ``(1,)``,
    it is automatically squeezed to scalar.
    """
    f = _ensure_scalar(f, x.shape)
    n = x.size

    expected = coloring.sparsity.input_shape
    if x.shape != expected:
        raise ValueError(
            f"Input shape {x.shape} does not match the colored pattern, "
            f"which expects shape {expected}."
        )

    sparsity = coloring.sparsity

    # Handle edge case: all-zero Hessian
    if sparsity.nnz == 0:
        return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(n, n))

    grads = _compute_hvps(f, x, coloring)
    return _decompress(coloring, grads)


# Private helpers: Jacobian


def _jacobian_rows(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
    out_shape: tuple[int, ...],
) -> BCOO:
    """Compute sparse Jacobian via row coloring + VJPs."""
    seeds = jnp.asarray(coloring._seed_matrix, dtype=x.dtype)
    _, vjp_fn = jax.vjp(f, x)

    def single_vjp(seed: jax.Array) -> jax.Array:
        (grad,) = vjp_fn(seed.reshape(out_shape))
        return grad.ravel()

    return _decompress(coloring, jax.vmap(single_vjp)(seeds))


def _jacobian_cols(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
) -> BCOO:
    """Compute sparse Jacobian via column coloring + JVPs."""
    seeds = jnp.asarray(coloring._seed_matrix, dtype=x.dtype)

    def single_jvp(seed: jax.Array) -> jax.Array:
        _, jvp_out = jax.jvp(f, (x,), (seed.reshape(x.shape),))
        return jvp_out.ravel()

    return _decompress(coloring, jax.vmap(single_jvp)(seeds))


# Private helpers: Hessian


def _compute_hvps(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
) -> jax.Array:
    """Compute one HVP per color using pre-computed seed matrix."""
    seeds = jnp.asarray(coloring._seed_matrix, dtype=x.dtype)

    match coloring.mode:
        case "fwd_over_rev":

            def single_hvp(v: jax.Array) -> jax.Array:
                _, hvp = jax.jvp(jax.grad(f), (x,), (v.reshape(x.shape),))
                return hvp.ravel()

        case "rev_over_fwd":

            def single_hvp(v: jax.Array) -> jax.Array:
                return jax.grad(lambda p: jax.jvp(f, (p,), (v.reshape(x.shape),))[1])(
                    x
                ).ravel()

        case "rev_over_rev":

            def single_hvp(v: jax.Array) -> jax.Array:
                return jax.grad(lambda y: jnp.vdot(jax.grad(f)(y), v.reshape(x.shape)))(
                    x
                ).ravel()

        case _ as unreachable:
            assert_never(unreachable)  # type: ignore[type-assertion-failure]

    return jax.vmap(single_hvp)(seeds)


# Private helpers: decompression


def _decompress(coloring: ColoredPattern, compressed: jax.Array) -> BCOO:
    """Extract sparse entries from compressed gradient rows.

    Uses pre-computed extraction indices on the ``ColoredPattern``
    to vectorize the decompression step
    (no Python loop over nnz entries).

    Args:
        coloring: Colored sparsity pattern with cached indices.
        compressed: JAX array of shape (num_colors, vector_len),
            one row per color.

    Returns:
        Sparse matrix as BCOO in sparsity-pattern order.
    """
    color_idx, elem_idx = coloring._extraction_indices
    data = compressed[jnp.asarray(color_idx), jnp.asarray(elem_idx)]
    return coloring.sparsity.to_bcoo(data=data)

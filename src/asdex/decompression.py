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

from asdex.coloring import ColoredPattern, color, star_color
from asdex.detection import hessian_sparsity as _detect_hessian_sparsity
from asdex.detection import jacobian_sparsity as _detect_sparsity
from asdex.pattern import SparsityPattern


def _compute_vjp_for_color(
    f: Callable[[ArrayLike], ArrayLike],
    x: NDArray,
    row_mask: NDArray[np.bool_],
    out_shape: tuple[int, ...],
) -> NDArray:
    """Compute VJP with seed vector having 1s at masked positions.

    Args:
        f: Function to differentiate
        x: Input point
        row_mask: Boolean mask of shape (m,) indicating which rows to compute
        out_shape: Shape of the function output, used to reshape the seed

    Returns:
        Flattened gradient vector of shape (n,) - the VJP result
    """
    _, vjp_fn = jax.vjp(f, x)
    seed = row_mask.astype(x.dtype).reshape(out_shape)
    (grad,) = vjp_fn(seed)
    return np.asarray(grad).ravel()


def _compute_jvp_for_color(
    f: Callable[[ArrayLike], ArrayLike],
    x: NDArray,
    col_mask: NDArray[np.bool_],
    in_shape: tuple[int, ...],
) -> NDArray:
    """Compute JVP with tangent vector having 1s at masked positions.

    Args:
        f: Function to differentiate
        x: Input point
        col_mask: Boolean mask of shape (n,) indicating which columns to compute
        in_shape: Shape of the function input, used to reshape the tangent

    Returns:
        Flattened tangent-output vector of shape (m,) - the JVP result
    """
    tangent = col_mask.astype(x.dtype).reshape(in_shape)
    _, jvp_out = jax.jvp(f, (x,), (tangent,))
    return np.asarray(jvp_out).ravel()


def _compute_hvp_for_color(
    f: Callable[[ArrayLike], ArrayLike],
    x: NDArray,
    row_mask: NDArray[np.bool_],
    in_shape: tuple[int, ...],
) -> NDArray:
    """Compute HVP with tangent vector having 1s at masked positions.

    Uses forward-over-reverse AD which is more efficient than VJP-on-gradient.

    Args:
        f: Scalar-valued function to differentiate
        x: Input point
        row_mask: Boolean mask of shape (n,) indicating which rows to compute
        in_shape: Shape of the function input, used to reshape the tangent

    Returns:
        Flattened HVP result vector of shape (n,)
    """
    tangent = row_mask.astype(x.dtype).reshape(in_shape)
    _, hvp = jax.jvp(jax.grad(f), (x,), (tangent,))
    return np.asarray(hvp).ravel()


def _decompress_jacobian(
    sparsity: SparsityPattern,
    colors: NDArray[np.int32],
    grads: list[NDArray],
) -> BCOO:
    """Extract Jacobian entries from VJP results (row coloring).

    For each non-zero (i, j) in the sparsity pattern, the value is extracted
    from the gradient corresponding to row i's color. Due to the orthogonality
    property (same-colored rows don't share columns), each gradient entry
    uniquely corresponds to one Jacobian row.

    Args:
        sparsity: Sparsity pattern
        colors: Color assignment for each row
        grads: List of gradient vectors, one per color

    Returns:
        Sparse Jacobian as BCOO matrix
    """
    rows = sparsity.rows
    cols = sparsity.cols

    data = np.empty(len(rows), dtype=grads[0].dtype)
    for k, (i, j) in enumerate(zip(rows, cols, strict=True)):
        color = colors[i]
        data[k] = grads[color][j]

    return sparsity.to_bcoo(data=jnp.array(data))


def _decompress_jacobian_from_jvps(
    sparsity: SparsityPattern,
    colors: NDArray[np.int32],
    jvps: list[NDArray],
) -> BCOO:
    """Extract Jacobian entries from JVP results (column coloring).

    For each non-zero (i, j) in the sparsity pattern, the value is extracted
    from the JVP corresponding to column j's color.
    Due to the orthogonality property
    (same-colored columns don't share rows),
    each JVP output entry uniquely corresponds to one Jacobian column.

    Args:
        sparsity: Sparsity pattern
        colors: Color assignment for each column
        jvps: List of JVP output vectors, one per color

    Returns:
        Sparse Jacobian as BCOO matrix
    """
    rows = sparsity.rows
    cols = sparsity.cols

    data = np.empty(len(rows), dtype=jvps[0].dtype)
    for k, (i, j) in enumerate(zip(rows, cols, strict=True)):
        color = colors[j]
        data[k] = jvps[color][i]

    return sparsity.to_bcoo(data=jnp.array(data))


def _decompress_hessian_star(
    sparsity: SparsityPattern,
    colors: NDArray[np.int32],
    compressed: list[NDArray],
) -> BCOO:
    """Extract Hessian entries from HVP results using star coloring.

    For diagonal entries (i, i): extract from compressed[colors[i]][i].
    For off-diagonal entries (i, j): use compressed[colors[i]][j] if
    colors[i] is unique among column j's neighbors;
    otherwise use compressed[colors[j]][i].
    Star coloring guarantees at least one direction is valid.

    Args:
        sparsity: Symmetric sparsity pattern
        colors: Star coloring assignment
        compressed: List of HVP result vectors, one per color

    Returns:
        Sparse Hessian as BCOO matrix
    """
    rows = sparsity.rows
    cols = sparsity.cols

    # Build col_to_rows for checking color uniqueness
    col_to_rows = sparsity.col_to_rows

    data = np.empty(len(rows), dtype=compressed[0].dtype)
    for k, (i, j) in enumerate(zip(rows, cols, strict=True)):
        i, j = int(i), int(j)
        if i == j:
            # Diagonal: always compressed[colors[i]][i]
            data[k] = compressed[colors[i]][i]
        else:
            # Off-diagonal: try row i's color first.
            # colors[i] is "unique" in column j if no other row in column j
            # has the same color.
            color_i = colors[i]
            unique = True
            for r in col_to_rows.get(j, []):
                if r != i and colors[r] == color_i:
                    unique = False
                    break
            if unique:
                data[k] = compressed[color_i][j]
            else:
                data[k] = compressed[colors[j]][i]

    return sparsity.to_bcoo(data=jnp.array(data))


def sparse_jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    coloring: ColoredPattern | None = None,
) -> BCOO:
    """Compute sparse Jacobian using coloring and AD.

    Uses row coloring + VJPs or column coloring + JVPs,
    depending on which needs fewer colors.

    Args:
        f: Function taking an array and returning an array.
            Both may be multi-dimensional.
        x: Input point (any shape).
        coloring: Optional pre-computed :class:`ColoredPattern`
            from :func:`color`.
            If None, sparsity is detected and colored automatically.

    Returns:
        Sparse Jacobian matrix of shape (m, n) as BCOO,
        where n = x.size and m = prod(output_shape)
    """
    x = np.asarray(x)
    n = x.size

    if coloring is None:
        sparsity = _detect_sparsity(f, x.shape)
        coloring = color(sparsity)

    sparsity = coloring.sparsity
    m = sparsity.m
    out_shape = jax.eval_shape(f, jnp.zeros_like(x)).shape

    # Handle edge case: no outputs
    if m == 0:
        return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(0, n))

    # Handle edge case: all-zero Jacobian
    if sparsity.nse == 0:
        return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(m, n))

    if coloring.partition == "row":
        return _sparse_jacobian_rows(f, x, sparsity, coloring.colors, out_shape)
    else:
        return _sparse_jacobian_cols(f, x, sparsity, coloring.colors)


def _sparse_jacobian_rows(
    f: Callable[[ArrayLike], ArrayLike],
    x: NDArray,
    sparsity: SparsityPattern,
    colors: NDArray[np.int32],
    out_shape: tuple[int, ...],
) -> BCOO:
    """Compute sparse Jacobian via row coloring + VJPs."""
    num_colors = int(colors.max()) + 1 if len(colors) > 0 else 0

    grads: list[NDArray] = []
    for c in range(num_colors):
        row_mask = colors == c
        grad = _compute_vjp_for_color(f, x, row_mask, out_shape)
        grads.append(grad)

    return _decompress_jacobian(sparsity, colors, grads)


def _sparse_jacobian_cols(
    f: Callable[[ArrayLike], ArrayLike],
    x: NDArray,
    sparsity: SparsityPattern,
    colors: NDArray[np.int32],
) -> BCOO:
    """Compute sparse Jacobian via column coloring + JVPs."""
    num_colors = int(colors.max()) + 1 if len(colors) > 0 else 0

    jvps: list[NDArray] = []
    for c in range(num_colors):
        col_mask = colors == c
        jvp_out = _compute_jvp_for_color(f, x, col_mask, x.shape)
        jvps.append(jvp_out)

    return _decompress_jacobian_from_jvps(sparsity, colors, jvps)


def sparse_hessian(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    sparsity: SparsityPattern | None = None,
    colors: NDArray[np.int32] | None = None,
) -> BCOO:
    """Compute sparse Hessian using coloring and HVPs.

    When ``colors`` is None (default), uses star coloring for fewer colors
    and symmetric decompression.
    When ``colors`` is provided (backward compatibility with pre-computed
    ``color_rows`` colors), uses the original row-coloring decompression.

    Uses forward-over-reverse Hessian-vector products for efficiency.

    Args:
        f: Scalar-valued function returning a scalar.
            Input may be multi-dimensional.
        x: Input point (any shape).
        sparsity: Optional pre-computed Hessian sparsity pattern from
            hessian_sparsity(). If None, detected automatically.
        colors: Optional pre-computed row coloring from color_rows().
            If None, star coloring is computed and used automatically.

    Returns:
        Sparse Hessian matrix of shape (n, n) as BCOO
    """

    x = np.asarray(x)
    n = x.size

    if sparsity is None:
        sparsity = _detect_hessian_sparsity(f, x.shape)

    # Handle edge case: all-zero Hessian
    if sparsity.nse == 0:
        return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(n, n))

    if colors is not None:
        # Backward compat: pre-computed row coloring → row-based decompression
        num_colors = int(colors.max()) + 1 if len(colors) > 0 else 0
        grads = _compute_hvps(f, x, colors, num_colors)
        return _decompress_jacobian(sparsity, colors, grads)

    # Default: star coloring + symmetric decompression
    colors, num_colors = star_color(sparsity)
    grads = _compute_hvps(f, x, colors, num_colors)
    return _decompress_hessian_star(sparsity, colors, grads)


def _compute_hvps(
    f: Callable[[ArrayLike], ArrayLike],
    x: NDArray,
    colors: NDArray[np.int32],
    num_colors: int,
) -> list[NDArray]:
    """Compute one HVP per color."""
    grads: list[NDArray] = []
    for c in range(num_colors):
        row_mask = colors == c
        hvp_result = _compute_hvp_for_color(f, x, row_mask, x.shape)
        grads.append(hvp_result)
    return grads

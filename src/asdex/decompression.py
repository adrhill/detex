"""Sparse Jacobian and Hessian computation using VJPs and row coloring.

The key insight: rows that don't share non-zero columns can be computed together
in a single VJP by using a combined seed vector. Coloring identifies which rows
are structurally orthogonal, reducing the number of backward passes from m
(output dimension) to the number of colors.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from numpy.typing import ArrayLike, NDArray

from asdex.coloring import color_rows
from asdex.detection import hessian_sparsity as _detect_hessian_sparsity
from asdex.detection import jacobian_sparsity as _detect_sparsity


def _compute_vjp_for_color(
    f: Callable[[ArrayLike], ArrayLike],
    x: NDArray,
    row_mask: NDArray[np.bool_],
) -> NDArray:
    """Compute VJP with seed vector having 1s at masked positions.

    Args:
        f: Function to differentiate
        x: Input point
        row_mask: Boolean mask of shape (m,) indicating which rows to compute

    Returns:
        Gradient vector of shape (n,) - the VJP result
    """
    _, vjp_fn = jax.vjp(f, x)
    seed = row_mask.astype(x.dtype)
    (grad,) = vjp_fn(seed)
    return np.asarray(grad)


def _compute_hvp_for_color(
    f: Callable[[ArrayLike], ArrayLike],
    x: NDArray,
    row_mask: NDArray[np.bool_],
) -> NDArray:
    """Compute HVP with tangent vector having 1s at masked positions.

    Uses forward-over-reverse AD which is more efficient than VJP-on-gradient.

    Args:
        f: Scalar-valued function to differentiate
        x: Input point
        row_mask: Boolean mask of shape (n,) indicating which rows to compute

    Returns:
        HVP result vector of shape (n,)
    """
    tangent = row_mask.astype(x.dtype)
    _, hvp = jax.jvp(jax.grad(f), (x,), (tangent,))
    return np.asarray(hvp)


def _decompress_jacobian(
    sparsity: BCOO,
    colors: NDArray[np.int32],
    grads: list[NDArray],
) -> BCOO:
    """Extract Jacobian entries from VJP results.

    For each non-zero (i, j) in the sparsity pattern, the value is extracted
    from the gradient corresponding to row i's color. Due to the orthogonality
    property (same-colored rows don't share columns), each gradient entry
    uniquely corresponds to one Jacobian row.

    Args:
        sparsity: Sparsity pattern as BCOO matrix
        colors: Color assignment for each row
        grads: List of gradient vectors, one per color

    Returns:
        Sparse Jacobian as BCOO matrix
    """
    indices = np.asarray(sparsity.indices)
    rows = indices[:, 0]
    cols = indices[:, 1]

    data = np.empty(len(rows), dtype=grads[0].dtype)
    for k, (i, j) in enumerate(zip(rows, cols, strict=True)):
        color = colors[i]
        data[k] = grads[color][j]

    return BCOO((jnp.array(data), sparsity.indices), shape=sparsity.shape)


def sparse_jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    sparsity: BCOO | None = None,
    colors: NDArray[np.int32] | None = None,
) -> BCOO:
    """Compute sparse Jacobian using coloring and VJPs.

    Uses row-wise coloring to identify structurally orthogonal rows, then
    computes the Jacobian with one VJP per color instead of one per row.

    Args:
        f: Function from R^n to R^m (takes 1D array, returns 1D array)
        x: Input point of shape (n,)
        sparsity: Optional pre-computed sparsity pattern. If None, detected
            automatically.
        colors: Optional pre-computed row coloring from color_rows(). If None,
            computed automatically from sparsity.

    Returns:
        Sparse Jacobian matrix of shape (m, n) as BCOO
    """
    x = np.asarray(x)
    n = x.shape[0]

    if sparsity is None:
        sparsity = _detect_sparsity(f, n)

    m = sparsity.shape[0]

    # Handle edge case: no outputs
    if m == 0:
        return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(0, n))

    if colors is None:
        colors, num_colors = color_rows(sparsity)
    else:
        num_colors = int(colors.max()) + 1 if len(colors) > 0 else 0

    # Handle edge case: all-zero Jacobian
    if sparsity.nse == 0:
        return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(m, n))

    # Compute one VJP per color
    grads: list[NDArray] = []
    for c in range(num_colors):
        row_mask = colors == c
        grad = _compute_vjp_for_color(f, x, row_mask)
        grads.append(grad)

    return _decompress_jacobian(sparsity, colors, grads)


def sparse_hessian(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    sparsity: BCOO | None = None,
    colors: NDArray[np.int32] | None = None,
) -> BCOO:
    """Compute sparse Hessian using coloring and HVPs.

    Uses forward-over-reverse Hessian-vector products for efficiency.
    This is faster than VJP-on-gradient because forward-mode has less
    overhead than reverse-mode for the outer differentiation.

    Args:
        f: Scalar-valued function from R^n to R (takes 1D array, returns scalar)
        x: Input point of shape (n,)
        sparsity: Optional pre-computed Hessian sparsity pattern from
            hessian_sparsity(). If None, detected automatically.
        colors: Optional pre-computed row coloring from color_rows(). If None,
            computed automatically from sparsity.

    Returns:
        Sparse Hessian matrix of shape (n, n) as BCOO
    """

    x = np.asarray(x)
    n = x.shape[0]

    if sparsity is None:
        sparsity = _detect_hessian_sparsity(f, n)

    if colors is None:
        colors, num_colors = color_rows(sparsity)
    else:
        num_colors = int(colors.max()) + 1 if len(colors) > 0 else 0

    # Handle edge case: all-zero Hessian
    if sparsity.nse == 0:
        return BCOO((jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=(n, n))

    # Compute one HVP per color (forward-over-reverse)
    grads: list[NDArray] = []
    for c in range(num_colors):
        row_mask = colors == c
        hvp_result = _compute_hvp_for_color(f, x, row_mask)
        grads.append(hvp_result)

    return _decompress_jacobian(sparsity, colors, grads)

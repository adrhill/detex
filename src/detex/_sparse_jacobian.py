"""Sparse Jacobian computation using VJPs and row coloring.

The key insight: rows that don't share non-zero columns can be computed together
in a single VJP by using a combined seed vector. Coloring identifies which rows
are structurally orthogonal, reducing the number of backward passes from m
(output dimension) to the number of colors.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csr_matrix

from detex._coloring import color_rows
from detex._propagate import prop_jaxpr


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


def _decompress_jacobian(
    sparsity: coo_matrix,
    colors: NDArray[np.int32],
    grads: list[NDArray],
) -> csr_matrix:
    """Extract Jacobian entries from VJP results.

    For each non-zero (i, j) in the sparsity pattern, the value is extracted
    from the gradient corresponding to row i's color. Due to the orthogonality
    property (same-colored rows don't share columns), each gradient entry
    uniquely corresponds to one Jacobian row.

    Args:
        sparsity: Sparsity pattern as COO matrix
        colors: Color assignment for each row
        grads: List of gradient vectors, one per color

    Returns:
        Sparse Jacobian in CSR format
    """
    coo = sparsity.tocoo()
    rows = coo.row
    cols = coo.col

    data = np.empty(len(rows), dtype=grads[0].dtype)
    for k, (i, j) in enumerate(zip(rows, cols, strict=True)):
        color = colors[i]
        data[k] = grads[color][j]

    return csr_matrix((data, (rows, cols)), shape=sparsity.shape)


def sparse_jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    sparsity: coo_matrix | None = None,
    colors: NDArray[np.int32] | None = None,
) -> csr_matrix:
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
        Sparse Jacobian matrix of shape (m, n) in CSR format
    """
    x = np.asarray(x)
    n = x.shape[0]

    if sparsity is None:
        sparsity = _detect_sparsity(f, n)

    m = sparsity.shape[0]

    # Handle edge case: no outputs
    if m == 0:
        return csr_matrix((0, n), dtype=x.dtype)

    if colors is None:
        colors, num_colors = color_rows(sparsity)
    else:
        num_colors = int(colors.max()) + 1 if len(colors) > 0 else 0

    # Handle edge case: all-zero Jacobian
    if sparsity.nnz == 0:
        return csr_matrix((m, n), dtype=x.dtype)

    # Compute one VJP per color
    grads: list[NDArray] = []
    for c in range(num_colors):
        row_mask = colors == c
        grad = _compute_vjp_for_color(f, x, row_mask)
        grads.append(grad)

    return _decompress_jacobian(sparsity, colors, grads)


def _detect_sparsity(f: Callable[[ArrayLike], ArrayLike], n: int) -> coo_matrix:
    """Detect Jacobian sparsity pattern via jaxpr analysis.

    This is a copy of the logic from jacobian_sparsity to avoid circular imports.
    """
    dummy_input = jnp.zeros(n)
    closed_jaxpr = jax.make_jaxpr(f)(dummy_input)
    jaxpr = closed_jaxpr.jaxpr
    m = int(jax.eval_shape(f, dummy_input).size)

    input_indices = [[{i} for i in range(n)]]
    output_indices_list = prop_jaxpr(jaxpr, input_indices)
    out_indices = output_indices_list[0] if output_indices_list else []

    rows = []
    cols = []
    for i, deps in enumerate(out_indices):
        for j in deps:
            rows.append(i)
            cols.append(j)

    return coo_matrix(([True] * len(rows), (rows, cols)), shape=(m, n), dtype=bool)

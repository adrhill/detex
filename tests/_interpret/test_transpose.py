"""Tests for the transpose propagation handler.

Tests N-dimensional permutations, identity permutations,
size-1 dimensions, and higher-order cyclic permutations.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


def _transpose_jacobian(in_shape: tuple[int, ...], permutation: tuple[int, ...]):
    """Build the expected permutation Jacobian for a transpose.

    For each flat output index, compute which flat input index it reads from.
    """
    n = int(np.prod(in_shape))
    out_shape = tuple(in_shape[p] for p in permutation)

    # Inverse permutation: maps input dim → output dim.
    inv_perm = [0] * len(permutation)
    for d, p in enumerate(permutation):
        inv_perm[p] = d

    expected = np.zeros((n, n), dtype=int)
    for out_flat in range(n):
        out_coord = np.unravel_index(out_flat, out_shape)
        in_coord = tuple(out_coord[inv_perm[d]] for d in range(len(in_shape)))
        in_flat = np.ravel_multi_index(in_coord, in_shape)
        expected[out_flat, in_flat] = 1
    return expected


@pytest.mark.array_ops
def test_transpose_1d():
    """1D transpose with permutation (0,) is identity."""

    def f(x):
        return jnp.transpose(x, (0,))

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_transpose_identity_2d():
    """2D transpose with identity permutation (0, 1) is identity Jacobian."""

    def f(x):
        mat = x.reshape(2, 3)
        return jnp.transpose(mat, (0, 1)).flatten()

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_transpose_3d_swap_last_two():
    """3D transpose swapping the last two axes: (0, 2, 1)."""
    in_shape = (2, 3, 4)

    def f(x):
        arr = x.reshape(in_shape)
        return jnp.transpose(arr, (0, 2, 1)).flatten()

    result = jacobian_sparsity(f, input_shape=24).todense().astype(int)
    expected = _transpose_jacobian(in_shape, (0, 2, 1))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_transpose_3d_cyclic():
    """3D transpose with cyclic permutation (1, 2, 0).

    result[i, j, k] = input[k, i, j].
    """
    in_shape = (2, 3, 4)

    def f(x):
        arr = x.reshape(in_shape)
        return jnp.transpose(arr, (1, 2, 0)).flatten()

    result = jacobian_sparsity(f, input_shape=24).todense().astype(int)
    expected = _transpose_jacobian(in_shape, (1, 2, 0))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_transpose_3d_reverse():
    """3D transpose with full reversal (2, 1, 0)."""
    in_shape = (2, 3, 4)

    def f(x):
        arr = x.reshape(in_shape)
        return jnp.transpose(arr, (2, 1, 0)).flatten()

    result = jacobian_sparsity(f, input_shape=24).todense().astype(int)
    expected = _transpose_jacobian(in_shape, (2, 1, 0))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_transpose_size_one_dims():
    """Transpose with size-1 dimensions (no-op in practice, but tests indexing)."""
    in_shape = (1, 3, 1)

    def f(x):
        arr = x.reshape(in_shape)
        return jnp.transpose(arr, (2, 1, 0)).flatten()

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # All dimensions except dim 1 have size 1, so this is effectively identity.
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_transpose_4d_nhwc_to_nchw():
    """4D transpose simulating NHWC → NCHW layout change: (0, 3, 1, 2)."""
    in_shape = (1, 2, 3, 2)
    perm = (0, 3, 1, 2)

    def f(x):
        arr = x.reshape(in_shape)
        return jnp.transpose(arr, perm).flatten()

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    expected = _transpose_jacobian(in_shape, perm)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_transpose_square_involution():
    """Transposing a square matrix twice gives the identity Jacobian."""

    def f(x):
        mat = x.reshape(3, 3)
        return mat.T.T.flatten()

    result = jacobian_sparsity(f, input_shape=9).todense().astype(int)
    expected = np.eye(9, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_moveaxis():
    """jnp.moveaxis lowers to transpose; verify end-to-end."""
    in_shape = (2, 3, 4)

    def f(x):
        arr = x.reshape(in_shape)
        return jnp.moveaxis(arr, 0, -1).flatten()  # (3, 4, 2)

    result = jacobian_sparsity(f, input_shape=24).todense().astype(int)
    # moveaxis(arr, 0, -1) is equivalent to transpose(arr, (1, 2, 0)).
    expected = _transpose_jacobian(in_shape, (1, 2, 0))
    np.testing.assert_array_equal(result, expected)

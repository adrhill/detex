"""Tests for the reduce_max propagation handler.

Tests max reduction along single and multiple dimensions,
full reductions, size-1 dimensions, and high-level functions
that lower to reduce_max.
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


def _reduction_jacobian(in_shape: tuple[int, ...], axes: tuple[int, ...]) -> np.ndarray:
    """Build the expected Jacobian for a reduction operation.

    Each output element has a 1 for every input element
    that reduces into it (all elements sharing the same non-reduced coordinates).
    """
    n_in = int(np.prod(in_shape))
    kept_dims = [d for d in range(len(in_shape)) if d not in axes]
    out_shape = tuple(in_shape[d] for d in kept_dims)
    n_out = int(np.prod(out_shape)) if out_shape else 1

    expected = np.zeros((n_out, n_in), dtype=int)
    for in_flat in range(n_in):
        in_coord = np.unravel_index(in_flat, in_shape)
        out_coord = tuple(in_coord[d] for d in kept_dims)
        out_flat = np.ravel_multi_index(out_coord, out_shape) if out_shape else 0
        expected[out_flat, in_flat] = 1
    return expected


# ── 1D ──────────────────────────────────────────────────────────────────


@pytest.mark.reduction
def test_reduce_max_1d_full():
    """Full reduction of a 1D array: single output depends on all inputs."""
    shape = (5,)

    def f(x):
        return lax.reduce_max(x, axes=(0,))

    result = jacobian_sparsity(f, input_shape=shape).todense().astype(int)
    expected = _reduction_jacobian(shape, (0,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_max_1d_size_one():
    """Reducing a single-element array is trivial: output = input."""
    shape = (1,)

    def f(x):
        return lax.reduce_max(x, axes=(0,))

    result = jacobian_sparsity(f, input_shape=shape).todense().astype(int)
    expected = np.array([[1]])
    np.testing.assert_array_equal(result, expected)


# ── 2D ──────────────────────────────────────────────────────────────────


@pytest.mark.reduction
def test_reduce_max_2d_axis0():
    """Reduce along axis 0: each output column depends on its column of inputs."""
    shape = (3, 4)

    def f(x):
        return lax.reduce_max(x.reshape(shape), axes=(0,))

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    expected = _reduction_jacobian(shape, (0,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_max_2d_axis1():
    """Reduce along axis 1: each output row depends on its row of inputs."""
    shape = (3, 4)

    def f(x):
        return lax.reduce_max(x.reshape(shape), axes=(1,))

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    expected = _reduction_jacobian(shape, (1,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_max_2d_both_axes():
    """Reduce along both axes: full reduction to scalar."""
    shape = (3, 4)

    def f(x):
        return lax.reduce_max(x.reshape(shape), axes=(0, 1))

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    expected = _reduction_jacobian(shape, (0, 1))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_max_2d_size_one_reduced_dim():
    """Reducing a size-1 dimension: output shape unchanged, identity-like."""
    shape = (3, 1)

    def f(x):
        return lax.reduce_max(x.reshape(shape), axes=(1,))

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_max_2d_size_one_kept_dim():
    """Reducing along the non-size-1 dim: each output depends on a full column."""
    shape = (1, 5)

    def f(x):
        return lax.reduce_max(x.reshape(shape), axes=(1,))

    result = jacobian_sparsity(f, input_shape=5).todense().astype(int)
    expected = np.ones((1, 5), dtype=int)
    np.testing.assert_array_equal(result, expected)


# ── 3D ──────────────────────────────────────────────────────────────────


@pytest.mark.reduction
def test_reduce_max_3d_single_axis():
    """Reduce one axis of a 3D array."""
    shape = (2, 3, 4)

    def f(x):
        return lax.reduce_max(x.reshape(shape), axes=(1,)).flatten()

    result = jacobian_sparsity(f, input_shape=24).todense().astype(int)
    expected = _reduction_jacobian(shape, (1,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_max_3d_two_axes():
    """Reduce two axes of a 3D array."""
    shape = (2, 3, 4)

    def f(x):
        return lax.reduce_max(x.reshape(shape), axes=(0, 2)).flatten()

    result = jacobian_sparsity(f, input_shape=24).todense().astype(int)
    expected = _reduction_jacobian(shape, (0, 2))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_max_3d_full():
    """Full reduction of a 3D array."""
    shape = (2, 3, 4)

    def f(x):
        return lax.reduce_max(x.reshape(shape), axes=(0, 1, 2))

    result = jacobian_sparsity(f, input_shape=24).todense().astype(int)
    expected = np.ones((1, 24), dtype=int)
    np.testing.assert_array_equal(result, expected)


# ── 4D ──────────────────────────────────────────────────────────────────


@pytest.mark.reduction
def test_reduce_max_4d():
    """Reduce selected axes of a 4D array."""
    shape = (2, 2, 3, 2)

    def f(x):
        return lax.reduce_max(x.reshape(shape), axes=(1, 3)).flatten()

    result = jacobian_sparsity(f, input_shape=24).todense().astype(int)
    expected = _reduction_jacobian(shape, (1, 3))
    np.testing.assert_array_equal(result, expected)


# ── High-level functions ────────────────────────────────────────────────


@pytest.mark.reduction
def test_jnp_max_no_axis():
    """jnp.max without axis lowers to reduce_max over all axes."""

    def f(x):
        return jnp.max(x.reshape(2, 3)) * jnp.ones(1)

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    expected = np.ones((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_max_with_axis():
    """jnp.max with axis lowers to reduce_max along that axis."""
    shape = (2, 3)

    def f(x):
        return jnp.max(x.reshape(shape), axis=1)

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    expected = _reduction_jacobian(shape, (1,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_amax():
    """jnp.amax is an alias for jnp.max."""
    shape = (3, 4)

    def f(x):
        return jnp.amax(x.reshape(shape), axis=0)

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    expected = _reduction_jacobian(shape, (0,))
    np.testing.assert_array_equal(result, expected)


# ── Compositions ────────────────────────────────────────────────────────


@pytest.mark.reduction
def test_reduce_max_then_reduce_max():
    """Chained reductions: max along axis 0 then axis 0 again."""
    shape = (2, 3, 4)

    def f(x):
        a = lax.reduce_max(x.reshape(shape), axes=(0,))  # (3, 4)
        return lax.reduce_max(a, axes=(0,))  # (4,)

    result = jacobian_sparsity(f, input_shape=24).todense().astype(int)
    # Equivalent to reducing axes (0, 1) at once
    expected = _reduction_jacobian(shape, (0, 1))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_max_after_broadcast():
    """Reduce after broadcast: non-contiguous input dependencies."""

    def f(x):
        # x.shape = (3,), broadcast to (2, 3), then max along axis 0
        return lax.reduce_max(jnp.broadcast_to(x, (2, 3)), axes=(0,))

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # Each output depends only on the corresponding input (broadcast duplicates)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_max_after_reduce_sum():
    """Mix of reduce_sum then reduce_max."""
    shape = (2, 3, 4)

    def f(x):
        a = jnp.sum(x.reshape(shape), axis=2)  # (2, 3)
        return lax.reduce_max(a, axes=(1,))  # (2,)

    result = jacobian_sparsity(f, input_shape=24).todense().astype(int)
    # Equivalent to reducing all of axes 1 and 2
    expected = _reduction_jacobian(shape, (1, 2))
    np.testing.assert_array_equal(result, expected)

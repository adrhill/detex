"""Tests for the reduce_and, reduce_or, and reduce_xor propagation handlers.

These are bitwise reductions with zero derivative,
so the Jacobian is always the zero matrix.
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity

# ── reduce_and ──────────────────────────────────────────────────────────


@pytest.mark.reduction
def test_reduce_and_1d_full():
    """Full AND reduction of a 1D boolean array: zero Jacobian."""
    shape = (5,)

    def f(x):
        return lax.reduce_and(x > 0, axes=(0,)) * jnp.ones(1)

    result = jacobian_sparsity(f, input_shape=shape).todense().astype(int)
    expected = np.zeros((1, 5), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_and_2d_axis0():
    """AND reduction along axis 0: zero Jacobian."""
    shape = (3, 4)

    def f(x):
        return lax.reduce_and(x.reshape(shape) > 0, axes=(0,)) * jnp.ones(4)

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    expected = np.zeros((4, 12), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_and_2d_axis1():
    """AND reduction along axis 1: zero Jacobian."""
    shape = (3, 4)

    def f(x):
        return lax.reduce_and(x.reshape(shape) > 0, axes=(1,)) * jnp.ones(3)

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    expected = np.zeros((3, 12), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_and_2d_both_axes():
    """AND reduction along both axes: zero Jacobian."""
    shape = (3, 4)

    def f(x):
        return lax.reduce_and(x.reshape(shape) > 0, axes=(0, 1)) * jnp.ones(1)

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    expected = np.zeros((1, 12), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_and_3d_single_axis():
    """AND reduction of one axis in 3D: zero Jacobian."""
    shape = (2, 3, 4)

    def f(x):
        a = lax.reduce_and(x.reshape(shape) > 0, axes=(1,))
        return a.flatten() * jnp.ones(8)

    result = jacobian_sparsity(f, input_shape=24).todense().astype(int)
    expected = np.zeros((8, 24), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_all_no_axis():
    """jnp.all without axis lowers to reduce_and: zero Jacobian."""

    def f(x):
        return jnp.all(x.reshape(2, 3) > 0) * jnp.ones(1)

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    expected = np.zeros((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_all_with_axis():
    """jnp.all with axis lowers to reduce_and: zero Jacobian."""
    shape = (2, 3)

    def f(x):
        return jnp.all(x.reshape(shape) > 0, axis=1) * jnp.ones(2)

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    expected = np.zeros((2, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


# ── reduce_or ───────────────────────────────────────────────────────────


@pytest.mark.reduction
def test_reduce_or_1d_full():
    """Full OR reduction of a 1D boolean array: zero Jacobian."""
    shape = (5,)

    def f(x):
        return lax.reduce_or(x > 0, axes=(0,)) * jnp.ones(1)

    result = jacobian_sparsity(f, input_shape=shape).todense().astype(int)
    expected = np.zeros((1, 5), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_or_2d_axis0():
    """OR reduction along axis 0: zero Jacobian."""
    shape = (3, 4)

    def f(x):
        return lax.reduce_or(x.reshape(shape) > 0, axes=(0,)) * jnp.ones(4)

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    expected = np.zeros((4, 12), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_or_2d_both_axes():
    """OR reduction along both axes: zero Jacobian."""
    shape = (3, 4)

    def f(x):
        return lax.reduce_or(x.reshape(shape) > 0, axes=(0, 1)) * jnp.ones(1)

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    expected = np.zeros((1, 12), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_any_no_axis():
    """jnp.any without axis lowers to reduce_or: zero Jacobian."""

    def f(x):
        return jnp.any(x.reshape(2, 3) > 0) * jnp.ones(1)

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    expected = np.zeros((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_any_with_axis():
    """jnp.any with axis lowers to reduce_or: zero Jacobian."""
    shape = (2, 3)

    def f(x):
        return jnp.any(x.reshape(shape) > 0, axis=1) * jnp.ones(2)

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    expected = np.zeros((2, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


# ── reduce_xor ──────────────────────────────────────────────────────────


@pytest.mark.reduction
def test_reduce_xor_1d_full():
    """Full XOR reduction of a 1D boolean array: zero Jacobian."""
    shape = (5,)

    def f(x):
        return lax.reduce_xor(x > 0, axes=(0,)) * jnp.ones(1)

    result = jacobian_sparsity(f, input_shape=shape).todense().astype(int)
    expected = np.zeros((1, 5), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_xor_2d_axis0():
    """XOR reduction along axis 0: zero Jacobian."""
    shape = (3, 4)

    def f(x):
        return lax.reduce_xor(x.reshape(shape) > 0, axes=(0,)) * jnp.ones(4)

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    expected = np.zeros((4, 12), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_xor_2d_both_axes():
    """XOR reduction along both axes: zero Jacobian."""
    shape = (3, 4)

    def f(x):
        return lax.reduce_xor(x.reshape(shape) > 0, axes=(0, 1)) * jnp.ones(1)

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    expected = np.zeros((1, 12), dtype=int)
    np.testing.assert_array_equal(result, expected)


# ── Compositions ────────────────────────────────────────────────────────


@pytest.mark.reduction
def test_reduce_and_after_reduce_sum():
    """Threshold check after summation: reduce_and zeros out the sum deps."""
    shape = (2, 3)

    def f(x):
        s = jnp.sum(x.reshape(shape), axis=1)  # (2,) with deps
        mask = lax.reduce_and(s > 0, axes=(0,))  # scalar, zero deriv
        return mask * jnp.ones(1)

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    expected = np.zeros((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_and_size_one():
    """Reducing a single-element boolean array: zero Jacobian."""
    shape = (1,)

    def f(x):
        return lax.reduce_and(x > 0, axes=(0,)) * jnp.ones(1)

    result = jacobian_sparsity(f, input_shape=shape).todense().astype(int)
    expected = np.zeros((1, 1), dtype=int)
    np.testing.assert_array_equal(result, expected)

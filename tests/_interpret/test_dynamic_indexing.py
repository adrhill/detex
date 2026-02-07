"""Tests for dynamic_slice and dynamic_update_slice propagation."""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity

# =============================================================================
# dynamic_slice
# =============================================================================


@pytest.mark.array_ops
def test_dynamic_slice_static_start():
    """dynamic_slice with static start indices tracks precise dependencies.

    This is the pattern emitted by jnp.split.
    """

    def f(x):
        return lax.dynamic_slice(x, (1,), (3,))

    result = jacobian_sparsity(f, input_shape=5).todense().astype(int)
    # out = x[1:4], so out[0]←{1}, out[1]←{2}, out[2]←{3}
    expected = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_dynamic_slice_dynamic_start():
    """dynamic_slice with runtime-dependent start falls back to conservative."""

    def f(x):
        # Start index depends on input → dynamic
        idx = jnp.argmax(x[:2])
        return lax.dynamic_slice(x, (idx,), (3,))

    result = jacobian_sparsity(f, input_shape=5).todense().astype(int)
    expected = np.ones((3, 5), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_dynamic_slice_swap_halves():
    """Two dynamic_slice calls with static starts swap halves precisely."""

    def f(x):
        first_half = lax.dynamic_slice(x, (0,), (2,))
        second_half = lax.dynamic_slice(x, (2,), (2,))
        return jnp.concatenate([second_half, first_half])

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# dynamic_update_slice
# =============================================================================


@pytest.mark.array_ops
def test_dynamic_update_slice_static_start():
    """dynamic_update_slice with static start replaces a sub-region precisely."""

    def f(x):
        arr = jnp.zeros(5)
        return lax.dynamic_update_slice(arr, x[:2], (1,))

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    # out = [0, x[0], x[1], 0, 0]
    expected = np.array(
        [
            [0, 0, 0, 0],  # out[0] ← constant 0
            [1, 0, 0, 0],  # out[1] ← x[0]
            [0, 1, 0, 0],  # out[2] ← x[1]
            [0, 0, 0, 0],  # out[3] ← constant 0
            [0, 0, 0, 0],  # out[4] ← constant 0
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_dynamic_update_slice_dynamic_start():
    """dynamic_update_slice with runtime start falls back to conservative."""

    def f(x):
        idx = jnp.argmax(x[:2])
        return lax.dynamic_update_slice(x, x[2:4], (idx,))

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    # Conservative: all outputs depend on all inputs
    expected = np.ones((4, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)

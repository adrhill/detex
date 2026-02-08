"""Tests for reduction operations.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce.html
"""

import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.reduction
def test_reduce_max():
    """jnp.max (reduce_max) has correct global sparsity (all inputs matter).

    Unlike reduce_sum which has a handler, reduce_max falls to default.
    Both should produce the same result: output depends on all inputs.
    """

    def f(x):
        return jnp.array([jnp.max(x)])

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # All inputs can affect the max (global sparsity)
    expected = np.array([[1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_along_axis():
    """Reduction along one axis tracks per-row dependencies precisely."""

    def f(x):
        mat = x.reshape(2, 3)
        return jnp.sum(mat, axis=1)  # Sum each row

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    # out[0] = sum of row 0 = x[0]+x[1]+x[2], depends on inputs 0,1,2
    # out[1] = sum of row 1 = x[3]+x[4]+x[5], depends on inputs 3,4,5
    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_zero_size_input():
    """Zero-size input exercises empty union edge case."""

    def f(x):
        # Sum over empty array gives scalar 0 with no dependencies
        return jnp.sum(x)

    result = jacobian_sparsity(f, input_shape=0)
    assert result.shape == (1, 0)
    assert result.nnz == 0


@pytest.mark.reduction
def test_argmax():
    """argmax has zero derivative (returns integer index, not differentiable).

    Only x[0] contributes because argmax output has empty dependency sets.
    """

    def f(x):
        # argmax returns int, multiply by x[0] to get float output
        idx = jnp.argmax(x)
        return x[0] * idx.astype(float)

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.array([[1, 0, 0]])
    np.testing.assert_array_equal(result, expected)

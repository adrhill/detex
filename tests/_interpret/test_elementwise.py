"""Tests for elementwise operation propagation."""

import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.array_ops
def test_constant_in_elementwise_op():
    """Constant array in binary elementwise operation preserves input structure.

    Adding a constant array to input doesn't change the sparsity pattern.
    """

    def f(x):
        const = jnp.array([1.0, 2.0, 3.0])
        return x + const

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # Each output depends only on corresponding input (identity)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_zero_size_binary_elementwise():
    """Binary elementwise on size-0 arrays produces size-0 output."""

    def f(x):
        # Slicing to empty then adding exercises the size-0 binary path.
        a = x[:0]
        return a + a

    result = jacobian_sparsity(f, input_shape=3)
    assert result.shape == (0, 3)
    assert result.nnz == 0

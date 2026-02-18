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


@pytest.mark.elementwise
def test_binary_broadcast_size1_dim():
    """Binary ops with size-1 broadcasting map dependencies correctly.

    For mul of (3,4) * (3,1) → (3,4),
    out[i,j] depends on in1[i,j] and in2[i,0].
    The flat modular indexing ``i % len`` gives wrong results here
    because it maps ``(i*4 + j) % 3`` instead of projecting coordinates.
    """

    weights = jnp.ones((3, 1))

    def f(x):
        mat = x.reshape(3, 4)
        return (mat * weights).reshape(-1)

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    # Each output depends only on its own input (weights are constant).
    expected = np.eye(12, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_binary_broadcast_leading_dim():
    """Broadcasting along the leading dimension tracks dependencies per row.

    For mul of (4,3) * (1,3) → (4,3),
    out[i,j] depends on in1[i,j] and in2[0,j].
    """

    scale = jnp.ones((1, 3))

    def f(x):
        mat = x.reshape(4, 3)
        return (mat * scale).reshape(-1)

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    expected = np.eye(12, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_convert_element_type_propagates_const():
    """convert_element_type propagates const values for downstream gather.

    JAX inserts convert_element_type (int64 → int32) before gather.
    Without const propagation, the gather falls back to conservative.
    """

    indices = jnp.array([2, 0, 1])

    def f(x):
        return x[indices]

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # out[0] <- x[2], out[1] <- x[0], out[2] <- x[1]
    expected = np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)

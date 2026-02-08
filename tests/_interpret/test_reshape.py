"""Tests for reshape propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.reshape.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.array_ops
def test_reshape_with_dimensions_2d():
    """reshape with dimensions=(1,0) permutes a 2D array before flattening.

    ravel(order='F') on a (2, 3) matrix emits dimensions=(1, 0).
    Each output element still depends on exactly one input element.
    """

    def f(x):
        mat = x.reshape(2, 3)
        return mat.ravel(order="F")  # column-major: [a, d, b, e, c, f]

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    # Input flat: [a=0, b=1, c=2, d=3, e=4, f=5] in (2,3)
    # F-order ravel: [a, d, b, e, c, f] = [0, 3, 1, 4, 2, 5]
    expected = np.zeros((6, 6), dtype=int)
    expected[0, 0] = 1  # out[0] <- a
    expected[1, 3] = 1  # out[1] <- d
    expected[2, 1] = 1  # out[2] <- b
    expected[3, 4] = 1  # out[3] <- e
    expected[4, 2] = 1  # out[4] <- c
    expected[5, 5] = 1  # out[5] <- f
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_with_dimensions_3d():
    """reshape with dimensions=(2,1,0) permutes a 3D array before flattening.

    ravel(order='F') on a (2, 3, 4) tensor emits dimensions=(2, 1, 0).
    Verifies correct handling with higher-rank permutations.
    """

    def f(x):
        tensor = x.reshape(2, 3, 4)
        return tensor.ravel(order="F")

    result = jacobian_sparsity(f, input_shape=24).todense().astype(int)
    # Verify against actual Jacobian
    x_test = jax.random.normal(jax.random.key(42), (24,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)
    np.testing.assert_array_equal(result, actual_nonzero)


@pytest.mark.array_ops
def test_reshape_constant():
    """Reshaping a constant array produces zero sparsity."""

    def f(_x):
        const = jnp.array([1.0, 2.0, 3.0, 4.0])
        return const.reshape(2, 2).flatten()

    result = jacobian_sparsity(f, input_shape=2).todense().astype(int)
    expected = np.zeros((4, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_then_slice_constant():
    """Reshaping and slicing a constant produces zero sparsity."""

    def f(_x):
        const = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        mat = const.reshape(2, 3)
        return mat[0, :]  # First row

    result = jacobian_sparsity(f, input_shape=2).todense().astype(int)
    expected = np.zeros((3, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)

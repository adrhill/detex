"""Tests for pad primitive handler."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import hessian_sparsity, jacobian_sparsity


@pytest.mark.array_ops
def test_pad_1d_constant():
    """1D padding with constant values has no dependencies on padded positions."""

    def f(x):
        return jnp.pad(x, (1, 1), constant_values=0)

    result = jacobian_sparsity(f, input_shape=2).todense().astype(int)
    expected = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_1d_asymmetric():
    """Asymmetric padding: different amounts on each side."""

    def f(x):
        return jnp.pad(x, (2, 1), constant_values=0)

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # 2 pad + 3 original + 1 pad = 6 output elements
    expected = np.array(
        [
            [0, 0, 0],  # pad
            [0, 0, 0],  # pad
            [1, 0, 0],  # x[0]
            [0, 1, 0],  # x[1]
            [0, 0, 1],  # x[2]
            [0, 0, 0],  # pad
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_2d():
    """2D padding preserves per-element structure."""

    def f(x):
        mat = x.reshape(2, 2)
        return jnp.pad(mat, ((1, 0), (0, 1)), constant_values=0).flatten()

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    # Input: [[x0, x1], [x2, x3]]
    # Padded: [[0, 0, 0], [x0, x1, 0], [x2, x3, 0]]  (3x3 = 9 outputs)
    expected = np.array(
        [
            [0, 0, 0, 0],  # pad (0,0)
            [0, 0, 0, 0],  # pad (0,1)
            [0, 0, 0, 0],  # pad (0,2)
            [1, 0, 0, 0],  # x0  (1,0)
            [0, 1, 0, 0],  # x1  (1,1)
            [0, 0, 0, 0],  # pad (1,2)
            [0, 0, 1, 0],  # x2  (2,0)
            [0, 0, 0, 1],  # x3  (2,1)
            [0, 0, 0, 0],  # pad (2,2)
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_negative():
    """Negative padding trims the array (used by JAX's grad)."""

    def f(x):
        # Negative padding = trimming: removes first and last element
        return jax.lax.pad(x, 0.0, [(-1, -1, 0)])

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    # Removes first and last: output = [x1, x2]
    expected = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_interior():
    """Interior padding inserts elements between each pair of original elements."""

    def f(x):
        return jax.lax.pad(x, 0.0, [(0, 0, 1)])

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # x = [a, b, c] -> [a, 0, b, 0, c] (5 elements)
    expected = np.array(
        [
            [1, 0, 0],  # a
            [0, 0, 0],  # interior pad
            [0, 1, 0],  # b
            [0, 0, 0],  # interior pad
            [0, 0, 1],  # c
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_interior_with_border():
    """Interior and border padding combined."""

    def f(x):
        return jax.lax.pad(x, 0.0, [(1, 1, 2)])

    result = jacobian_sparsity(f, input_shape=2).todense().astype(int)
    # x = [a, b] -> [0, a, 0, 0, b, 0] (6 elements)
    # low=1, then a, then 2 interior pads, then b, then high=1
    expected = np.array(
        [
            [0, 0],  # low pad
            [1, 0],  # a
            [0, 0],  # interior
            [0, 0],  # interior
            [0, 1],  # b
            [0, 0],  # high pad
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_noop():
    """Zero padding on all sides is an identity operation."""

    def f(x):
        return jax.lax.pad(x, 0.0, [(0, 0, 0)])

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_negative_with_interior():
    """Negative border padding combined with interior padding.

    Interior padding dilates first,
    then negative low/high trims from edges.
    """

    def f(x):
        # Dilate [a, b, c] -> [a, 0, b, 0, c], then trim first element
        return jax.lax.pad(x, 0.0, [(-1, 0, 1)])

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # Dilated: [a, 0, b, 0, c] (5 elements), trim first -> [0, b, 0, c] (4 elements)
    expected = np.array(
        [
            [0, 0, 0],  # interior pad (was between a and b)
            [0, 1, 0],  # b
            [0, 0, 0],  # interior pad
            [0, 0, 1],  # c
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_value_with_deps():
    """Padding value that depends on inputs propagates to padding positions."""

    def f(x):
        # Pad with x[0] as the padding value — padding positions depend on x[0]
        return jax.lax.pad(x, x[0], [(1, 1, 0)])

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # Output: [x[0], x[0], x[1], x[2], x[0]]
    # Padding positions inherit x[0]'s dependency
    expected = np.array(
        [
            [1, 0, 0],  # pad = x[0]
            [1, 0, 0],  # x[0]
            [0, 1, 0],  # x[1]
            [0, 0, 1],  # x[2]
            [1, 0, 0],  # pad = x[0]
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_2d_interior():
    """2D interior padding dilates both dimensions."""

    def f(x):
        mat = x.reshape(2, 2)
        # Interior padding of 1 in both dims
        return jax.lax.pad(mat, 0.0, [(0, 0, 1), (0, 0, 1)]).flatten()

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    # Input: [[a, b], [c, d]]
    # Dilated: [[a, 0, b], [0, 0, 0], [c, 0, d]]  (3x3 = 9 elements)
    expected = np.array(
        [
            [1, 0, 0, 0],  # a
            [0, 0, 0, 0],  # interior
            [0, 1, 0, 0],  # b
            [0, 0, 0, 0],  # interior
            [0, 0, 0, 0],  # interior
            [0, 0, 0, 0],  # interior
            [0, 0, 1, 0],  # c
            [0, 0, 0, 0],  # interior
            [0, 0, 0, 1],  # d
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.hessian
def test_pad_in_hessian():
    """Pad enables precise tridiagonal Hessian for finite-difference stencils.

    This is the motivating use case:
    JAX's grad of sliced operations emits pad primitives.
    """

    def f(x):
        return ((x[1:] - x[:-1]) ** 2).sum()

    H = hessian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ]
    )
    np.testing.assert_array_equal(H, expected)

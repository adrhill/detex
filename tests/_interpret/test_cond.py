"""Tests for cond propagation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.control_flow
def test_cond_union_of_branches():
    """cond unions deps from different branches.

    One branch returns x[:2], the other returns x[1:3].
    The union gives each output deps from both branches.
    """

    def f(x):
        return jax.lax.cond(
            True,
            lambda operand: operand[:2],
            lambda operand: operand[1:3],
            x,
        )

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # Branch 0: out[0]←{0}, out[1]←{1}
    # Branch 1: out[0]←{1}, out[1]←{2}
    # Union:    out[0]←{0,1}, out[1]←{1,2}
    expected = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_cond_identical_branches():
    """cond with identical branches returns the same deps as either branch."""

    def f(x):
        return jax.lax.cond(
            True,
            lambda operand: operand * 2,
            lambda operand: operand * 3,
            x,
        )

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # Both branches are elementwise, so the union is still diagonal
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_cond_one_branch_constant():
    """cond where one branch returns a constant still unions both."""

    def f(x):
        return jax.lax.cond(
            True,
            lambda operand: operand,
            lambda operand: jnp.ones_like(operand),
            x,
        )

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # Branch 0: identity (diagonal).  Branch 1: constant (zeros).
    # Union is just the diagonal.
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)

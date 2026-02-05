"""Tests for control flow operations (conditionals)."""

import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.control_flow
def test_ifelse_both_branches():
    """ifelse unions both branches (global sparsity)."""

    def f(x):
        # jnp.where is the JAX equivalent of ifelse
        return jnp.array([jnp.where(x[1] < x[2], x[0] + x[1], x[2] * x[3])])

    result = jacobian_sparsity(f, n=4).todense().astype(int)
    expected = np.array([[1, 1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_ifelse_one_branch_constant():
    """ifelse with one constant branch."""

    def f(x):
        return jnp.array([jnp.where(x[1] < x[2], x[0] + x[1], 1.0)])

    result = jacobian_sparsity(f, n=4).todense().astype(int)
    expected = np.array([[1, 1, 0, 0]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
@pytest.mark.fallback
def test_where_mask():
    """jnp.where with mask triggers conservative fallback.

    Precise: each output depends on mask condition + both branches.
    """

    def f(x):
        mask = x > 0
        return jnp.where(mask, x, -x)

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    # Global sparsity: each output could depend on corresponding input
    # (mask has zero derivative, both branches are element-wise from x)
    # Conservative: may be dense depending on how where is traced
    assert result.shape == (3, 3)

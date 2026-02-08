"""Tests for select_n propagation (jnp.where, lax.select).

https://docs.jax.dev/en/latest/_autosummary/jax.lax.select_n.html
"""

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

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array([[1, 1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_ifelse_one_branch_constant():
    """ifelse with one constant branch."""

    def f(x):
        return jnp.array([jnp.where(x[1] < x[2], x[0] + x[1], 1.0)])

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array([[1, 1, 0, 0]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_where_mask():
    """jnp.where with array mask is element-wise.

    Each output depends only on the corresponding input
    since both branches are element-wise and the mask has zero derivative.
    """

    def f(x):
        mask = x > 0
        return jnp.where(mask, x, -x)

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_select_n_mixed_deps():
    """select_n where branches have different per-element dependencies."""

    def f(x):
        a = x[:3]
        b = jnp.array([x[3], x[4], x[3]])
        pred = jnp.array([True, False, True])
        return jnp.where(pred, a, b)

    result = jacobian_sparsity(f, input_shape=5).todense().astype(int)
    # out[0] ← {0} ∪ {3}, out[1] ← {1} ∪ {4}, out[2] ← {2} ∪ {3}
    expected = np.array([[1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 1, 1, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)

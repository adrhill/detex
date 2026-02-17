"""Tests for select_if_vmap propagation (Equinox vmapped cond).

Requires equinox to be installed.
"""

import numpy as np
import pytest

eqx = pytest.importorskip("equinox")
select_if_vmap_p = eqx.internal._loop.common.select_if_vmap_p

import jax.numpy as jnp  # noqa: E402

from asdex import jacobian_sparsity  # noqa: E402


@pytest.mark.control_flow
def test_select_if_vmap_both_branches():
    """select_if_vmap unions both branches (global sparsity)."""

    def f(x):
        pred = x[1] < x[2]
        on_true = x[0] + x[1]
        on_false = x[2] * x[3]
        return jnp.array([select_if_vmap_p.bind(pred, on_true, on_false)])

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array([[1, 1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_select_if_vmap_one_branch_constant():
    """select_if_vmap with one constant branch."""

    def f(x):
        pred = x[1] < x[2]
        on_true = x[0] + x[1]
        on_false = 1.0
        return jnp.array([select_if_vmap_p.bind(pred, on_true, on_false)])

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array([[1, 1, 0, 0]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_select_if_vmap_elementwise():
    """Vectorized select_if_vmap preserves element-wise sparsity."""

    def f(x):
        pred = x > 0
        return select_if_vmap_p.bind(pred, x, -x)

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)

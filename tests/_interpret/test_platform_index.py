"""Tests for the prop_platform_index handler.

``platform_index`` is used internally by ``jax.lax.platform_dependent``
and by extension ``jnp.diag`` and other platform-dispatched ops.
Since its output is a constant scalar,
it should not introduce any input dependencies.
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


# Direct platform_dependent tests
@pytest.mark.control_flow
def test_platform_dependent_elementwise():
    """platform_dependent with element-wise branches gives identity pattern."""

    def f(x):
        return lax.platform_dependent(x, default=lambda x: x * 2.0)

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_platform_dependent_scalar():
    """platform_dependent returning a scalar output."""

    def f(x):
        return lax.platform_dependent(x, default=lambda x: jnp.sum(x))

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.ones((1, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


# High-level ops that use platform_dependent
@pytest.mark.fallback
@pytest.mark.control_flow
def test_diag_1d():
    """jnp.diag on a 1D input uses platform_dependent internally.

    The sparsity is conservative because diag lowers to dynamic_update_slice,
    so each diagonal element depends on all inputs in the same column mod n.

    TODO(scatter): the true pattern is a 9x3 matrix with identity on the diagonal:
    out[0]<-{0}, out[4]<-{1}, out[8]<-{2}, all others zero.
    Resolving the dynamic_update_slice indices as constants would make this precise.
    """

    def f(x):
        return jnp.diag(x)

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.tile(np.eye(3, dtype=int), (3, 1))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.fallback
@pytest.mark.control_flow
def test_diag_2d():
    """jnp.diag on a 2D input extracts the diagonal via platform_dependent.

    The gather resolves the row index precisely but the column index
    exploits known iota values and zero-clearing in mul
    to produce the precise diagonal pattern.
    """

    def f(x):
        return jnp.diag(x.reshape(3, 3))

    result = jacobian_sparsity(f, input_shape=9).todense().astype(int)
    # out[i] = mat[i, i], so out[i] depends only on x[i*4]:
    # x[0] for out[0], x[4] for out[1], x[8] for out[2].
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)

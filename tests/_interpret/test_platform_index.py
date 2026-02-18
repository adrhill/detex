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
@pytest.mark.control_flow
def test_diag_1d():
    """jnp.diag on a 1D input uses platform_dependent internally.

    The sparsity is conservative because diag lowers to scatter,
    so each diagonal element depends on all inputs.
    """

    def f(x):
        return jnp.diag(x)

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # diag lowers to dynamic_update_slice which is conservative per-slice:
    # output element at flat index j depends on input j % n.
    expected = np.tile(np.eye(3, dtype=int), (3, 1))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.fallback
@pytest.mark.control_flow
def test_diag_2d():
    """jnp.diag on a 2D input extracts the diagonal via platform_dependent.

    The gather resolves the row index precisely but the column index
    is data-dependent (iota), so each output depends on its entire row.

    TODO(gather): the true pattern is diagonal — out[i] depends only on in[i*4].
    Resolving iota values as const_vals would make this precise.
    """

    def f(x):
        return jnp.diag(x.reshape(3, 3))

    result = jacobian_sparsity(f, input_shape=9).todense().astype(int)
    # out[i] depends on row i: elements {3i, 3i+1, 3i+2}
    expected = np.array(
        [
            [1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)

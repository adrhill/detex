"""Tests for while_loop propagation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.control_flow
def test_while_simple_accumulation():
    """while_loop that accumulates carry + const preserves carry dependencies.

    The body adds a constant to the carry,
    so output depends only on the initial carry (identity pattern).
    """

    def f(x):
        def body(carry):
            return carry + 1.0

        def cond(carry):
            return jnp.sum(carry) < 100.0

        return jax.lax.while_loop(cond, body, x)

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_while_dependency_spreading():
    """while_loop where the body mixes carry elements spreads dependencies.

    The body shifts elements cyclically,
    so after enough iterations all outputs depend on all inputs.
    """

    def f(x):
        def body(carry):
            return jnp.roll(carry, 1)

        def cond(carry):
            return jnp.sum(carry) < 100.0

        return jax.lax.while_loop(cond, body, x)

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # Rolling mixes all elements after enough iterations
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_while_immediate_convergence():
    """while_loop with identity body converges in one iteration."""

    def f(x):
        def body(carry):
            return carry

        def cond(carry):
            return jnp.sum(carry) < 100.0

        return jax.lax.while_loop(cond, body, x)

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)

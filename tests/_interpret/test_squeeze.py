"""Tests for squeeze propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.squeeze.html
"""

import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.array_ops
def test_squeeze_constant():
    """Squeezing a constant array produces zero sparsity."""

    def f(_x):
        const = jnp.array([[1.0, 2.0, 3.0]])  # Shape (1, 3)
        return jnp.squeeze(const, axis=0)  # Shape (3,)

    result = jacobian_sparsity(f, input_shape=2).todense().astype(int)
    expected = np.zeros((3, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)

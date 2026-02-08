"""Tests for gather propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.array_ops
def test_gather_fancy_indexing():
    """Fancy indexing (gather) with static indices tracks precise dependencies.

    Each output element depends on the corresponding indexed input.
    """

    def f(x):
        indices = jnp.array([2, 0, 1])
        return x[indices]

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # Permutation: out[0] <- in[2], out[1] <- in[0], out[2] <- in[1]
    expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_indices_through_select_n():
    """Gather stays precise when indices pass through select_n.

    JAX's negative-index normalization emits select_n between the literal
    indices and the gather.
    Const tracking through select_n keeps the indices statically known.
    """

    def f(x):
        indices = jnp.array([2, 0, 1])
        pred = indices < 0
        wrapped = indices + 3
        final_indices = lax.select(pred, wrapped, indices)
        return x[final_indices]

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # Permutation: out[0] <- in[2], out[1] <- in[0], out[2] <- in[1]
    expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_dynamic_indices_fallback():
    """Gather with dynamic (traced) indices uses conservative fallback.

    When indices depend on input,
    we cannot determine dependencies at trace time.
    """

    def f(x):
        # indices depend on x, so they're dynamic
        idx = jnp.argmax(x[:2])  # Dynamic index based on input
        indices = jnp.array([0, 1]) + idx
        return jnp.take(x, indices)

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    # Conservative: all outputs depend on all inputs
    expected = np.ones((2, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_2d_row_select():
    """2D gather selecting rows tracks per-row dependencies.

    Each output row depends only on the corresponding selected input row.
    """

    def f(x):
        mat = x.reshape(3, 2)
        indices = jnp.array([2, 0])  # Select rows 2 and 0
        return mat[indices].flatten()

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    # Output: row 2 (indices 4,5), then row 0 (indices 0,1)
    expected = np.array(
        [
            [0, 0, 0, 0, 1, 0],  # out[0] <- in[4]
            [0, 0, 0, 0, 0, 1],  # out[1] <- in[5]
            [1, 0, 0, 0, 0, 0],  # out[2] <- in[0]
            [0, 1, 0, 0, 0, 0],  # out[3] <- in[1]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)

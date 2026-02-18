"""Tests for scatter propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.array_ops
def test_scatter_at_set():
    """In-place update with .at[].set() tracks precise dependencies.

    Only the updated position depends on the update value.
    """

    def f(x):
        arr = jnp.zeros(3)
        return arr.at[1].set(x[0])

    result = jacobian_sparsity(f, input_shape=2).todense().astype(int)
    # Only index 1 depends on x[0], indices 0 and 2 are constant (zeros)
    expected = np.array([[0, 0], [1, 0], [0, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_add():
    """Scatter-add unions dependencies from operand and updates.

    Positions receiving updates depend on both the original value and the update.
    """

    def f(x):
        arr = jnp.array([1.0, 2.0, 3.0])
        return arr.at[1].add(x[0])  # arr[1] += x[0]

    result = jacobian_sparsity(f, input_shape=2).todense().astype(int)
    # Index 1 depends on x[0] (no dependency on arr since arr is constant)
    expected = np.array(
        [
            [0, 0],  # out[0] <- constant
            [1, 0],  # out[1] <- x[0]
            [0, 0],  # out[2] <- constant
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_multiple():
    """Scatter at multiple indices tracks each update separately."""

    def f(x):
        arr = jnp.zeros(4)
        return arr.at[jnp.array([0, 2])].set(x[:2])

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # out[0] <- x[0]
            [0, 0, 0],  # out[1] <- zeros (constant)
            [0, 1, 0],  # out[2] <- x[1]
            [0, 0, 0],  # out[3] <- zeros (constant)
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_dynamic_indices():
    """Scatter with dynamic (traced) indices uses conservative fallback.

    When scatter indices depend on input,
    we cannot determine targets at trace time.
    The conservative path unions operand and updates deps across all outputs.
    """

    def f(x):
        arr = x[:3]
        idx = jnp.argmax(x[3:]).astype(int)
        return arr.at[idx].set(x[3])

    result = jacobian_sparsity(f, input_shape=5).todense().astype(int)
    # Conservative: unions operand deps ({0},{1},{2}) and update dep ({3})
    # Index deps are empty (argmax has zero derivative)
    expected = np.array(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_2d():
    """2D scatter falls back to conservative.

    Multi-dimensional scatter patterns don't match the optimized 1D path.
    """

    def f(x):
        mat = jnp.zeros((2, 3))
        updates = x[:2].reshape(1, 2)
        return mat.at[0, :2].set(updates.flatten()).flatten()

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # At minimum, updated positions depend on x[0] and x[1]
    assert result[0, 0] == 1
    assert result[1, 1] == 1


@pytest.mark.array_ops
@pytest.mark.parametrize("method", ["mul", "min", "max"])
def test_scatter_combine(method):
    """Scatter combine variants union dependencies from operand and updates.

    Targeted positions depend on both the original value and the update.
    Non-targeted positions depend only on the operand.
    """

    def f(x):
        arr = x[:3]
        return getattr(arr.at[1], method)(x[3])

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0, 0],  # out[0] <- x[0]
            [0, 1, 0, 1],  # out[1] <- combine(x[1], x[3])
            [0, 0, 1, 0],  # out[2] <- x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_segment_sum():
    """segment_sum groups elements by segment ID.

    Each output depends on all inputs in the corresponding segment.
    """

    def f(x):
        segment_ids = jnp.array([0, 0, 1, 1, 1])
        return jax.ops.segment_sum(x, segment_ids, num_segments=2)

    result = jacobian_sparsity(f, input_shape=5).todense().astype(int)
    # Segment 0: inputs 0,1 -> output 0
    # Segment 1: inputs 2,3,4 -> output 1
    expected = np.array(
        [
            [1, 1, 0, 0, 0],  # out[0] <- x[0] + x[1]
            [0, 0, 1, 1, 1],  # out[1] <- x[2] + x[3] + x[4]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_2d_batched_dim0():
    """2D scatter-add along dim 0 tracks per-column dependencies precisely.

    The backward of ``features[indices]`` on 2D arrays produces scatter-add
    with ``update_window_dims=(1,)``, ``inserted_window_dims=(0,)``.
    Each update row targets an operand row,
    with trailing dimensions passed through element-wise.
    """
    indices = jnp.array([2, 0, 1])

    def f(x):
        mat = x.reshape(3, 4)
        gathered = mat[indices]  # [3, 4]: rows reordered
        return gathered.reshape(-1)

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    # gathered[0] = mat[2], gathered[1] = mat[0], gathered[2] = mat[1]
    # Each output element depends on exactly one input element.
    expected = np.zeros((12, 12), dtype=int)
    for i in range(3):
        for j in range(4):
            out_flat = i * 4 + j
            in_flat = indices[i] * 4 + j
            expected[out_flat, in_flat] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_set_middle_dim():
    """Scatter along a middle dimension tracks precise dependencies.

    ``arr.at[:, idx, :].set(value)`` writes a 2D slice at one position
    along dim 1. Non-target positions keep their original dependencies.
    """

    def f(x):
        arr = x.reshape(2, 3, 4)
        # Zero out the last neighbor slot (dim 1 = 2).
        arr = arr.at[:, 2, :].set(0.0)
        return arr.reshape(-1)

    result = jacobian_sparsity(f, input_shape=24).todense().astype(int)
    expected = np.zeros((24, 24), dtype=int)
    for i in range(24):
        # Flat index i corresponds to (a, n, h) = (i//12, (i//4)%3, i%4)
        n = (i // 4) % 3
        if n != 2:
            # Non-target positions keep identity dependency.
            expected[i, i] = 1
        # Target positions (n == 2) are set to constant 0, no dependencies.
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_add_middle_dim():
    """Scatter-add along a middle dimension unions operand and update deps.

    ``arr.at[:, idx, :].add(value)`` adds a 2D slice at one position
    along dim 1. Target positions depend on both operand and updates.
    """
    values = jnp.ones((2, 4))

    def f(x):
        arr = x.reshape(2, 3, 4)
        arr = arr.at[:, 1, :].add(values)
        return arr.reshape(-1)

    result = jacobian_sparsity(f, input_shape=24).todense().astype(int)
    # All positions keep identity since updates are constant.
    expected = np.eye(24, dtype=int)
    np.testing.assert_array_equal(result, expected)

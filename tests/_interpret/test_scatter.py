"""Tests for scatter propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity

# Existing basic tests


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


# Precision verification against jax.jacobian


def _check_precision(f, input_size: int) -> None:
    """Verify detected sparsity exactly matches actual Jacobian nonzeros.

    Computes the true Jacobian with ``jax.jacobian`` and asserts the
    detected pattern matches it element-wise.
    """
    x_test = jax.random.normal(jax.random.key(42), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)
    detected = jacobian_sparsity(f, input_shape=input_size).todense().astype(int)
    np.testing.assert_array_equal(
        detected, actual_nonzero, "Detected sparsity should match actual Jacobian"
    )


@pytest.mark.array_ops
def test_scatter_at_set_precision():
    """1D scatter verified against jax.jacobian.

    ``arr.at[[0,2]].set(x[:2])`` replaces positions 0 and 2.
    """

    def f(x):
        arr = x[:4]
        return arr.at[jnp.array([0, 2])].set(x[4:6])

    _check_precision(f, 6)


@pytest.mark.array_ops
def test_scatter_2d_batched_precision():
    """2D batched scatter (Pattern 1) verified against jax.jacobian.

    Reshapes into (3, 4) and replaces rows via fancy indexing.
    """
    indices = jnp.array([2, 0])

    def f(x):
        mat = x.reshape(3, 4)
        updates = jnp.ones((2, 4))
        return mat.at[indices].set(updates).reshape(-1)

    _check_precision(f, 12)


@pytest.mark.array_ops
def test_scatter_middle_dim_precision():
    """Middle-dim scatter (Pattern 2) verified against jax.jacobian.

    Sets ``arr.at[:, 1, :] = 0`` on a (2, 3, 4) array.
    """

    def f(x):
        arr = x.reshape(2, 3, 4)
        arr = arr.at[:, 1, :].set(0.0)
        return arr.reshape(-1)

    _check_precision(f, 24)


@pytest.mark.array_ops
def test_scatter_multi_index_precision():
    """Multi-index scatter (Pattern 3) verified against jax.jacobian.

    ``mat.at[rows, cols].set(vals)`` writes scalars at multi-dim coordinates.
    """

    def f(x):
        mat = x.reshape(3, 4)
        rows = jnp.array([0, 1, 2])
        cols = jnp.array([1, 3, 0])
        return mat.at[rows, cols].set(jnp.zeros(3)).reshape(-1)

    _check_precision(f, 12)


# Asymmetric (non-square) shapes


@pytest.mark.array_ops
def test_scatter_batched_nonsquare():
    """Batched scatter on a non-square (3, 5) operand.

    Replaces row 1 with constant updates,
    so row 1 loses input dependencies.
    """

    def f(x):
        mat = x.reshape(3, 5)
        updates = jnp.zeros((1, 5))
        return mat.at[jnp.array([1])].set(updates).reshape(-1)

    _check_precision(f, 15)


@pytest.mark.array_ops
def test_scatter_middle_dim_nonsquare():
    """Middle-dim scatter on a non-square (2, 3, 5) operand.

    Sets ``arr.at[:, 0, :] = 0`` to replace all entries along dim 1 = 0.
    """

    def f(x):
        arr = x.reshape(2, 3, 5)
        arr = arr.at[:, 0, :].set(0.0)
        return arr.reshape(-1)

    _check_precision(f, 30)


@pytest.mark.array_ops
def test_scatter_multi_index_nonsquare():
    """Multi-index scatter on a non-square (3, 5) operand.

    Writes three scalars at specific coordinates.
    """

    def f(x):
        mat = x.reshape(3, 5)
        rows = jnp.array([0, 2, 1])
        cols = jnp.array([4, 0, 2])
        return mat.at[rows, cols].set(jnp.zeros(3)).reshape(-1)

    _check_precision(f, 15)


# Conservative audit


def _check_sparser_than_conservative(f, input_size: int) -> None:
    """Verify the detected pattern has fewer nonzeros than the conservative bound.

    Conservative gives every output the union of all input deps,
    which is an upper bound.
    The precise handler should produce strictly fewer nonzeros.
    """
    detected = jacobian_sparsity(f, input_shape=input_size).todense().astype(int)
    out_size, in_size = detected.shape

    # Conservative upper bound: each output depends on union of all inputs.
    # The actual conservative count depends on which inputs are live,
    # but the densest possible is out_size * in_size.
    conservative_nnz = out_size * in_size
    detected_nnz = int(detected.sum())

    assert detected_nnz < conservative_nnz, (
        f"Precise handler ({detected_nnz} nnz) should be strictly sparser "
        f"than conservative ({conservative_nnz} nnz)"
    )


@pytest.mark.array_ops
def test_scatter_batched_sparser_than_conservative():
    """Batched scatter (Pattern 1) produces a sparser result than conservative."""

    def f(x):
        mat = x.reshape(3, 4)
        return mat.at[jnp.array([0, 2])].set(jnp.zeros((2, 4))).reshape(-1)

    _check_sparser_than_conservative(f, 12)


@pytest.mark.array_ops
def test_scatter_middle_dim_sparser_than_conservative():
    """Middle-dim scatter (Pattern 2) produces a sparser result than conservative."""

    def f(x):
        arr = x.reshape(2, 3, 4)
        arr = arr.at[:, 2, :].set(0.0)
        return arr.reshape(-1)

    _check_sparser_than_conservative(f, 24)


@pytest.mark.array_ops
def test_scatter_multi_index_sparser_than_conservative():
    """Multi-index scatter (Pattern 3) produces a sparser result than conservative."""

    def f(x):
        mat = x.reshape(3, 5)
        rows = jnp.array([0, 2])
        cols = jnp.array([1, 3])
        return mat.at[rows, cols].set(jnp.zeros(2)).reshape(-1)

    _check_sparser_than_conservative(f, 15)


# Composition


@pytest.mark.array_ops
def test_scatter_then_gather():
    """Scatter followed by gather: only the gathered positions matter.

    Sets position 1 to zero, then gathers positions [0, 2].
    The final output should depend only on x[0] and x[2].
    """

    def f(x):
        arr = x.at[1].set(0.0)
        return arr[jnp.array([0, 2])]

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # out[0] <- x[0]
            [0, 0, 1],  # out[1] <- x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_with_const_chain_indices():
    """Scatter with indices derived from broadcast and reshape.

    Indices are built through a const-propagation chain
    (literal -> broadcast -> reshape) and should still be resolved statically.
    """

    def f(x):
        arr = x[:4]
        # Build indices through broadcast + reshape
        base = jnp.array([1])
        broadcasted = jnp.broadcast_to(base, (2,))
        idx = broadcasted + jnp.array([0, 1])  # [1, 2]
        return arr.at[idx].set(jnp.zeros(2))

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    # Positions 1 and 2 replaced with constant zeros.
    expected = np.array(
        [
            [1, 0, 0, 0],  # out[0] <- x[0]
            [0, 0, 0, 0],  # out[1] <- constant 0
            [0, 0, 0, 0],  # out[2] <- constant 0
            [0, 0, 0, 1],  # out[3] <- x[3]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_after_reshape():
    """Scatter into a reshaped array preserves per-element tracking."""

    def f(x):
        mat = x.reshape(2, 3)
        # Replace row 0 with zeros
        mat = mat.at[0].set(jnp.zeros(3))
        return mat.reshape(-1)

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    # Row 0 (positions 0-2) replaced with constant 0, row 1 (3-5) kept.
    expected = np.zeros((6, 6), dtype=int)
    for i in range(3, 6):
        expected[i, i] = 1
    np.testing.assert_array_equal(result, expected)


# Edge cases


@pytest.mark.array_ops
def test_scatter_duplicate_indices_set():
    """Duplicate indices with set: last write wins.

    When two updates target the same position,
    only the last update's dependencies survive (replace semantics).
    """

    def f(x):
        arr = jnp.zeros(3)
        # Both x[0] and x[1] target position 1; x[1] wins.
        return arr.at[jnp.array([1, 1])].set(x[:2])

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 0],  # out[0] <- constant 0
            [0, 1, 0],  # out[1] <- x[1] (last write wins)
            [0, 0, 0],  # out[2] <- constant 0
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_duplicate_indices_add():
    """Duplicate indices with add: both updates contribute.

    When two updates target the same position,
    the output depends on both (union semantics).
    """

    def f(x):
        arr = jnp.zeros(3)
        # Both x[0] and x[1] are added to position 1.
        return arr.at[jnp.array([1, 1])].add(x[:2])

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 0],  # out[0] <- constant 0
            [1, 1, 0],  # out[1] <- x[0] + x[1]
            [0, 0, 0],  # out[2] <- constant 0
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_oob_indices():
    """Out-of-bounds indices are clipped by JAX.

    JAX clips OOB scatter indices to valid range.
    The handler should match this behavior.
    """

    def f(x):
        arr = x[:3]
        # Index 10 is out of bounds for size 3; JAX clips to 2.
        return arr.at[jnp.array([10])].set(jnp.array([0.0]))

    _check_precision(f, 3)


@pytest.mark.array_ops
def test_scatter_replace_all():
    """Scatter that replaces every position: output depends only on updates."""

    def f(x):
        arr = x[:3]
        return arr.at[jnp.array([0, 1, 2])].set(x[3:6])

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    # Every position is replaced by the corresponding update.
    expected = np.array(
        [
            [0, 0, 0, 1, 0, 0],  # out[0] <- x[3]
            [0, 0, 0, 0, 1, 0],  # out[1] <- x[4]
            [0, 0, 0, 0, 0, 1],  # out[2] <- x[5]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_multi_index_duplicate_set():
    """Multi-index scatter with duplicate coordinates: last write wins.

    Two updates target ``(0, 1)``; the second update's dep survives.
    """

    def f(x):
        mat = x.reshape(2, 3)
        rows = jnp.array([0, 0])
        cols = jnp.array([1, 1])
        vals = x[:2]  # x[0] and x[1] both target (0, 1)
        return mat.at[rows, cols].set(vals).reshape(-1)

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    # Position (0,1) = flat 1 gets x[1] (last write wins).
    # All other positions keep their original identity deps.
    expected = np.eye(6, dtype=int)
    expected[1, :] = 0
    expected[1, 1] = 1  # out[1] <- x[1]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_multi_index_duplicate_add():
    """Multi-index scatter-add with duplicate coordinates: union of both.

    Two updates target ``(0, 1)``; the output unions both deps.
    """

    def f(x):
        mat = jnp.zeros((2, 3))
        rows = jnp.array([0, 0])
        cols = jnp.array([1, 1])
        vals = x[:2]
        return mat.at[rows, cols].add(vals).reshape(-1)

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # Position (0,1) = flat 1 gets x[0] + x[1].
    expected = np.array(
        [
            [0, 0, 0],  # out[0] <- constant 0
            [1, 1, 0],  # out[1] <- x[0] + x[1]
            [0, 0, 0],  # out[2] <- constant 0
            [0, 0, 0],  # out[3] <- constant 0
            [0, 0, 0],  # out[4] <- constant 0
            [0, 0, 0],  # out[5] <- constant 0
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_multi_index_oob():
    """Multi-index scatter with out-of-bounds coordinates.

    JAX clips OOB indices; the handler should match.
    """

    def f(x):
        mat = x.reshape(2, 3)
        # (5, 10) is out of bounds for (2, 3); JAX clips to (1, 2).
        rows = jnp.array([5])
        cols = jnp.array([10])
        return mat.at[rows, cols].set(jnp.zeros(1)).reshape(-1)

    _check_precision(f, 6)


@pytest.mark.fallback
@pytest.mark.array_ops
def test_scatter_2d():
    """2D partial-row scatter ``mat.at[0, :2].set(updates)`` falls back to conservative.

    TODO(scatter): ``mat.at[0, :2].set(updates)`` could track precise
    per-element dependencies, but the current handler unions operand
    and update deps across all outputs.
    """

    def f(x):
        mat = jnp.zeros((2, 3))
        updates = x[:2].reshape(1, 2)
        return mat.at[0, :2].set(updates.flatten()).flatten()

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # Conservative: all outputs depend on all update inputs.
    # TODO(scatter): the true pattern is:
    #   [[1, 0, 0],   out[0] = mat[0,0] <- x[0]
    #    [0, 1, 0],   out[1] = mat[0,1] <- x[1]
    #    [0, 0, 0],   out[2] = mat[0,2] <- constant 0
    #    [0, 0, 0],   out[3] = mat[1,0] <- constant 0
    #    [0, 0, 0],   out[4] = mat[1,1] <- constant 0
    #    [0, 0, 0]]   out[5] = mat[1,2] <- constant 0
    expected = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)

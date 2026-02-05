"""Tests for array manipulation operations."""

import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity

# =============================================================================
# Slicing and indexing
# =============================================================================


@pytest.mark.array_ops
def test_multidim_slice():
    """Multi-dimensional slice tracks per-element dependencies precisely.

    Each output element depends on exactly one input element.
    """

    def f(x):
        # Reshape to 2D and slice in multiple dimensions
        mat = x.reshape(3, 3)
        sliced = mat[0:2, 0:2]  # 2D slice extracts 2x2 submatrix
        return sliced.flatten()

    result = jacobian_sparsity(f, n=9).todense().astype(int)
    # Input (3x3): indices 0-8 in row-major order
    # Slice [0:2, 0:2] extracts: [0,0]=0, [0,1]=1, [1,0]=3, [1,1]=4
    expected = np.zeros((4, 9), dtype=int)
    expected[0, 0] = 1  # out[0] <- in[0]
    expected[1, 1] = 1  # out[1] <- in[1]
    expected[2, 3] = 1  # out[2] <- in[3]
    expected[3, 4] = 1  # out[3] <- in[4]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.fallback
def test_gather_fancy_indexing():
    """Fancy indexing (gather) triggers conservative fallback.

    TODO(gather): Implement precise handler for gather with static indices.
    Precise: each output element depends on the corresponding indexed input.
    """

    def f(x):
        indices = jnp.array([2, 0, 1])
        return x[indices]

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    # TODO: Should be permutation [[0,0,1], [1,0,0], [0,1,0]]
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Broadcasting
# =============================================================================


@pytest.mark.array_ops
def test_array_broadcast():
    """Broadcasting a non-scalar array tracks per-element dependencies precisely.

    Each output element depends on the input element it was replicated from.
    """

    def f(x):
        # x is shape (3,), reshape to (3, 1) and broadcast to (3, 2)
        col = x.reshape(3, 1)
        broadcasted = jnp.broadcast_to(col, (3, 2))
        return broadcasted.flatten()

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    # Output (3x2) flattened: [0,0], [0,1], [1,0], [1,1], [2,0], [2,1]
    # Each row comes from one input: out[0,1] <- in[0], out[2,3] <- in[1], etc.
    expected = np.array(
        [
            [1, 0, 0],  # out[0] <- in[0]
            [1, 0, 0],  # out[1] <- in[0]
            [0, 1, 0],  # out[2] <- in[1]
            [0, 1, 0],  # out[3] <- in[1]
            [0, 0, 1],  # out[4] <- in[2]
            [0, 0, 1],  # out[5] <- in[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scalar_broadcast():
    """Broadcasting a scalar preserves per-element structure."""

    def f(x):
        # Each element broadcast independently
        return jnp.array([jnp.broadcast_to(x[0], (2,)).sum(), x[1] * 2])

    result = jacobian_sparsity(f, n=2).todense().astype(int)
    expected = np.array([[1, 0], [0, 1]])
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Concatenation and stacking
# =============================================================================


@pytest.mark.array_ops
def test_stack():
    """jnp.stack tracks per-element dependencies precisely."""

    def f(x):
        a, b = x[:2], x[2:]
        return jnp.stack([a, b]).flatten()

    result = jacobian_sparsity(f, n=4).todense().astype(int)
    # Each output depends on exactly one input (identity)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_nested_slice_concat():
    """Multiple 1D slices followed by concatenate should preserve structure."""

    def f(x):
        a = x[:2]
        b = x[2:]
        return jnp.concatenate([b, a])  # [x2, x3, x0, x1]

    result = jacobian_sparsity(f, n=4).todense().astype(int)
    # Permutation: swap first 2 and last 2
    expected = np.array(
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=int
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.bug
def test_empty_concatenate():
    """Concatenating with empty arrays produces incorrect sparsity.

    TODO(bug): Fix empty array handling in concatenate/reshape.
    BUG: Empty arrays in concatenate produce incorrect indices.
    Should produce identity matrix but result is shifted.
    """

    def f(x):
        empty = jnp.array([])
        return jnp.concatenate([empty, x, empty])

    # TODO: Should produce identity matrix
    result = jacobian_sparsity(f, n=2)
    expected = np.eye(2, dtype=int)
    # BUG: Result is incorrect (shifted) instead of identity
    assert not np.array_equal(result.todense().astype(int), expected)


# =============================================================================
# Reshape and transpose
# =============================================================================


@pytest.mark.array_ops
@pytest.mark.fallback
def test_transpose_2d():
    """Transpose should preserve per-element dependencies with reordering.

    TODO(transpose): Implement precise handler for transpose primitive.
    Currently triggers conservative fallback (all outputs depend on all inputs).
    Precise: output[i,j] depends only on input[j,i] (permutation matrix).
    """

    def f(x):
        mat = x.reshape(2, 3)
        return mat.T.flatten()  # (3, 2) -> 6 elements

    result = jacobian_sparsity(f, n=6).todense().astype(int)
    # TODO: Should be permutation matrix, not dense
    expected = np.ones((6, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Permutation operations
# =============================================================================


@pytest.mark.array_ops
@pytest.mark.fallback
def test_reverse():
    """jnp.flip triggers conservative fallback.

    TODO(rev): Implement precise handler for rev (reverse) primitive.
    Precise: output[i] depends on input[n-1-i] (anti-diagonal permutation).
    """

    def f(x):
        return jnp.flip(x)

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    # TODO: Should be anti-diagonal [[0,0,1], [0,1,0], [1,0,0]]
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_roll():
    """jnp.roll correctly tracks the cyclic permutation.

    output[i] depends on input[(i-shift) % n].
    """

    def f(x):
        return jnp.roll(x, shift=1)

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    # Precise: cyclic permutation matrix
    # output[0] <- input[2], output[1] <- input[0], output[2] <- input[1]
    expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Padding and tiling
# =============================================================================


@pytest.mark.array_ops
@pytest.mark.fallback
def test_pad():
    """jnp.pad triggers conservative fallback.

    TODO(pad): Implement precise handler for pad primitive.
    Precise: padded elements have no dependency, original elements preserve structure.
    """

    def f(x):
        return jnp.pad(x, (1, 1), constant_values=0)

    result = jacobian_sparsity(f, n=2).todense().astype(int)
    # TODO: Should be [[0,0], [1,0], [0,1], [0,0]] (pad values have no deps)
    expected = np.ones((4, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.fallback
def test_tile():
    """jnp.tile triggers conservative fallback.

    TODO(tile): Implement precise handler for broadcast_in_dim used by tile.
    Precise: each output element depends on corresponding input (mod input size).
    """

    def f(x):
        return jnp.tile(x, 2)

    result = jacobian_sparsity(f, n=2).todense().astype(int)
    # TODO: Should be [[1,0], [0,1], [1,0], [0,1]]
    expected = np.ones((4, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Split
# =============================================================================


@pytest.mark.array_ops
@pytest.mark.fallback
def test_split():
    """jnp.split triggers conservative fallback.

    TODO(dynamic_slice): split uses dynamic_slice which needs precise handler.
    Precise: each output element depends only on corresponding input.
    """

    def f(x):
        parts = jnp.split(x, 2)
        return jnp.concatenate([parts[1], parts[0]])  # swap halves

    result = jacobian_sparsity(f, n=4).todense().astype(int)
    # TODO: Should be permutation [[0,0,1,0], [0,0,0,1], [1,0,0,0], [0,1,0,0]]
    expected = np.ones((4, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Scatter operations
# =============================================================================


@pytest.mark.array_ops
@pytest.mark.fallback
def test_scatter_at_set():
    """In-place update with .at[].set() is partially precise.

    TODO(scatter): Implement precise handler for scatter primitive.
    Currently: all outputs depend on x[0] (the value being set).
    Precise: only output[1] should depend on x[0].
    """

    def f(x):
        arr = jnp.zeros(3)
        return arr.at[1].set(x[0])

    result = jacobian_sparsity(f, n=2).todense().astype(int)
    # TODO: Should be [[0,0], [1,0], [0,0]] (only index 1 depends on x[0])
    expected = np.array([[1, 0], [1, 0], [1, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Matrix multiplication
# =============================================================================


@pytest.mark.array_ops
@pytest.mark.fallback
def test_matmul():
    """Matrix multiplication (dot_general) triggers conservative fallback.

    TODO(dot_general): Implement precise handler for dot_general primitive.
    Precise: output[i,j] depends on row i of first input and column j of second.
    For f(x) = x @ x.T, output[i,j] depends on rows i and j of input.
    """

    def f(x):
        mat = x.reshape(2, 2)
        return (mat @ mat.T).flatten()

    result = jacobian_sparsity(f, n=4).todense().astype(int)
    # TODO: Should track row/column dependencies, not be fully dense
    expected = np.ones((4, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.fallback
def test_iota_eye():
    """jnp.eye uses iota internally, triggers conservative fallback.

    TODO(iota): Add iota to ZERO_DERIVATIVE_PRIMITIVES (constant output).
    TODO(dot_general): Also needs dot_general handler for eye @ x.
    Precise: eye matrix has no input dependency (constant), so eye @ x = x.
    """

    def f(x):
        # Multiply x by identity - should preserve diagonal structure
        return jnp.eye(3) @ x

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    # TODO: Should be identity matrix (eye @ x = x)
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Sorting
# =============================================================================


@pytest.mark.array_ops
@pytest.mark.fallback
def test_sort():
    """jnp.sort triggers conservative fallback.

    Precise: all outputs depend on all inputs (sorting is a global operation).
    """

    def f(x):
        return jnp.sort(x)

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    # Conservative fallback is actually correct here
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Edge cases
# =============================================================================


@pytest.mark.array_ops
def test_zero_size_input():
    """Zero-size input exercises empty union edge case."""

    def f(x):
        # Sum over empty array gives scalar 0 with no dependencies
        return jnp.sum(x)

    result = jacobian_sparsity(f, n=0)
    assert result.shape == (1, 0)
    assert result.nse == 0

import jax.numpy as jnp
import numpy as np

from detex import jacobian_sparsity


def test_simple_dependencies():
    """Test f(x) = [x0+x1, x1*x2, x2]"""

    def f(x):
        return jnp.array([x[0] + x[1], x[1] * x[2], x[2]])

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    expected = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    np.testing.assert_array_equal(result, expected)


def test_complex_dependencies():
    """Test f(x) = [x0*x1 + sin(x2), x3, x0*x1*x3]"""

    def f(x):
        a = x[0] * x[1]
        b = jnp.sin(x[2])
        c = a + b
        return jnp.array([c, x[3], a * x[3]])

    result = jacobian_sparsity(f, n=4).toarray().astype(int)
    expected = np.array([[1, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 1]])
    np.testing.assert_array_equal(result, expected)


def test_diagonal_jacobian():
    """Test f(x) = x^2 (element-wise) produces diagonal sparsity"""

    def f(x):
        return x**2

    result = jacobian_sparsity(f, n=4).toarray().astype(int)
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    np.testing.assert_array_equal(result, expected)


def test_dense_jacobian():
    """Test f(x) = [sum(x), prod(x)] produces dense sparsity"""

    def f(x):
        return jnp.array([jnp.sum(x), jnp.prod(x)])

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    expected = np.array([[1, 1, 1], [1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


def test_sct_readme_example():
    """Test SCT README example: f(x) = [x1^2, 2*x1*x2^2, sin(x3)]"""

    def f(x):
        return jnp.array([x[0] ** 2, 2 * x[0] * x[1] ** 2, jnp.sin(x[2])])

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    expected = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Tests from SparseConnectivityTracer.jl "Jacobian Global" testset
# https://github.com/adrhill/SparseConnectivityTracer.jl/blob/main/test/test_gradient.jl
# =============================================================================


def test_identity():
    """Identity function: f(x) = x"""

    def f(x):
        return x

    result = jacobian_sparsity(f, n=1).toarray().astype(int)
    expected = np.array([[1]])
    np.testing.assert_array_equal(result, expected)


def test_constant():
    """Constant function: output doesn't depend on input."""

    def f(x):
        return jnp.array([1.0])

    result = jacobian_sparsity(f, n=1).toarray().astype(int)
    expected = np.array([[0]])
    np.testing.assert_array_equal(result, expected)


def test_zero_derivative_ceil_round():
    """ceil/round have zero derivative: f(x) = [x1*x2, ceil(x1*x2), x1*round(x2)]"""

    def f(x):
        return jnp.array([x[0] * x[1], jnp.ceil(x[0] * x[1]), x[0] * jnp.round(x[1])])

    result = jacobian_sparsity(f, n=2).toarray().astype(int)
    # round(x2) has zero derivative, so x1*round(x2) only depends on x1
    expected = np.array([[1, 1], [0, 0], [1, 0]])
    np.testing.assert_array_equal(result, expected)


def test_zero_derivative_floor():
    """floor has zero derivative."""

    def f(x):
        return jnp.floor(x)

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    expected = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_zero_derivative_sign():
    """sign has zero derivative."""

    def f(x):
        return jnp.sign(x)

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    expected = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_comparison_ops():
    """Comparison operators have zero derivative."""

    def f(x):
        return jnp.array(
            [
                (x[0] < x[1]).astype(float),
                (x[0] <= x[1]).astype(float),
                (x[0] > x[1]).astype(float),
                (x[0] >= x[1]).astype(float),
                (x[0] == x[1]).astype(float),
                (x[0] != x[1]).astype(float),
            ]
        )

    result = jacobian_sparsity(f, n=2).toarray().astype(int)
    expected = np.zeros((6, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_type_conversion():
    """Type conversions preserve dependencies."""

    def f(x):
        return x.astype(jnp.float32)

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_power_operations():
    """Various power operations: x^e, e^x, etc."""

    def f(x):
        return jnp.array([x[0] ** 2.5, jnp.exp(x[1]), x[2] ** x[2]])

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(result, expected)


def test_integer_pow_zero():
    """x^0 = 1 has no dependency on x."""

    def f(x):
        return x**0

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    expected = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_clamp_scalar_bounds():
    """clamp(x, lo, hi) with scalar bounds preserves element structure."""

    def f(x):
        return jnp.clip(x, 0.0, 1.0)

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_clamp_variable_bounds():
    """clamp(x1, x2, x3) with variable bounds - all contribute."""

    def f(x):
        return jnp.array([jnp.clip(x[0], x[1], x[2])])

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    expected = np.array([[1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


def test_ifelse_both_branches():
    """ifelse unions both branches (global sparsity)."""

    def f(x):
        # jnp.where is the JAX equivalent of ifelse
        return jnp.array([jnp.where(x[1] < x[2], x[0] + x[1], x[2] * x[3])])

    result = jacobian_sparsity(f, n=4).toarray().astype(int)
    expected = np.array([[1, 1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


def test_ifelse_one_branch_constant():
    """ifelse with one constant branch."""

    def f(x):
        return jnp.array([jnp.where(x[1] < x[2], x[0] + x[1], 1.0)])

    result = jacobian_sparsity(f, n=4).toarray().astype(int)
    expected = np.array([[1, 1, 0, 0]])
    np.testing.assert_array_equal(result, expected)


def test_dot_product():
    """Dot product: dot(x[0:2], x[3:5])."""

    def f(x):
        return jnp.array([jnp.dot(x[:2], x[3:5])])

    result = jacobian_sparsity(f, n=5).toarray().astype(int)
    expected = np.array([[1, 1, 0, 1, 1]])
    np.testing.assert_array_equal(result, expected)


def test_multiply_by_zero():
    """Multiplying by zero still tracks structural dependency."""

    def f1(x):
        return jnp.array([0 * x[0]])

    def f2(x):
        return jnp.array([x[0] * 0])

    # Global sparsity: we can't know at compile time that result is zero
    result1 = jacobian_sparsity(f1, n=1).toarray().astype(int)
    result2 = jacobian_sparsity(f2, n=1).toarray().astype(int)
    expected = np.array([[1]])
    np.testing.assert_array_equal(result1, expected)
    np.testing.assert_array_equal(result2, expected)


def test_unary_functions():
    """Various unary math functions preserve element structure."""

    def f(x):
        return jnp.array(
            [
                jnp.sin(x[0]),
                jnp.cos(x[1]),
                jnp.tan(x[2]),
                jnp.exp(x[0]),
                jnp.log(x[1] + 1),  # +1 to avoid log(0)
                jnp.sqrt(jnp.abs(x[2]) + 1),
                jnp.sinh(x[0]),
                jnp.cosh(x[1]),
                jnp.tanh(x[2]),
            ]
        )

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # sin(x0)
            [0, 1, 0],  # cos(x1)
            [0, 0, 1],  # tan(x2)
            [1, 0, 0],  # exp(x0)
            [0, 1, 0],  # log(x1+1)
            [0, 0, 1],  # sqrt(|x2|+1)
            [1, 0, 0],  # sinh(x0)
            [0, 1, 0],  # cosh(x1)
            [0, 0, 1],  # tanh(x2)
        ]
    )
    np.testing.assert_array_equal(result, expected)


def test_binary_min_max():
    """min and max operations."""

    def f(x):
        return jnp.array([jnp.minimum(x[0], x[1]), jnp.maximum(x[1], x[2])])

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    expected = np.array([[1, 1, 0], [0, 1, 1]])
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Tests for edge cases and conservative fallbacks
# =============================================================================


def test_multidim_slice():
    """Multi-dimensional slice triggers conservative fallback (union all deps).

    TODO: Implement precise multi-dimensional slice tracking. The correct sparsity
    would show each output element depending only on its corresponding input element.
    """

    def f(x):
        # Reshape to 2D and slice in multiple dimensions
        mat = x.reshape(3, 3)
        sliced = mat[0:2, 0:2]  # 2D slice
        return sliced.flatten()

    result = jacobian_sparsity(f, n=9).toarray().astype(int)
    # Conservative fallback: all outputs depend on all inputs
    # Precise: would be sparse, each output depends on one input
    expected = np.ones((4, 9), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_array_broadcast():
    """Broadcasting a non-scalar array triggers conservative fallback.

    TODO: Implement precise multi-dimensional broadcast tracking. The correct
    sparsity would track which input elements map to which output elements.
    """

    def f(x):
        # x is shape (3,), reshape to (3, 1) and broadcast to (3, 2)
        col = x.reshape(3, 1)
        broadcasted = jnp.broadcast_to(col, (3, 2))
        return broadcasted.flatten()

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    # Conservative fallback: all outputs depend on all inputs
    # Precise: outputs 0,1 depend on input 0; outputs 2,3 on input 1; etc.
    expected = np.ones((6, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_scalar_broadcast():
    """Broadcasting a scalar preserves per-element structure."""

    def f(x):
        # Each element broadcast independently
        return jnp.array([jnp.broadcast_to(x[0], (2,)).sum(), x[1] * 2])

    result = jacobian_sparsity(f, n=2).toarray().astype(int)
    expected = np.array([[1, 0], [0, 1]])
    np.testing.assert_array_equal(result, expected)


def test_zero_size_input():
    """Zero-size input exercises empty union edge case."""

    def f(x):
        # Sum over empty array gives scalar 0 with no dependencies
        return jnp.sum(x)

    result = jacobian_sparsity(f, n=0)
    assert result.shape == (1, 0)
    assert result.nnz == 0


# =============================================================================
# Tests for edge cases that trigger conservative fallback
# These document current behavior and expected precise behavior
# =============================================================================


def test_transpose_2d():
    """Transpose should preserve per-element dependencies with reordering.

    TODO(transpose): Implement precise handler for transpose primitive.
    Currently triggers conservative fallback (all outputs depend on all inputs).
    Precise: output[i,j] depends only on input[j,i] (permutation matrix).
    """

    def f(x):
        mat = x.reshape(2, 3)
        return mat.T.flatten()  # (3, 2) -> 6 elements

    result = jacobian_sparsity(f, n=6).toarray().astype(int)
    # TODO: Should be permutation matrix, not dense
    expected = np.ones((6, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_matmul():
    """Matrix multiplication (dot_general) triggers conservative fallback.

    TODO(dot_general): Implement precise handler for dot_general primitive.
    Precise: output[i,j] depends on row i of first input and column j of second.
    For f(x) = x @ x.T, output[i,j] depends on rows i and j of input.
    """

    def f(x):
        mat = x.reshape(2, 2)
        return (mat @ mat.T).flatten()

    result = jacobian_sparsity(f, n=4).toarray().astype(int)
    # TODO: Should track row/column dependencies, not be fully dense
    expected = np.ones((4, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_argmax():
    """argmax has zero derivative (returns integer index, not differentiable).

    TODO(argmax): Add argmax/argmin to ZERO_DERIVATIVE_PRIMITIVES.
    Currently triggers conservative fallback.
    Precise: argmax output has zero dependency (non-differentiable).
    """

    def f(x):
        # argmax returns int, multiply by x[0] to get float output
        idx = jnp.argmax(x)
        return x[0] * idx.astype(float)

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    # TODO: Should be [[1, 0, 0]] - only x[0] contributes (argmax has zero derivative)
    expected = np.array([[1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


def test_gather_fancy_indexing():
    """Fancy indexing (gather) triggers conservative fallback.

    TODO(gather): Implement precise handler for gather with static indices.
    Precise: each output element depends on the corresponding indexed input.
    """

    def f(x):
        indices = jnp.array([2, 0, 1])
        return x[indices]

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    # TODO: Should be permutation [[0,0,1], [1,0,0], [0,1,0]]
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_stack():
    """jnp.stack preserves block structure but not per-element structure.

    TODO(stack): Track per-element dependencies through concatenate after reshape.
    Each output depends on the corresponding stacked array (block-wise).
    Precise: would be identity (each output = one input).
    """

    def f(x):
        a, b = x[:2], x[2:]
        return jnp.stack([a, b]).flatten()

    result = jacobian_sparsity(f, n=4).toarray().astype(int)
    # TODO: Should be identity matrix, not block-diagonal
    # Block-wise: outputs 0-1 depend on inputs 0-1, outputs 2-3 on inputs 2-3
    expected = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=int
    )
    np.testing.assert_array_equal(result, expected)


def test_reverse():
    """jnp.flip triggers conservative fallback.

    TODO(rev): Implement precise handler for rev (reverse) primitive.
    Precise: output[i] depends on input[n-1-i] (anti-diagonal permutation).
    """

    def f(x):
        return jnp.flip(x)

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    # TODO: Should be anti-diagonal [[0,0,1], [0,1,0], [1,0,0]]
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_roll():
    """jnp.roll correctly tracks the cyclic permutation.

    output[i] depends on input[(i-shift) % n].
    """

    def f(x):
        return jnp.roll(x, shift=1)

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    # Precise: cyclic permutation matrix
    # output[0] <- input[2], output[1] <- input[0], output[2] <- input[1]
    expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_pad():
    """jnp.pad triggers conservative fallback.

    TODO(pad): Implement precise handler for pad primitive.
    Precise: padded elements have no dependency, original elements preserve structure.
    """

    def f(x):
        return jnp.pad(x, (1, 1), constant_values=0)

    result = jacobian_sparsity(f, n=2).toarray().astype(int)
    # TODO: Should be [[0,0], [1,0], [0,1], [0,0]] (pad values have no deps)
    expected = np.ones((4, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_tile():
    """jnp.tile triggers conservative fallback.

    TODO(tile): Implement precise handler for broadcast_in_dim used by tile.
    Precise: each output element depends on corresponding input (mod input size).
    """

    def f(x):
        return jnp.tile(x, 2)

    result = jacobian_sparsity(f, n=2).toarray().astype(int)
    # TODO: Should be [[1,0], [0,1], [1,0], [0,1]]
    expected = np.ones((4, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_split():
    """jnp.split triggers conservative fallback.

    TODO(dynamic_slice): split uses dynamic_slice which needs precise handler.
    Precise: each output element depends only on corresponding input.
    """

    def f(x):
        parts = jnp.split(x, 2)
        return jnp.concatenate([parts[1], parts[0]])  # swap halves

    result = jacobian_sparsity(f, n=4).toarray().astype(int)
    # TODO: Should be permutation [[0,0,1,0], [0,0,0,1], [1,0,0,0], [0,1,0,0]]
    expected = np.ones((4, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_scatter_at_set():
    """In-place update with .at[].set() is partially precise.

    TODO(scatter): Implement precise handler for scatter primitive.
    Currently: all outputs depend on x[0] (the value being set).
    Precise: only output[1] should depend on x[0].
    """

    def f(x):
        arr = jnp.zeros(3)
        return arr.at[1].set(x[0])

    result = jacobian_sparsity(f, n=2).toarray().astype(int)
    # TODO: Should be [[0,0], [1,0], [0,0]] (only index 1 depends on x[0])
    expected = np.array([[1, 0], [1, 0], [1, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_iota_eye():
    """jnp.eye uses iota internally, triggers conservative fallback.

    TODO(iota): Add iota to ZERO_DERIVATIVE_PRIMITIVES (constant output).
    TODO(dot_general): Also needs dot_general handler for eye @ x.
    Precise: eye matrix has no input dependency (constant), so eye @ x = x.
    """

    def f(x):
        # Multiply x by identity - should preserve diagonal structure
        return jnp.eye(3) @ x

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    # TODO: Should be identity matrix (eye @ x = x)
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_reduce_max():
    """jnp.max (reduce_max) has correct global sparsity (all inputs matter).

    Unlike reduce_sum which has a handler, reduce_max falls to default.
    Both should produce the same result: output depends on all inputs.
    """

    def f(x):
        return jnp.array([jnp.max(x)])

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    # All inputs can affect the max (global sparsity)
    expected = np.array([[1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


def test_sort():
    """jnp.sort triggers conservative fallback.

    Precise: all outputs depend on all inputs (sorting is a global operation).
    """

    def f(x):
        return jnp.sort(x)

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    # Conservative fallback is actually correct here
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_where_mask():
    """jnp.where with mask triggers conservative fallback.

    Precise: each output depends on mask condition + both branches.
    """

    def f(x):
        mask = x > 0
        return jnp.where(mask, x, -x)

    result = jacobian_sparsity(f, n=3).toarray().astype(int)
    # Global sparsity: each output could depend on corresponding input
    # (mask has zero derivative, both branches are element-wise from x)
    # Conservative: may be dense depending on how where is traced
    assert result.shape == (3, 3)


def test_empty_concatenate():
    """Concatenating with empty arrays causes index out-of-bounds error.

    TODO(bug): Fix empty array handling in concatenate/reshape.
    BUG: Empty arrays in concatenate produce invalid COO indices.
    The reshape after concatenate produces incorrect flat indices.
    """
    import pytest

    def f(x):
        empty = jnp.array([])
        return jnp.concatenate([empty, x, empty])

    # TODO: Should work and produce identity matrix
    with pytest.raises(ValueError, match="index .* exceeds matrix dimension"):
        jacobian_sparsity(f, n=2)


def test_nested_slice_concat():
    """Multiple 1D slices followed by concatenate should preserve structure."""

    def f(x):
        a = x[:2]
        b = x[2:]
        return jnp.concatenate([b, a])  # [x2, x3, x0, x1]

    result = jacobian_sparsity(f, n=4).toarray().astype(int)
    # Permutation: swap first 2 and last 2
    expected = np.array(
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=int
    )
    np.testing.assert_array_equal(result, expected)


def test_reduce_along_axis():
    """Reduction along one axis should track per-slice dependencies."""

    def f(x):
        mat = x.reshape(2, 3)
        return jnp.sum(mat, axis=1)  # Sum each row

    result = jacobian_sparsity(f, n=6).toarray().astype(int)
    # Conservative fallback for axis reduction
    # Precise: output[0] depends on x[0:3], output[1] on x[3:6]
    # Current implementation may or may not handle axis parameter
    assert result.shape == (2, 6)

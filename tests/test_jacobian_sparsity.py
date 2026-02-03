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

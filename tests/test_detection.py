"""Core API tests and simple element-wise operations."""

import jax.numpy as jnp
import numpy as np
import pytest

from asdex import hessian_sparsity, jacobian_sparsity

# =============================================================================
# Core API tests
# =============================================================================


@pytest.mark.elementwise
def test_simple_dependencies():
    """Test f(x) = [x0+x1, x1*x2, x2]"""

    def f(x):
        return jnp.array([x[0] + x[1], x[1] * x[2], x[2]])

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    expected = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Hessian sparsity tests
# =============================================================================


@pytest.mark.hessian
def test_hessian_linear():
    """Linear functions have zero Hessian."""

    def f(x):
        return x[0] + 2 * x[1] + 3 * x[2]

    H = hessian_sparsity(f, n=3).todense()
    assert H.sum() == 0


@pytest.mark.hessian
@pytest.mark.fallback
def test_hessian_product():
    """f(x) = x[0] * x[1] has H[0,1] = H[1,0] != 0.

    TODO(pad): Precise behavior would be [[0,1,0], [1,0,0], [0,0,0]].
    Currently conservative due to pad primitive in gradient jaxpr.
    """

    def f(x):
        return x[0] * x[1]

    H = hessian_sparsity(f, n=3).todense().astype(int)
    # Conservative: all rows depending on x[0], x[1] are dense
    expected = jnp.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    assert jnp.array_equal(H, expected)


@pytest.mark.hessian
@pytest.mark.fallback
def test_hessian_quadratic():
    """f(x) = x[0]^2 + x[1]^2 has diagonal Hessian.

    TODO(pad): Precise behavior would be [[1,0,0], [0,1,0], [0,0,0]].
    Currently conservative due to pad primitive in gradient jaxpr.
    """

    def f(x):
        return x[0] ** 2 + x[1] ** 2

    H = hessian_sparsity(f, n=3).todense().astype(int)
    # Conservative: all rows depending on x[0], x[1] are dense
    expected = jnp.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    assert jnp.array_equal(H, expected)


@pytest.mark.elementwise
def test_complex_dependencies():
    """Test f(x) = [x0*x1 + sin(x2), x3, x0*x1*x3]"""

    def f(x):
        a = x[0] * x[1]
        b = jnp.sin(x[2])
        c = a + b
        return jnp.array([c, x[3], a * x[3]])

    result = jacobian_sparsity(f, n=4).todense().astype(int)
    expected = np.array([[1, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_diagonal_jacobian():
    """Test f(x) = x^2 (element-wise) produces diagonal sparsity"""

    def f(x):
        return x**2

    result = jacobian_sparsity(f, n=4).todense().astype(int)
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_dense_jacobian():
    """Test f(x) = [sum(x), prod(x)] produces dense sparsity"""

    def f(x):
        return jnp.array([jnp.sum(x), jnp.prod(x)])

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    expected = np.array([[1, 1, 1], [1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_sct_readme_example():
    """Test SCT README example: f(x) = [x1^2, 2*x1*x2^2, sin(x3)]"""

    def f(x):
        return jnp.array([x[0] ** 2, 2 * x[0] * x[1] ** 2, jnp.sin(x[2])])

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    expected = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Identity and constant functions
# =============================================================================


@pytest.mark.elementwise
def test_identity():
    """Identity function: f(x) = x"""

    def f(x):
        return x

    result = jacobian_sparsity(f, n=1).todense().astype(int)
    expected = np.array([[1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_constant():
    """Constant function: output doesn't depend on input."""

    def f(x):
        return jnp.array([1.0])

    result = jacobian_sparsity(f, n=1).todense().astype(int)
    expected = np.array([[0]])
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Zero derivative operations
# =============================================================================


@pytest.mark.elementwise
def test_zero_derivative_ceil_round():
    """ceil/round have zero derivative: f(x) = [x1*x2, ceil(x1*x2), x1*round(x2)]"""

    def f(x):
        return jnp.array([x[0] * x[1], jnp.ceil(x[0] * x[1]), x[0] * jnp.round(x[1])])

    result = jacobian_sparsity(f, n=2).todense().astype(int)
    # round(x2) has zero derivative, so x1*round(x2) only depends on x1
    expected = np.array([[1, 1], [0, 0], [1, 0]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_zero_derivative_floor():
    """floor has zero derivative."""

    def f(x):
        return jnp.floor(x)

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    expected = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_zero_derivative_sign():
    """sign has zero derivative."""

    def f(x):
        return jnp.sign(x)

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    expected = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
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

    result = jacobian_sparsity(f, n=2).todense().astype(int)
    expected = np.zeros((6, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Type conversion
# =============================================================================


@pytest.mark.elementwise
def test_type_conversion():
    """Type conversions preserve dependencies."""

    def f(x):
        return x.astype(jnp.float32)

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Power operations
# =============================================================================


@pytest.mark.elementwise
def test_power_operations():
    """Various power operations: x^e, e^x, etc."""

    def f(x):
        return jnp.array([x[0] ** 2.5, jnp.exp(x[1]), x[2] ** x[2]])

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_integer_pow_zero():
    """x^0 = 1 has no dependency on x."""

    def f(x):
        return x**0

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    expected = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Clamp operations
# =============================================================================


@pytest.mark.elementwise
def test_clamp_scalar_bounds():
    """clamp(x, lo, hi) with scalar bounds preserves element structure."""

    def f(x):
        return jnp.clip(x, 0.0, 1.0)

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_clamp_variable_bounds():
    """clamp(x1, x2, x3) with variable bounds - all contribute."""

    def f(x):
        return jnp.array([jnp.clip(x[0], x[1], x[2])])

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    expected = np.array([[1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Binary operations
# =============================================================================


@pytest.mark.elementwise
def test_dot_product():
    """Dot product: dot(x[0:2], x[3:5])."""

    def f(x):
        return jnp.array([jnp.dot(x[:2], x[3:5])])

    result = jacobian_sparsity(f, n=5).todense().astype(int)
    expected = np.array([[1, 1, 0, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_multiply_by_zero():
    """Multiplying by zero still tracks structural dependency."""

    def f1(x):
        return jnp.array([0 * x[0]])

    def f2(x):
        return jnp.array([x[0] * 0])

    # Global sparsity: we can't know at compile time that result is zero
    result1 = jacobian_sparsity(f1, n=1).todense().astype(int)
    result2 = jacobian_sparsity(f2, n=1).todense().astype(int)
    expected = np.array([[1]])
    np.testing.assert_array_equal(result1, expected)
    np.testing.assert_array_equal(result2, expected)


@pytest.mark.elementwise
def test_binary_min_max():
    """min and max operations."""

    def f(x):
        return jnp.array([jnp.minimum(x[0], x[1]), jnp.maximum(x[1], x[2])])

    result = jacobian_sparsity(f, n=3).todense().astype(int)
    expected = np.array([[1, 1, 0], [0, 1, 1]])
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Unary functions
# =============================================================================


@pytest.mark.elementwise
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

    result = jacobian_sparsity(f, n=3).todense().astype(int)
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

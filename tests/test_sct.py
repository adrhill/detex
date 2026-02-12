"""Tests ported from SparseConnectivityTracer.jl's Global Jacobian/Hessian testsets.

Reference:
- https://github.com/adrhill/SparseConnectivityTracer.jl/blob/main/test/test_gradient.jl
- https://github.com/adrhill/SparseConnectivityTracer.jl/blob/main/test/test_hessian.jl
"""

import jax.numpy as jnp
import numpy as np
import pytest

from asdex import hessian_sparsity, jacobian_sparsity

# =============================================================================
# Global Jacobian — from SparseConnectivityTracer test_gradient.jl
# =============================================================================


@pytest.mark.elementwise
def test_jacobian_sincos():
    """Both sin and cos of the same input depend on that input.

    SCT: x -> [sincos(x)...]
    """

    def f(x):
        return jnp.array([jnp.sin(x[0]), jnp.cos(x[0])])

    result = jacobian_sparsity(f, input_shape=1).todense().astype(int)
    expected = np.array([[1], [1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_jacobian_vector_index_arithmetic():
    """Complex index arithmetic with abs2 differences.

    SCT: ret[i] = abs2(x[i+1]) - abs2(x[i]) + abs2(x[n-i]) - abs2(x[n-i+1])
    """

    def f(x):
        n = x.shape[0]
        out = [
            x[i + 1] ** 2 - x[i] ** 2 + x[n - 1 - i] ** 2 - x[n - i - 2] ** 2
            for i in range(n - 1)
        ]
        return jnp.array(out)

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 1],
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_jacobian_where_both_branches():
    """jnp.where unions both branches for global sparsity.

    SCT: x -> ifelse(x[2] < x[3], x[1] + x[2], x[3] * x[4])
    """

    def f(x):
        return jnp.array([jnp.where(x[1] < x[2], x[0] + x[1], x[2] * x[3])])

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array([[1, 1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_jacobian_where_one_constant_true():
    """jnp.where with constant false branch only tracks true branch.

    SCT: x -> ifelse(x[2] < x[3], x[1] + x[2], 1.0)
    """

    def f(x):
        return jnp.array([jnp.where(x[1] < x[2], x[0] + x[1], 1.0)])

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array([[1, 1, 0, 0]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_jacobian_where_one_constant_false():
    """jnp.where with constant true branch only tracks false branch.

    SCT: x -> ifelse(x[2] < x[3], 1.0, x[3] * x[4])
    """

    def f(x):
        return jnp.array([jnp.where(x[1] < x[2], 1.0, x[2] * x[3])])

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array([[0, 0, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_jacobian_composite_foo():
    """Composite function: all inputs contribute to output.

    SCT: foo(x) = x[1] + x[2] * x[3] + 1 / x[4] + 1 * x[5]
    """

    def f(x):
        return jnp.array([x[0] + x[1] * x[2] + 1.0 / x[3] + 1.0 * x[4]])

    result = jacobian_sparsity(f, input_shape=5).todense().astype(int)
    expected = np.array([[1, 1, 1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_jacobian_composite_bar():
    """Composite function with power term: foo(x) + x[2]^x[5].

    SCT: bar(x) = foo(x) + x[2]^x[5]
    """

    def f(x):
        foo = x[0] + x[1] * x[2] + 1.0 / x[3] + 1.0 * x[4]
        return jnp.array([foo + x[1] ** x[4]])

    result = jacobian_sparsity(f, input_shape=5).todense().astype(int)
    expected = np.array([[1, 1, 1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_jacobian_ampgo07():
    """AMPGO07 benchmark: complex expression with comparison, sin, log, abs.

    SCT: f(x) = (x <= 0) * Inf + sin(x) + sin(10/3 * x) + log(|x|) - 0.84*x + 3
    The (x <= 0) comparison has zero derivative,
    so the entire expression depends on x.
    """

    def f(x):
        return jnp.array(
            [
                (x[0] <= 0).astype(float) * jnp.inf
                + jnp.sin(x[0])
                + jnp.sin(10.0 / 3.0 * x[0])
                + jnp.log(jnp.abs(x[0]))
                - 0.84 * x[0]
                + 3.0
            ]
        )

    result = jacobian_sparsity(f, input_shape=1).todense().astype(int)
    expected = np.array([[1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.hessian
def test_hessian_ampgo07():
    """AMPGO07 benchmark: Hessian is nonzero due to sin, log.

    SCT: f(x) = (x <= 0) * Inf + sin(x) + sin(10/3 * x) + log(|x|) - 0.84*x + 3
    """

    def f(x):
        return (
            (x[0] <= 0).astype(float) * jnp.inf
            + jnp.sin(x[0])
            + jnp.sin(10.0 / 3.0 * x[0])
            + jnp.log(jnp.abs(x[0]))
            - 0.84 * x[0]
            + 3.0
        )

    H = hessian_sparsity(f, input_shape=1).todense().astype(int)
    expected = np.array([[1]])
    np.testing.assert_array_equal(H, expected)


# =============================================================================
# Global Hessian — from SparseConnectivityTracer test_hessian.jl
# =============================================================================


@pytest.mark.hessian
def test_hessian_sqrt():
    """Sqrt has nonzero second derivative.

    SCT: x -> sqrt(x) → H = [1]
    """

    def f(x):
        return jnp.sqrt(x[0])

    H = hessian_sparsity(f, input_shape=1).todense().astype(int)
    expected = np.array([[1]])
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_linear_mul():
    """Multiplication by constant is linear, zero Hessian.

    SCT: x -> 1 * x, x -> x * 1
    """

    def f1(x):
        return 1.0 * x[0]

    def f2(x):
        return x[0] * 1.0

    H1 = hessian_sparsity(f1, input_shape=1).todense().astype(int)
    H2 = hessian_sparsity(f2, input_shape=1).todense().astype(int)
    expected = np.array([[0]])
    np.testing.assert_array_equal(H1, expected)
    np.testing.assert_array_equal(H2, expected)


@pytest.mark.hessian
def test_hessian_diff_cubic():
    """sum(diff(x).^3) has tridiagonal Hessian.

    SCT: x -> sum(diff(x) .^ 3)
    """

    def f(x):
        d = x[1:] - x[:-1]
        return jnp.sum(d**3)

    H = hessian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_division():
    """Division creates second-order interactions.

    SCT: x -> x[1] / x[2] + x[3] / 1 + 1 / x[4]
    """

    def f(x):
        return x[0] / x[1] + x[2] / 1.0 + 1.0 / x[3]

    H = hessian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_product_linear():
    """Products with constants are linear, only cross-terms are nonzero.

    SCT: x -> x[1] * x[2] + x[3] * 1 + 1 * x[4]
    """

    def f(x):
        return x[0] * x[1] + x[2] * 1.0 + 1.0 * x[3]

    H = hessian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_product_of_products():
    """(x1*x2)*(x3*x4) has all cross-terms nonzero except diagonal.

    SCT: x -> (x[1] * x[2]) * (x[3] * x[4])
    """

    def f(x):
        return (x[0] * x[1]) * (x[2] * x[3])

    H = hessian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_product_of_sums():
    """(x1+x2)*(x3+x4) only has cross-group interactions.

    SCT: x -> (x[1] + x[2]) * (x[3] + x[4])
    """

    def f(x):
        return (x[0] + x[1]) * (x[2] + x[3])

    H = hessian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_sum_squared():
    """(x1+x2+x3+x4)^2 has fully dense Hessian.

    SCT: x -> (x[1] + x[2] + x[3] + x[4])^2
    """

    def f(x):
        return (x[0] + x[1] + x[2] + x[3]) ** 2

    H = hessian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.ones((4, 4), dtype=int)
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_reciprocal_sum():
    """1/(x1+x2+x3+x4) has fully dense Hessian.

    SCT: x -> 1 / (x[1] + x[2] + x[3] + x[4])
    """

    def f(x):
        return 1.0 / (x[0] + x[1] + x[2] + x[3])

    H = hessian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.ones((4, 4), dtype=int)
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_subtraction_linear():
    """Subtraction is linear, zero Hessian.

    SCT: x -> (x[1] - x[2]) + (x[3] - 1) + (1 - x[4])
    """

    def f(x):
        return (x[0] - x[1]) + (x[2] - 1.0) + (1.0 - x[3])

    H = hessian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.zeros((4, 4), dtype=int)
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_composite_foo():
    """Composite function with mixed operations.

    SCT: foo(x) = x[1] + x[2] * x[3] + 1 / x[4] + 1 * x[5]
    """

    def f(x):
        return x[0] + x[1] * x[2] + 1.0 / x[3] + 1.0 * x[4]

    H = hessian_sparsity(f, input_shape=5).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_composite_bar():
    """Composite function with power term: foo(x) + x[2]^x[5].

    SCT: bar(x) = foo(x) + x[2]^x[5]
    where foo(x) = x[1] + x[2] * x[3] + 1 / x[4] + 1 * x[5]
    """

    def f(x):
        foo = x[0] + x[1] * x[2] + 1.0 / x[3] + 1.0 * x[4]
        return foo + x[1] ** x[4]

    H = hessian_sparsity(f, input_shape=5).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_clamp_scalar():
    """Clamp with scalar bounds is piecewise linear, zero Hessian.

    SCT: x -> clamp(x, 0.1, 0.9)
    """

    def f(x):
        return jnp.clip(x[0], 0.1, 0.9)

    H = hessian_sparsity(f, input_shape=1).todense().astype(int)
    expected = np.array([[0]])
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_clamp_variable_bounds():
    """Clamp with variable bounds has zero Hessian (piecewise linear in each arg).

    SCT: x -> clamp(x[1], x[2], x[3])
    """

    def f(x):
        return jnp.clip(x[0], x[1], x[2])

    H = hessian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_clamp_times_x1():
    """x1 * clamp(x1, x2, x3) creates cross-term interactions.

    SCT: x -> x[1] * clamp(x[1], x[2], x[3])
    """

    def f(x):
        return x[0] * jnp.clip(x[0], x[1], x[2])

    H = hessian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_clamp_times_x2():
    """x2 * clamp(x1, x2, x3) creates cross-term interactions.

    SCT: x -> x[2] * clamp(x[1], x[2], x[3])
    """

    def f(x):
        return x[1] * jnp.clip(x[0], x[1], x[2])

    H = hessian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_clamp_times_x3():
    """x3 * clamp(x1, x2, x3) creates cross-term interactions.

    SCT: x -> x[3] * clamp(x[1], x[2], x[3])
    """

    def f(x):
        return x[2] * jnp.clip(x[0], x[1], x[2])

    H = hessian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
@pytest.mark.control_flow
def test_hessian_where_both_branches():
    """jnp.where unions Hessian patterns from both branches.

    SCT: x -> ifelse(x[1], x[1]^x[2], x[3] * x[4])
    """

    def f(x):
        return jnp.where(x[0] > 0, x[0] ** x[1], x[2] * x[3])

    H = hessian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
@pytest.mark.control_flow
def test_hessian_where_one_constant_true():
    """jnp.where with constant false branch.

    SCT: x -> ifelse(x[1], x[1]^x[2], 1.0)
    """

    def f(x):
        return jnp.where(x[0] > 0, x[0] ** x[1], 1.0)

    H = hessian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
@pytest.mark.control_flow
def test_hessian_where_one_constant_false():
    """jnp.where with constant true branch.

    SCT: x -> ifelse(x[1], 1.0, x[3] * x[4])
    """

    def f(x):
        return jnp.where(x[0] > 0, 1.0, x[2] * x[3])

    H = hessian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_multiply_by_zero():
    """Multiplying by zero still tracks structural Hessian.

    SCT: f1(x) = 0 * x[1]^2, f2(x) = x[1]^2 * 0
    """

    def f1(x):
        return 0.0 * x[0] ** 2

    def f2(x):
        return x[0] ** 2 * 0.0

    H1 = hessian_sparsity(f1, input_shape=1).todense().astype(int)
    H2 = hessian_sparsity(f2, input_shape=1).todense().astype(int)
    expected = np.array([[1]])
    np.testing.assert_array_equal(H1, expected)
    np.testing.assert_array_equal(H2, expected)

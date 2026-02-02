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

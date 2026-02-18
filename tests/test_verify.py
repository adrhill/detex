"""Tests for the verification utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from asdex import (
    VerificationError,
    check_hessian_correctness,
    check_jacobian_correctness,
    hessian_coloring,
    jacobian_coloring,
)

# Jacobian verification


@pytest.mark.jacobian
def test_check_jacobian_passes():
    """check_jacobian_correctness returns silently on correct results."""

    def f(x):
        return (x[1:] - x[:-1]) ** 2

    x = np.array([1.0, 2.0, 3.0, 4.0])
    check_jacobian_correctness(f, x)


@pytest.mark.jacobian
def test_check_jacobian_with_precomputed_pattern():
    """check_jacobian_correctness works with a pre-computed colored pattern."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0])
    colored_pattern = jacobian_coloring(f, input_shape=x.shape)
    check_jacobian_correctness(f, x, colored_pattern=colored_pattern)


@pytest.mark.jacobian
def test_check_jacobian_custom_tolerances():
    """check_jacobian_correctness respects custom tolerances."""

    def f(x):
        return jnp.sin(x)

    x = np.array([0.5, 1.0, 1.5])
    check_jacobian_correctness(f, x, rtol=1e-5, atol=1e-5)


@pytest.mark.jacobian
def test_check_jacobian_raises_on_mismatch():
    """check_jacobian_correctness raises VerificationError on wrong results.

    Uses a diagonal colored pattern for a function with off-diagonal entries,
    so the sparse Jacobian misses non-zeros.
    """

    def f_dense(x):
        return jnp.array([x[0] + x[1] + x[2], x[0] + x[1] + x[2], x[0] + x[1] + x[2]])

    # Diagonal pattern misses off-diagonal Jacobian entries
    colored_pattern = jacobian_coloring(lambda x: x**2, input_shape=(3,))

    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(VerificationError, match="does not match"):
        check_jacobian_correctness(f_dense, x, colored_pattern=colored_pattern)


# Hessian verification


@pytest.mark.hessian
def test_check_hessian_passes():
    """check_hessian_correctness returns silently on correct results."""

    def f(x):
        return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

    x = np.array([1.0, 1.0, 1.0, 1.0])
    check_hessian_correctness(f, x)


@pytest.mark.hessian
def test_check_hessian_custom_tolerances():
    """check_hessian_correctness respects custom tolerances."""

    def f(x):
        return jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    check_hessian_correctness(f, x, rtol=1e-5, atol=1e-5)


@pytest.mark.hessian
def test_check_hessian_raises_on_mismatch():
    """check_hessian_correctness raises VerificationError on wrong results.

    Uses a diagonal colored pattern for a function with off-diagonal Hessian entries,
    so the sparse Hessian misses non-zeros.
    """

    def f(x):
        return x[0] * x[1] + x[1] * x[2]

    # Diagonal pattern misses off-diagonal Hessian entries
    colored_pattern = hessian_coloring(lambda x: jnp.sum(x**2), input_shape=(3,))

    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(VerificationError, match="does not match"):
        check_hessian_correctness(f, x, colored_pattern=colored_pattern)


# VerificationError


def test_verification_error_is_assertion_error():
    """VerificationError subclasses AssertionError."""
    assert issubclass(VerificationError, AssertionError)

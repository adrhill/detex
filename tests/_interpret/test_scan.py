"""Tests for scan propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian, jacobian_sparsity


@pytest.mark.control_flow
def test_scan_cumulative_sum():
    """Cumulative sum: scalar carry accumulates over 1D xs.

    carry[t] = carry[t-1] + xs[t], so the final carry depends on all xs elements.
    ys[t] = carry[t], so ys[t] depends on xs[0..t].
    We overapproximate: every ys element depends on all xs elements.
    """

    def f(x):
        def body(carry, xi):
            new_carry = carry + xi
            return new_carry, new_carry

        _, ys = jax.lax.scan(body, 0.0, x)
        return ys

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.ones((4, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_2d_carry_and_xs():
    """2D carry and xs: elementwise accumulation preserves structure.

    Each carry element accumulates only from the corresponding xs element,
    so the pattern is block-diagonal.
    """

    def f(x):
        x_2d = x.reshape(3, 2)

        def body(carry, xi):
            new_carry = carry + xi
            return new_carry, new_carry

        _, ys = jax.lax.scan(body, jnp.zeros(2), x_2d)
        return ys.ravel()

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    # Overapproximate: all ys depend on all xs slices,
    # but within each slice, element 0 depends on column 0 and element 1 on column 1.
    expected = np.array(
        [
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_multiple_carries():
    """Multiple carry variables: (array, scalar) carry.

    The array carry accumulates elementwise,
    the scalar carry counts iterations.
    Only the array output is returned.
    """

    def f(x):
        def body(carry, xi):
            arr, count = carry
            new_arr = arr + xi
            return (new_arr, count + 1.0), new_arr

        (arr_out, _), _ = jax.lax.scan(body, (jnp.zeros(3), 0.0), x.reshape(2, 3))
        return arr_out

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    # arr_out depends on all xs slices elementwise
    expected = np.array(
        [
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_carry_only():
    """Carry-only scan with no xs (length from params).

    The body only transforms the carry without scanning over inputs.
    Identity body preserves diagonal pattern.
    """

    def f(x):
        def body(carry, _):
            return carry, None

        carry_out, _ = jax.lax.scan(body, x, None, length=5)
        return carry_out

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_with_closure_const():
    """Scan body captures an external array via closure.

    Elementwise multiply with a constant preserves diagonal pattern.
    """
    weights = jnp.array([2.0, 3.0])

    def f(x):
        def body(carry, xi):
            new_carry = carry * weights
            return new_carry, new_carry

        _, ys = jax.lax.scan(body, x, None, length=3)
        return ys.ravel()

    result = jacobian_sparsity(f, input_shape=2).todense().astype(int)
    # Each output element depends only on the corresponding input element
    expected = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_reverse():
    """reverse=True scans xs in reverse order.

    Sparsity pattern is the same as forward since we overapproximate
    across all time steps.
    """

    def f(x):
        def body(carry, xi):
            return carry + xi, carry + xi

        _, ys = jax.lax.scan(body, 0.0, x, reverse=True)
        return ys

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_reverse_multi_carry():
    """reverse=True with multi-element carry.

    Elementwise accumulation in reverse still produces the same
    overapproximate sparsity pattern.
    """

    def f(x):
        x_2d = x.reshape(2, 3)

        def body(carry, xi):
            return carry + xi, carry

        carry_out, _ = jax.lax.scan(body, jnp.zeros(3), x_2d, reverse=True)
        return carry_out

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_identity_body():
    """Identity body: carry passes through unchanged, ys copies carry.

    Converges in one iteration. Output depends only on carry_init.
    """

    def f(x):
        def body(carry, xi):
            return carry, carry

        _, ys = jax.lax.scan(body, x, None, length=4)
        return ys.ravel()

    result = jacobian_sparsity(f, input_shape=2).todense().astype(int)
    expected = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_composition():
    """Nested scan: outer scan calls inner scan in its body.

    Inner scan accumulates elementwise,
    outer scan accumulates the inner result.
    """

    def f(x):
        x_4d = x.reshape(2, 2, 2)

        def inner_body(carry, xi):
            return carry + xi, None

        def outer_body(carry, xi):
            inner_carry, _ = jax.lax.scan(inner_body, carry, xi)
            return inner_carry, inner_carry

        _, ys = jax.lax.scan(outer_body, jnp.zeros(2), x_4d)
        return ys.ravel()

    result = jacobian_sparsity(f, input_shape=8).todense().astype(int)
    # All outputs depend on all inputs with matching element index (mod 2)
    expected = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_noncontiguous_input():
    """Scan with non-contiguous input dependencies.

    Only odd-indexed inputs are used via slicing before scan.
    """

    def f(x):
        xs = x[1::2]  # select odd indices: x[1], x[3], x[5]

        def body(carry, xi):
            return carry + xi, carry + xi

        _, ys = jax.lax.scan(body, 0.0, xs)
        return ys

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_vs_jax_jacobian():
    """Verify scan sparsity against dense jax.jacobian for correctness.

    The detected sparsity pattern must be a superset of the true nonzero pattern.
    """

    def f(x):
        def body(carry, xi):
            return carry + xi**2, carry * xi

        carry_out, ys = jax.lax.scan(body, jnp.zeros(2), x.reshape(3, 2))
        return jnp.concatenate([carry_out, ys.ravel()])

    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    detected = jacobian_sparsity(f, input_shape=6).todense().astype(bool)
    dense_jac = jax.jacobian(f)(x)
    true_nonzero = np.abs(np.array(dense_jac)) > 0

    # Detected pattern must cover all true nonzeros
    assert np.all(detected | ~true_nonzero), "Detected sparsity misses true nonzeros"


@pytest.mark.control_flow
def test_scan_jacobian_values():
    """Verify sparse Jacobian values match dense jax.jacobian."""

    def f(x):
        def body(carry, xi):
            return carry + xi, carry * 2.0

        carry_out, ys = jax.lax.scan(body, 0.0, x)
        return jnp.concatenate([jnp.array([carry_out]), ys])

    x = jnp.array([1.0, 2.0, 3.0])
    sparse_jac = jacobian(f)(x).todense()
    dense_jac = np.array(jax.jacobian(f)(x))
    np.testing.assert_allclose(sparse_jac, dense_jac)

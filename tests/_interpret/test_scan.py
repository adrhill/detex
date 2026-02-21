"""Tests for scan propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_coloring, jacobian_from_coloring, jacobian_sparsity


@pytest.mark.fallback
@pytest.mark.control_flow
def test_scan_cumulative_sum():
    """Cumulative sum: scalar carry accumulates over 1D xs.

    carry[t] = carry[t-1] + xs[t], so the final carry depends on all xs elements.
    ys[t] = carry[t], so ys[t] depends on xs[0..t].
    We overapproximate: every ys element depends on all xs elements.

    TODO(scan): the true pattern is lower-triangular (ys[t] depends on xs[0..t]).
    The scan handler merges xs deps across all time steps.
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


@pytest.mark.fallback
@pytest.mark.control_flow
def test_scan_2d_carry_and_xs():
    """2D carry and xs: elementwise accumulation preserves structure.

    Each carry element accumulates only from the corresponding xs element,
    so the pattern is block-diagonal.

    TODO(scan): the true pattern is a progressive block-diagonal
    where ys[t] depends only on xs[0..t] (per element).
    The scan handler merges xs deps across all time steps,
    making every ys slice depend on all xs slices.
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


@pytest.mark.fallback
@pytest.mark.control_flow
def test_scan_reverse():
    """reverse=True scans xs in reverse order.

    Sparsity pattern is the same as forward since we overapproximate
    across all time steps.

    TODO(scan): the true pattern is upper-triangular
    (ys[t] depends on xs[t..n-1] when scanning in reverse).
    The scan handler merges xs deps across all time steps.
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


@pytest.mark.fallback
@pytest.mark.control_flow
def test_scan_noncontiguous_input():
    """Scan with non-contiguous input dependencies.

    Only odd-indexed inputs are used via slicing before scan.

    TODO(scan): the true pattern is progressive:
    [[0,1,0,0,0,0],      # ys[0] = x[1]
     [0,1,0,1,0,0],      # ys[1] = x[1] + x[3]
     [0,1,0,1,0,1]]      # ys[2] = x[1] + x[3] + x[5]
    The scan handler merges xs deps across all time steps.
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
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_scan_jacobian_values():
    """Verify sparse Jacobian values match dense jax.jacobian."""

    def f(x):
        def body(carry, xi):
            return carry + xi, carry * 2.0

        carry_out, ys = jax.lax.scan(body, 0.0, x)
        return jnp.concatenate([jnp.array([carry_out]), ys])

    x = jnp.array([1.0, 2.0, 3.0])
    sparse_jac = jacobian_from_coloring(f, jacobian_coloring(f, x.shape))(x).todense()
    dense_jac = np.array(jax.jacobian(f)(x))
    np.testing.assert_allclose(sparse_jac, dense_jac)


@pytest.mark.control_flow
def test_scan_pytree_xs():
    """Pytree xs: tuple of arrays as scan inputs.

    JAX flattens leaves into separate invars,
    so this exercises multiple xs vars with different dep patterns.
    """

    def f(x):
        a = x[:3]
        b = x[3:6]

        def body(carry, xsi):
            ai, bi = xsi
            return carry + ai * bi, carry

        carry_out, _ = jax.lax.scan(body, 0.0, (a, b))
        return jnp.array([carry_out])

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    expected = np.ones((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.fallback
@pytest.mark.control_flow
def test_scan_pytree_ys():
    """Pytree ys: body returns a tuple as the y output.

    Exercises multiple ys outvars in the body jaxpr.

    TODO(scan): the true pattern has lower-triangular sums
    and diagonal doubled:
    [[1,0,0], [1,1,0], [1,1,1],  # sums[t] depends on x[0..t]
     [1,0,0], [0,1,0], [0,0,1]]  # doubled[t] depends on x[t]
    The scan handler merges xs deps across all time steps.
    """

    def f(x):
        def body(carry, xi):
            new_carry = carry + xi
            return new_carry, (new_carry, xi * 2.0)

        _, (sums, doubled) = jax.lax.scan(body, 0.0, x)
        return jnp.concatenate([sums, doubled])

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.ones((6, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.fallback
@pytest.mark.control_flow
def test_scan_length_one():
    """Single iteration: carry deps equal init deps, ys has one slice.

    TODO(scan): the true pattern is:
    [[1,0], [0,1],  # carry_out = zeros + x (precise)
     [0,0], [0,0]]  # ys[0] = carry_init = zeros, no x deps
    The scan handler merges carry deps into ys across iterations.
    """

    def f(x):
        def body(carry, xi):
            return carry + xi, carry

        carry_out, ys = jax.lax.scan(body, jnp.zeros(2), x.reshape(1, 2))
        return jnp.concatenate([carry_out, ys.ravel()])

    result = jacobian_sparsity(f, input_shape=2).todense().astype(int)
    expected = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.fallback
@pytest.mark.control_flow
def test_scan_scalar_carry_scalar_xs():
    """Simplest possible scan: scalar carry, scalar xs slices.

    TODO(scan): the true pattern is:
    [[1,1,1],      # carry_out = sum(x)
     [0,0,0],      # ys[0] = carry_init = 0
     [1,0,0],      # ys[1] = x[0]
     [1,1,0]]      # ys[2] = x[0] + x[1]
    The scan handler merges deps across all time steps.
    """

    def f(x):
        def body(carry, xi):
            return carry + xi, carry

        carry_out, ys = jax.lax.scan(body, 0.0, x)
        return jnp.concatenate([jnp.array([carry_out]), ys])

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.ones((4, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_unroll():
    """unroll=True only affects XLA lowering, not the jaxpr.

    Sparsity should be identical to unroll=1.
    """

    def f_unrolled(x):
        def body(carry, xi):
            return carry + xi, carry + xi

        _, ys = jax.lax.scan(body, 0.0, x, unroll=True)
        return ys

    def f_default(x):
        def body(carry, xi):
            return carry + xi, carry + xi

        _, ys = jax.lax.scan(body, 0.0, x)
        return ys

    sp_unrolled = jacobian_sparsity(f_unrolled, input_shape=4).todense().astype(int)
    sp_default = jacobian_sparsity(f_default, input_shape=4).todense().astype(int)
    np.testing.assert_array_equal(sp_unrolled, sp_default)


@pytest.mark.fallback
@pytest.mark.control_flow
def test_scan_ys_independent_of_carry():
    """Ys depend only on xs, not on carry.

    Body: y = f(xi) only, carry is a counter.
    ys should not depend on carry_init.

    TODO(scan): the true pattern is diagonal per xs slice:
    [[0,1,0,0], [0,0,1,0], [0,0,0,1]]
    The scan handler merges xs deps across all time steps.
    """

    def f(x):
        xs = x[1:].reshape(3, 1)
        carry_init = x[0:1]

        def body(carry, xi):
            return carry + 1.0, xi * 2.0

        _, ys = jax.lax.scan(body, carry_init, xs)
        return ys.ravel()

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_carry_only_with_mixing():
    """Carry-only scan where the body mixes carry elements.

    Tests fixed-point convergence without xs,
    analogous to while_loop dependency spreading.
    """

    def f(x):
        def body(carry, _):
            # Mix: each element depends on both
            return jnp.array([carry[0] + carry[1], carry[1]]), None

        carry_out, _ = jax.lax.scan(body, x, None, length=3)
        return carry_out

    result = jacobian_sparsity(f, input_shape=2).todense().astype(int)
    # carry[0] depends on both inputs after mixing, carry[1] only on itself
    expected = np.array(
        [
            [1, 1],
            [0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_carry_interaction_across_tuple():
    """Carry elements interact across iterations via tuple carry.

    One carry element feeds into another,
    so deps spread through fixed-point iteration.
    """

    def f(x):
        def body(carry, _):
            a, b = carry
            # a gets b's value, b stays
            return (a + b, b), None

        (a_out, b_out), _ = jax.lax.scan(body, (x[:2], x[2:4]), None, length=3)
        return jnp.concatenate([a_out, b_out])

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    # a_out depends on both a_init and b_init (elementwise)
    # b_out depends only on b_init
    expected = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.fallback
@pytest.mark.control_flow
def test_scan_with_cond_inside():
    """Scan body contains a conditional branch.

    Exercises scan + cond interaction.

    TODO(scan): the true pattern is lower-triangular
    (each ys[t] depends on xs[0..t]).
    The scan handler merges xs deps across all time steps.
    """

    def f(x):
        def body(carry, xi):
            new_carry = jax.lax.cond(
                True,
                lambda c, v: c + v,
                lambda c, v: c * v,
                carry,
                xi,
            )
            return new_carry, new_carry

        _, ys = jax.lax.scan(body, 0.0, x)
        return ys

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # All ys depend on all xs (carry accumulates across iterations)
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_scan_jacobian_values_pytree_xs():
    """Verify sparse Jacobian values match dense jax.jacobian with pytree xs."""

    def f(x):
        a = x[:3]
        b = x[3:6]

        def body(carry, xsi):
            ai, bi = xsi
            return carry + ai * bi, carry

        carry_out, ys = jax.lax.scan(body, 0.0, (a, b))
        return jnp.concatenate([jnp.array([carry_out]), ys])

    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sparse_jac = jacobian_from_coloring(f, jacobian_coloring(f, x.shape))(x).todense()
    dense_jac = np.array(jax.jacobian(f)(x))
    np.testing.assert_allclose(sparse_jac, dense_jac)

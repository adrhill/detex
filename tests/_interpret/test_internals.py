"""Unit tests for internal propagation functions.

https://docs.jax.dev/en/latest/jaxpr.html
"""

import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.core import Primitive

from asdex import jacobian_sparsity
from asdex._interpret import (
    prop_custom_call,
    prop_dispatch,
    prop_jaxpr,
    prop_nested_jaxpr,
)
from asdex._interpret._reshape import prop_reshape


class FakeAval:
    """Fake abstract value with shape."""

    def __init__(self, shape):
        self.shape = shape


class FakeVar:
    """Fake Var for testing with shape info."""

    def __init__(self, shape):
        self.aval = FakeAval(shape)


class FakeEqn:
    """Fake JaxprEqn for testing."""

    def __init__(self, primitive_name: str, params: dict):
        self.primitive = Primitive(primitive_name)
        self.params = params
        self.invars = []
        self.outvars = []


def test_nested_jaxpr_missing_param_raises():
    """Error is raised when nested jaxpr primitive has no 'jaxpr' parameter."""
    eqn = FakeEqn("pjit", params={})
    env = {}
    const_vals = {}

    with pytest.raises(ValueError, match="has no 'jaxpr' parameter"):
        prop_nested_jaxpr(eqn, env, const_vals)  # type: ignore[arg-type]


def test_nested_jaxpr_missing_param_error_message():
    """Error message includes primitive name and issue tracker URL."""
    eqn = FakeEqn("xla_call", params={})
    env = {}
    const_vals = {}

    with pytest.raises(ValueError, match="xla_call"):
        prop_nested_jaxpr(eqn, env, const_vals)  # type: ignore[arg-type]


def test_custom_call_missing_param_raises():
    """Error is raised when custom call primitive has no 'call_jaxpr' parameter."""
    eqn = FakeEqn("custom_jvp_call", params={})
    env = {}
    const_vals = {}

    with pytest.raises(ValueError, match="has no 'call_jaxpr' parameter"):
        prop_custom_call(eqn, env, const_vals)  # type: ignore[arg-type]


def test_custom_call_missing_param_error_message():
    """Error message includes primitive name and issue tracker URL."""
    eqn = FakeEqn("custom_vjp_call", params={})
    env = {}
    const_vals = {}

    with pytest.raises(ValueError, match="custom_vjp_call"):
        prop_custom_call(eqn, env, const_vals)  # type: ignore[arg-type]


def test_unknown_primitive_raises():
    """Unknown primitives raise NotImplementedError."""
    eqn = FakeEqn("nonexistent_op", params={})
    deps = {}
    const_vals = {}

    with pytest.raises(NotImplementedError, match="No handler for primitive"):
        prop_dispatch(eqn, deps, const_vals)  # type: ignore[arg-type]


def test_unknown_primitive_error_message():
    """Error message includes primitive name and issue tracker URL."""
    eqn = FakeEqn("fake_primitive", params={})
    deps = {}
    const_vals = {}

    with pytest.raises(NotImplementedError) as exc_info:
        prop_dispatch(eqn, deps, const_vals)  # type: ignore[arg-type]

    assert "fake_primitive" in str(exc_info.value)
    assert "https://github.com/adrhill/asdex/issues" in str(exc_info.value)


def test_prop_jaxpr_default_const_vals():
    """prop_jaxpr works when const_vals is not provided (defaults to {})."""
    dummy = jnp.zeros(2)
    closed_jaxpr = jax.make_jaxpr(lambda x: x + 1)(dummy)
    jaxpr = closed_jaxpr.jaxpr

    input_indices = [[{0}, {1}]]
    # Call without const_vals — should default to empty dict
    result = prop_jaxpr(jaxpr, input_indices)
    assert len(result) == 1
    assert result[0] == [{0}, {1}]


@pytest.mark.elementwise
def test_stop_gradient():
    """stop_gradient passes dependencies through unchanged."""

    def f(x):
        return jax.lax.stop_gradient(x)

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_reshape_size_mismatch_fallback():
    """Reshape with input/output size mismatch uses conservative fallback.

    This defensive branch handles unexpected cases where element counts differ.
    """
    in_var = FakeVar(shape=(3,))
    out_var = FakeVar(shape=(2,))  # Mismatched size

    eqn = FakeEqn("reshape", params={"new_sizes": (2,), "dimensions": None})
    eqn.invars = [in_var]
    eqn.outvars = [out_var]

    deps = {in_var: [{0}, {1}, {2}]}
    prop_reshape(eqn, deps)  # type: ignore[arg-type]

    # Conservative: all outputs get union of all input deps
    assert deps[out_var] == [{0, 1, 2}, {0, 1, 2}]


# =============================================================================
# Conservative fallback tests
# =============================================================================


@pytest.mark.array_ops
def test_transpose_2d():
    """Transpose preserves per-element dependencies with coordinate reordering.

    output[i,j] depends only on input[j,i], so the Jacobian is a permutation matrix.
    """

    def f(x):
        mat = x.reshape(2, 3)
        return mat.T.flatten()  # (3, 2) -> 6 elements

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    # Transpose of (2,3) -> (3,2): out[i,j] = in[j,i].
    # Flat mapping: out[0]=in[0], out[1]=in[3], out[2]=in[1],
    #               out[3]=in[4], out[4]=in[2], out[5]=in[5].
    expected = np.zeros((6, 6), dtype=int)
    for out_idx, in_idx in enumerate([0, 3, 1, 4, 2, 5]):
        expected[out_idx, in_idx] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.fallback
def test_reverse():
    """jnp.flip triggers conservative fallback.

    TODO(rev): Implement precise handler for rev (reverse) primitive.
    Precise: output[i] depends on input[n-1-i] (anti-diagonal permutation).
    """

    def f(x):
        return jnp.flip(x)

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # TODO: Should be anti-diagonal [[0,0,1], [0,1,0], [1,0,0]]
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad():
    """Pad inserts constant elements; original elements preserve dependencies."""

    def f(x):
        return jnp.pad(x, (1, 1), constant_values=0)

    result = jacobian_sparsity(f, input_shape=2).todense().astype(int)
    expected = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
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

    result = jacobian_sparsity(f, input_shape=2).todense().astype(int)
    # TODO: Should be [[1,0], [0,1], [1,0], [0,1]]
    expected = np.ones((4, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.fallback
def test_split():
    """jnp.split triggers conservative fallback via the split primitive.

    TODO(split): Implement precise handler for split primitive.
    Precise: each output element depends only on corresponding input.
    """

    def f(x):
        parts = jnp.split(x, 2)
        return jnp.concatenate([parts[1], parts[0]])  # swap halves

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    # TODO: Should be permutation [[0,0,1,0], [0,0,0,1], [1,0,0,0], [0,1,0,0]]
    expected = np.ones((4, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)


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

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    # TODO: Should track row/column dependencies, not be fully dense
    expected = np.ones((4, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.fallback
def test_iota_eye():
    """jnp.eye uses iota internally, triggers conservative fallback.

    TODO(iota): Add prop_zero_derivative handler for iota (constant output).
    TODO(dot_general): Also needs dot_general handler for eye @ x.
    Precise: eye matrix has no input dependency (constant), so eye @ x = x.
    """

    def f(x):
        # Multiply x by identity - should preserve diagonal structure
        return jnp.eye(3) @ x

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # TODO: Should be identity matrix (eye @ x = x)
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.fallback
def test_sort():
    """jnp.sort triggers conservative fallback.

    Precise: all outputs depend on all inputs (sorting is a global operation).
    """

    def f(x):
        return jnp.sort(x)

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # Conservative fallback is actually correct here
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_custom_jvp_relu():
    """jax.nn.relu uses custom_jvp but tracks element-wise dependencies.

    ReLU is element-wise: each output depends only on corresponding input.
    """

    def f(x):
        return jax.nn.relu(x)

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_custom_vjp_user_defined():
    """User-defined custom_vjp traces forward computation."""

    @jax.custom_vjp
    def my_square(x):
        return x**2

    def my_square_fwd(x):
        return my_square(x), x

    def my_square_bwd(res, g):
        x = res
        return (2 * x * g,)

    my_square.defvjp(my_square_fwd, my_square_bwd)

    def f(x):
        return my_square(x)

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.eye(3, dtype=int)  # Element-wise operation
    np.testing.assert_array_equal(result, expected)

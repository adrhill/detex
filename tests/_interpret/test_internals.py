"""Unit tests for internal propagation functions.

https://docs.jax.dev/en/latest/jaxpr.html
"""

import jax
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

    with pytest.raises(ValueError) as exc_info:
        prop_nested_jaxpr(eqn, env, const_vals)  # type: ignore[arg-type]

    assert "xla_call" in str(exc_info.value)
    assert "https://github.com/adrhill/asdex/issues" in str(exc_info.value)


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

    with pytest.raises(ValueError) as exc_info:
        prop_custom_call(eqn, env, const_vals)  # type: ignore[arg-type]

    assert "custom_vjp_call" in str(exc_info.value)
    assert "https://github.com/adrhill/asdex/issues" in str(exc_info.value)


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

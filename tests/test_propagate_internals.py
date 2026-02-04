"""Unit tests for internal propagation functions."""

import pytest
from jax._src.core import Primitive

from detex._propagate import _propagate_nested_jaxpr


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

    with pytest.raises(ValueError, match="has no 'jaxpr' parameter"):
        _propagate_nested_jaxpr(eqn, env)  # type: ignore[arg-type]


def test_nested_jaxpr_missing_param_error_message():
    """Error message includes primitive name and issue tracker URL."""
    eqn = FakeEqn("xla_call", params={})
    env = {}

    with pytest.raises(ValueError) as exc_info:
        _propagate_nested_jaxpr(eqn, env)  # type: ignore[arg-type]

    assert "xla_call" in str(exc_info.value)
    assert "https://github.com/adrhill/detex/issues" in str(exc_info.value)

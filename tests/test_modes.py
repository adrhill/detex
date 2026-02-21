"""Tests for mode validation and resolution helpers."""

import warnings

import numpy as np
import pytest

from asdex import jacobian, jacobian_coloring
from asdex.modes import (
    _assert_coloring_mode,
    _assert_hessian_mode,
    _assert_jacobian_mode,
)

# Invalid mode validation


def test_invalid_coloring_mode_raises():
    """_assert_coloring_mode raises ValueError on unknown input."""
    with pytest.raises(ValueError, match="Unknown coloring_mode 'invalid'"):
        _assert_coloring_mode("invalid")


def test_invalid_jacobian_mode_raises():
    """_assert_jacobian_mode raises ValueError on unknown input."""
    with pytest.raises(ValueError, match="Unknown ad_mode 'invalid'"):
        _assert_jacobian_mode("invalid")


def test_invalid_hessian_mode_raises():
    """_assert_hessian_mode raises ValueError on unknown input."""
    with pytest.raises(ValueError, match="Unknown ad_mode 'invalid'"):
        _assert_hessian_mode("invalid")


# coloring_mode ignored warning


@pytest.mark.jacobian
def test_coloring_mode_ignored_warns(recwarn):
    """Jacobian warns when coloring_mode is set but colored_pattern is provided."""

    def f(x):
        return x**2

    cp = jacobian_coloring(f, (3,))
    x = np.array([1.0, 2.0, 3.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        jacobian(f, colored_pattern=cp, coloring_mode="row")(x)

    msgs = [str(wi.message) for wi in w]
    assert any("coloring_mode is ignored" in m for m in msgs)

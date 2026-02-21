"""Tests for mode validation and resolution helpers."""

import pytest

from asdex.modes import (
    _assert_hessian_mode,
    _assert_jacobian_mode,
)

# Invalid mode validation


def test_invalid_jacobian_mode_raises():
    """_assert_jacobian_mode raises ValueError on unknown input."""
    with pytest.raises(ValueError, match="Unknown mode 'invalid'"):
        _assert_jacobian_mode("invalid")


def test_invalid_hessian_mode_raises():
    """_assert_hessian_mode raises ValueError on unknown input."""
    with pytest.raises(ValueError, match="Unknown mode 'invalid'"):
        _assert_hessian_mode("invalid")

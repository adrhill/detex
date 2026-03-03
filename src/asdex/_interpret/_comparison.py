"""Propagation rules for comparison primitives (lt, le, gt, ge).

Comparisons are piecewise constant (zero derivative).
When value bounds prove the result is always True or always False,
the result is stored as a const value
so that ``select_n`` can pick the correct branch.
"""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    ConstVals,
    Deps,
    ValueBounds,
    atom_numel,
    atom_shape,
    atom_value_bounds,
    empty_index_sets,
    propagate_const_binary,
)


def _get_bounds(
    eqn: JaxprEqn,
    const_vals: ConstVals,
    value_bounds: ValueBounds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Get (lo1, hi1, lo2, hi2) for both inputs, or None if unavailable."""
    if eqn.outvars[0] in const_vals:
        return None
    b1 = atom_value_bounds(eqn.invars[0], const_vals, value_bounds)
    b2 = atom_value_bounds(eqn.invars[1], const_vals, value_bounds)
    if b1 is None or b2 is None:
        return None
    return (*b1, *b2)


def _set_const(eqn: JaxprEqn, const_vals: ConstVals, value: bool) -> None:
    """Store a constant boolean result."""
    const_vals[eqn.outvars[0]] = np.full(atom_shape(eqn.outvars[0]), value)


def prop_lt(
    eqn: JaxprEqn, deps: Deps, const_vals: ConstVals, value_bounds: ValueBounds
) -> None:
    """Less-than comparison with bounds resolution.

    Always true when ``hi(a) < lo(b)``.
    Always false when ``lo(a) >= hi(b)``.
    """
    deps[eqn.outvars[0]] = empty_index_sets(atom_numel(eqn.outvars[0]))
    propagate_const_binary(eqn, const_vals, np.less)
    bounds = _get_bounds(eqn, const_vals, value_bounds)
    if bounds is not None:
        lo1, hi1, lo2, hi2 = bounds
        if np.all(hi1 < lo2):
            _set_const(eqn, const_vals, True)
        elif np.all(lo1 >= hi2):
            _set_const(eqn, const_vals, False)


def prop_le(
    eqn: JaxprEqn, deps: Deps, const_vals: ConstVals, value_bounds: ValueBounds
) -> None:
    """Less-or-equal comparison with bounds resolution.

    Always true when ``hi(a) <= lo(b)``.
    Always false when ``lo(a) > hi(b)``.
    """
    deps[eqn.outvars[0]] = empty_index_sets(atom_numel(eqn.outvars[0]))
    propagate_const_binary(eqn, const_vals, np.less_equal)
    bounds = _get_bounds(eqn, const_vals, value_bounds)
    if bounds is not None:
        lo1, hi1, lo2, hi2 = bounds
        if np.all(hi1 <= lo2):
            _set_const(eqn, const_vals, True)
        elif np.all(lo1 > hi2):
            _set_const(eqn, const_vals, False)


def prop_gt(
    eqn: JaxprEqn, deps: Deps, const_vals: ConstVals, value_bounds: ValueBounds
) -> None:
    """Greater-than comparison with bounds resolution.

    Always true when ``lo(a) > hi(b)``.
    Always false when ``hi(a) <= lo(b)``.
    """
    deps[eqn.outvars[0]] = empty_index_sets(atom_numel(eqn.outvars[0]))
    propagate_const_binary(eqn, const_vals, np.greater)
    bounds = _get_bounds(eqn, const_vals, value_bounds)
    if bounds is not None:
        lo1, hi1, lo2, hi2 = bounds
        if np.all(lo1 > hi2):
            _set_const(eqn, const_vals, True)
        elif np.all(hi1 <= lo2):
            _set_const(eqn, const_vals, False)


def prop_ge(
    eqn: JaxprEqn, deps: Deps, const_vals: ConstVals, value_bounds: ValueBounds
) -> None:
    """Greater-or-equal comparison with bounds resolution.

    Always true when ``lo(a) >= hi(b)``.
    Always false when ``hi(a) < lo(b)``.
    """
    deps[eqn.outvars[0]] = empty_index_sets(atom_numel(eqn.outvars[0]))
    propagate_const_binary(eqn, const_vals, np.greater_equal)
    bounds = _get_bounds(eqn, const_vals, value_bounds)
    if bounds is not None:
        lo1, hi1, lo2, hi2 = bounds
        if np.all(lo1 >= hi2):
            _set_const(eqn, const_vals, True)
        elif np.all(hi1 < lo2):
            _set_const(eqn, const_vals, False)

"""Propagation rule for element-wise division."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    StateBounds,
    StateConsts,
    StateIndices,
    atom_const_val,
    atom_shape,
    atom_value_bounds,
    empty_index_set,
    numel,
    propagate_const_binary,
)
from ._elementwise import _binary_elementwise


def prop_div(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Division is element-wise with a special case for known zero numerators.

    Like other binary element-wise ops,
    each output depends on the corresponding elements from both inputs.
    However, since d(0 / y)/dy = 0,
    output positions where the numerator is a known constant zero
    have no dependency on the inputs.

    Example: z = [0, x] / [y, y]
        Input index sets: [{}, {1}], [{2}, {3}]
        Output index sets: [{}, {1, 3}]  (first cleared by known zero numerator)

    Jaxpr:
        invars[0]: numerator
        invars[1]: denominator
    """
    _binary_elementwise(eqn, state_indices)
    propagate_const_binary(eqn, state_consts, np.divide)

    # Zero-skipping: d(0/y)/dy = 0.
    num_val = atom_const_val(eqn.invars[0], state_consts)
    if num_val is not None:
        out_shape = atom_shape(eqn.outvars[0])
        out_size = numel(out_shape)
        ndim = len(out_shape)

        # Broadcast numerator to output shape.
        num_shape = atom_shape(eqn.invars[0])
        arr = (
            np.asarray(num_val).reshape(num_shape) if num_shape else np.asarray(num_val)
        )
        pad = ndim - len(num_shape)
        padded_shape = (1,) * pad + num_shape
        num_broadcast = np.broadcast_to(arr.reshape(padded_shape), out_shape).ravel()

        out_indices = state_indices[eqn.outvars[0]]
        for i in range(out_size):
            if num_broadcast[i] == 0:
                out_indices[i] = empty_index_set()

    # Bounds propagation via interval division.
    _propagate_bounds_div(eqn, state_consts, state_bounds)


def _propagate_bounds_div(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Propagate value bounds through ``div`` via interval arithmetic.

    Only propagates when divisor bounds have constant sign (no zero crossing),
    since division by an interval spanning zero is undefined.
    Uses ``floor_divide`` for integer dtypes and ``true_divide`` for floats.
    """
    in1_bounds = atom_value_bounds(eqn.invars[0], state_consts, state_bounds)
    in2_bounds = atom_value_bounds(eqn.invars[1], state_consts, state_bounds)
    if in1_bounds is None or in2_bounds is None:
        return

    lo1, hi1 = in1_bounds
    lo2, hi2 = in2_bounds

    # Skip if divisor bounds span zero.
    if not (np.all(lo2 > 0) or np.all(hi2 < 0)):
        return

    out_dtype = getattr(eqn.outvars[0].aval, "dtype", np.float64)
    divide = np.floor_divide if np.issubdtype(out_dtype, np.integer) else np.true_divide

    # All four endpoint combinations.
    c1 = divide(lo1, lo2)
    c2 = divide(lo1, hi2)
    c3 = divide(hi1, lo2)
    c4 = divide(hi1, hi2)

    lo = np.minimum(np.minimum(c1, c2), np.minimum(c3, c4))
    hi = np.maximum(np.maximum(c1, c2), np.maximum(c3, c4))
    state_bounds[eqn.outvars[0]] = (lo, hi)

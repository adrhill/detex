"""Propagation rule for element-wise multiplication."""

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


def prop_mul(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Multiplication is element-wise with a special case for known zeros.

    Like other binary element-wise ops,
    each output depends on the corresponding elements from both inputs.
    However, since d(0 * y)/dy = 0,
    output positions where either operand is a known constant zero
    have no dependency on the inputs.

    Example: z = x * [0, 1] where x = [a, b]
        Input index sets: [{0}, {1}], [{}, {}]
        Output index sets: [{}, {1}]  (first cleared by known zero)

    Jaxpr:
        invars[0]: first input array
        invars[1]: second input array
    """
    _binary_elementwise(eqn, state_indices)
    propagate_const_binary(eqn, state_consts, np.multiply)

    in1_val = atom_const_val(eqn.invars[0], state_consts)
    in2_val = atom_const_val(eqn.invars[1], state_consts)
    if in1_val is None and in2_val is None:
        return

    out_shape = atom_shape(eqn.outvars[0])
    out_size = numel(out_shape)
    ndim = len(out_shape)

    # Broadcast const values to the output shape,
    # respecting numpy broadcasting rules (size-1 dims expand).
    def _broadcast(
        val: np.ndarray | None, invar_shape: tuple[int, ...]
    ) -> np.ndarray | None:
        if val is None:
            return None
        arr = np.asarray(val).reshape(invar_shape) if invar_shape else np.asarray(val)
        # Left-pad with 1s to match output ndim.
        pad = ndim - len(invar_shape)
        padded_shape = (1,) * pad + invar_shape
        return np.broadcast_to(arr.reshape(padded_shape), out_shape).ravel()

    in1_val = _broadcast(in1_val, atom_shape(eqn.invars[0]))
    in2_val = _broadcast(in2_val, atom_shape(eqn.invars[1]))

    out_indices = state_indices[eqn.outvars[0]]
    for i in range(out_size):
        if (in1_val is not None and in1_val[i] == 0) or (
            in2_val is not None and in2_val[i] == 0
        ):
            out_indices[i] = empty_index_set()

    # Bounds propagation via interval multiplication.
    _propagate_bounds_mul(eqn, state_consts, state_bounds)


def _propagate_bounds_mul(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Propagate value bounds through ``mul`` via interval arithmetic.

    ``[a,b] * [c,d]`` → ``[min(ac,ad,bc,bd), max(ac,ad,bc,bd)]``.
    This handles all sign combinations correctly.
    """
    in1_bounds = atom_value_bounds(eqn.invars[0], state_consts, state_bounds)
    in2_bounds = atom_value_bounds(eqn.invars[1], state_consts, state_bounds)
    if in1_bounds is None or in2_bounds is None:
        return

    lo1, hi1 = in1_bounds
    lo2, hi2 = in2_bounds

    c1 = lo1 * lo2
    c2 = lo1 * hi2
    c3 = hi1 * lo2
    c4 = hi1 * hi2

    lo = np.minimum(np.minimum(c1, c2), np.minimum(c3, c4))
    hi = np.maximum(np.maximum(c1, c2), np.maximum(c3, c4))
    state_bounds[eqn.outvars[0]] = (lo, hi)

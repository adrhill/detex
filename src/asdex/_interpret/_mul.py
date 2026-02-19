"""Propagation rule for element-wise multiplication."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    ConstVals,
    Deps,
    atom_const_val,
    atom_shape,
    numel,
)
from ._elementwise import prop_binary_elementwise


def prop_mul(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
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
    prop_binary_elementwise(eqn, deps)

    in1_val = atom_const_val(eqn.invars[0], const_vals)
    in2_val = atom_const_val(eqn.invars[1], const_vals)
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

    out_indices = deps[eqn.outvars[0]]
    for i in range(out_size):
        if (in1_val is not None and in1_val[i] == 0) or (
            in2_val is not None and in2_val[i] == 0
        ):
            out_indices[i] = set()

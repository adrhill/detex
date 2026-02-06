"""Propagation rule for gather operations."""

import numpy as np
from jax._src.core import JaxprEqn, Literal, Var

from ._commons import ConstVals, Deps, IndexSets, atom_numel, index_sets, union_all


def _get_static_indices(
    indices_atom, deps: Deps, const_vals: ConstVals
) -> np.ndarray | None:
    """Get concrete index values if they're static, otherwise return None.

    For Literal indices, returns the value directly.
    For Var indices that are tracked in const_vals, returns the tracked value.
    Otherwise returns None, triggering conservative fallback.
    """
    if isinstance(indices_atom, Literal):
        return np.asarray(indices_atom.val)
    if isinstance(indices_atom, Var) and indices_atom in const_vals:
        return np.asarray(const_vals[indices_atom])
    return None


def prop_gather(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Gather extracts slices from operand at positions specified by indices.

    For static indices (Literal or tracked const), each output element depends
    on specific input elements determined by the index values. For dynamic
    (traced) indices, we fall back to conservative.

    For simple 1D gather: out[i] = operand[indices[i]]
        Each output depends on exactly one input.
    The Jacobian is a selection/permutation matrix.

    Example: x = [a, b, c], indices = [2, 0, 1], y = x[indices] = [c, a, b]
        Input deps:  [{0}, {1}, {2}]
        Output deps: [{2}, {0}, {1}]  (permuted by index array)

    Example with dynamic indices: x[traced_indices]
        Cannot determine which inputs each output depends on.
        Conservative: all outputs depend on all inputs.

    Jaxpr:
        invars[0]: operand array to gather from
        invars[1]: indices specifying gather positions
        dimension_numbers: GatherDimensionNumbers specifying axis mapping
        slice_sizes: size of slice to extract at each index
    """
    operand_indices = index_sets(deps, eqn.invars[0])
    indices_atom = eqn.invars[1]

    # Check if we can get static index values
    concrete_indices = _get_static_indices(indices_atom, deps, const_vals)

    if concrete_indices is not None:
        # Static indices - compute precise mapping
        dim_nums = eqn.params["dimension_numbers"]
        slice_sizes = eqn.params["slice_sizes"]

        operand_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))

        # Handle simple 1D case: operand is 1D, indices are 1D, slice_size is (1,)
        # This covers x[indices] where x is 1D and indices is a 1D array
        if (
            len(operand_shape) == 1
            and len(slice_sizes) == 1
            and slice_sizes[0] == 1
            and dim_nums.offset_dims == ()
            and dim_nums.collapsed_slice_dims == (0,)
            and dim_nums.start_index_map == (0,)
        ):
            # Simple 1D gather: out[i] = operand[indices[i]]
            flat_indices = concrete_indices.flatten()
            out_indices: IndexSets = []
            for idx in flat_indices:
                idx_int = int(idx)
                if 0 <= idx_int < len(operand_indices):
                    out_indices.append(operand_indices[idx_int].copy())
                else:
                    # Out of bounds - no dependency (will be filled with default)
                    out_indices.append(set())
            deps[eqn.outvars[0]] = out_indices
            return

        # For more complex gather patterns, fall through to conservative

    # Dynamic indices or complex gather - conservative fallback
    all_deps = union_all(operand_indices)
    out_size = atom_numel(eqn.outvars[0])
    deps[eqn.outvars[0]] = [all_deps.copy() for _ in range(out_size)]

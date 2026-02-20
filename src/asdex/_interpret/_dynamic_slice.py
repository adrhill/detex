"""Propagation rules for dynamic_slice and dynamic_update_slice."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    ConstVals,
    Deps,
    IndexSet,
    atom_const_val,
    atom_shape,
    check_no_index_sets,
    conservative_indices,
    copy_index_sets,
    index_sets,
    numel,
    permute_indices,
)


def _resolve_starts(
    eqn: JaxprEqn, start_offset: int, const_vals: ConstVals
) -> list[int] | None:
    """Try to resolve start indices as static ints.

    Returns None if any start depends on runtime values.
    """
    starts: list[int] = []
    for atom in eqn.invars[start_offset:]:
        val = atom_const_val(atom, const_vals)
        if val is None:
            return None
        starts.append(int(val.flat[0]))
    return starts


def prop_dynamic_slice(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """dynamic_slice extracts a sub-array at a potentially dynamic offset.

    With static start indices, each output element maps to exactly one input element.
    With dynamic starts, falls back to conservative.

    For static starts s and slice_sizes sz:
        out[i₀, i₁, ...] = in[s₀ + i₀, s₁ + i₁, ...]
    The Jacobian is a selection matrix with exactly one 1 per row.

    Example: x = [a, b, c, d, e], dynamic_slice(x, [1], [3]) = [b, c, d]
        Input deps:  [{0}, {1}, {2}, {3}, {4}]
        Output deps: [{1}, {2}, {3}]

    Jaxpr:
        invars: [operand, *start_indices]
        params: slice_sizes

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html
    """
    operand = eqn.invars[0]
    in_indices = index_sets(deps, operand)
    slice_sizes = eqn.params["slice_sizes"]

    # TODO: include start index sets in output dependencies.
    for start_atom in eqn.invars[1:]:
        check_no_index_sets(deps, start_atom, eqn.primitive.name)

    starts = _resolve_starts(eqn, 1, const_vals)
    if starts is None:
        deps[eqn.outvars[0]] = conservative_indices(in_indices, numel(slice_sizes))
        return

    # Build flat index map: for each output element, which input element it reads
    in_shape = atom_shape(operand)
    out_coords = np.indices(tuple(slice_sizes))
    in_coords = tuple(s + out_coords[d] for d, s in enumerate(starts))
    permutation_map = np.ravel_multi_index(in_coords, in_shape).ravel()

    deps[eqn.outvars[0]] = permute_indices(in_indices, permutation_map)


def prop_dynamic_update_slice(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """dynamic_update_slice overwrites a sub-array at a potentially dynamic offset.

    With static start indices, updated positions get update deps,
    the rest keep operand deps.
    With dynamic starts, falls back to conservative.

    For static starts s and update shape u_shape:
        out[i] = update[i - s]  if s ≤ i < s + u_shape
        out[i] = operand[i]     otherwise

    Example: operand = [a, b, c, d], update = [X, Y], start = [1]
        out = [a, X, Y, d]
        Output deps: [{0}, {upd_0}, {upd_1}, {3}]

    Jaxpr:
        invars: [operand, update, *start_indices]
        params: (none relevant)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_update_slice.html
    """
    operand = eqn.invars[0]
    update = eqn.invars[1]
    operand_indices = index_sets(deps, operand)
    upd_indices = index_sets(deps, update)
    operand_shape = atom_shape(operand)
    upd_shape = atom_shape(update)

    # TODO: include start index sets in output dependencies.
    for start_atom in eqn.invars[2:]:
        check_no_index_sets(deps, start_atom, eqn.primitive.name)

    starts = _resolve_starts(eqn, 2, const_vals)
    if starts is None:
        deps[eqn.outvars[0]] = conservative_indices(
            operand_indices + upd_indices, numel(operand_shape)
        )
        return

    # Start with operand deps, then overwrite the update region
    out_indices: list[IndexSet] = copy_index_sets(operand_indices)

    # Map each update element to its flat position in the operand
    upd_coords = np.indices(upd_shape)
    op_coords = tuple(s + upd_coords[d] for d, s in enumerate(starts))
    permutation_map = np.ravel_multi_index(op_coords, operand_shape).ravel()

    for upd_flat, op_flat in enumerate(permutation_map):
        out_indices[op_flat] = upd_indices[upd_flat].copy()

    deps[eqn.outvars[0]] = out_indices

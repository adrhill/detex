"""Propagation rules for dynamic_slice and dynamic_update_slice."""

from jax._src.core import JaxprEqn

from ._commons import (
    ConstVals,
    Deps,
    IndexSets,
    atom_const_val,
    index_sets,
    numel,
    row_strides,
    union_all,
)


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
    """
    operand = eqn.invars[0]
    start_atoms = eqn.invars[1:]
    slice_sizes = eqn.params["slice_sizes"]
    in_indices = index_sets(deps, operand)

    # Try to resolve start indices statically
    starts: list[int | None] = []
    for atom in start_atoms:
        val = atom_const_val(atom, const_vals)
        if val is not None:
            starts.append(int(val.flat[0]))
        else:
            starts.append(None)

    # If any start index is dynamic, fall back to conservative
    if any(s is None for s in starts):
        all_deps = union_all(in_indices)
        out_size = numel(slice_sizes)
        deps[eqn.outvars[0]] = [all_deps.copy() for _ in range(out_size)]
        return

    # All starts are static: precise element mapping (like prop_slice)
    static_starts: list[int] = starts  # type: ignore[assignment]
    in_shape = tuple(getattr(operand.aval, "shape", ()))
    out_shape = tuple(slice_sizes)

    in_strides = row_strides(in_shape)
    out_strides = row_strides(out_shape)
    out_size = numel(out_shape)

    out_indices: IndexSets = []
    for out_flat in range(out_size):
        # Convert flat output index to output coordinates
        out_coord = []
        remaining = out_flat
        for s in out_strides:
            out_coord.append(remaining // s)
            remaining %= s

        # Map to input: in_coord[d] = start[d] + out_coord[d]
        in_flat = sum(
            (static_starts[d] + out_coord[d]) * in_strides[d]
            for d in range(len(out_shape))
        )
        out_indices.append(in_indices[in_flat].copy())

    deps[eqn.outvars[0]] = out_indices


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
    """
    operand = eqn.invars[0]
    update = eqn.invars[1]
    start_atoms = eqn.invars[2:]

    op_indices = index_sets(deps, operand)
    upd_indices = index_sets(deps, update)

    # Try to resolve start indices statically
    starts: list[int | None] = []
    for atom in start_atoms:
        val = atom_const_val(atom, const_vals)
        if val is not None:
            starts.append(int(val.flat[0]))
        else:
            starts.append(None)

    op_shape = tuple(getattr(operand.aval, "shape", ()))
    upd_shape = tuple(getattr(update.aval, "shape", ()))

    # If any start index is dynamic, fall back to conservative
    if any(s is None for s in starts):
        all_deps = union_all(op_indices + upd_indices)
        out_size = numel(op_shape)
        deps[eqn.outvars[0]] = [all_deps.copy() for _ in range(out_size)]
        return

    # All starts are static: precise mapping
    static_starts: list[int] = starts  # type: ignore[assignment]
    op_strides = row_strides(op_shape)
    upd_strides = row_strides(upd_shape)
    out_size = numel(op_shape)

    out_indices: IndexSets = []
    for out_flat in range(out_size):
        # Convert flat index to multi-dim coordinates in operand shape
        coord = []
        remaining = out_flat
        for s in op_strides:
            coord.append(remaining // s)
            remaining %= s

        # Check if this coordinate falls within the update region
        in_update = True
        upd_coord = []
        for d in range(len(op_shape)):
            offset = coord[d] - static_starts[d]
            if 0 <= offset < upd_shape[d]:
                upd_coord.append(offset)
            else:
                in_update = False
                break

        if in_update:
            upd_flat = sum(upd_coord[d] * upd_strides[d] for d in range(len(upd_shape)))
            out_indices.append(upd_indices[upd_flat].copy())
        else:
            out_indices.append(op_indices[out_flat].copy())

    deps[eqn.outvars[0]] = out_indices

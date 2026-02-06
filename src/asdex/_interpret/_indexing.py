"""Propagation rules for indexing and shape manipulation operations."""

import numpy as np
from jax._src.core import JaxprEqn, Literal, Var

from ._commons import (
    ConstVals,
    Deps,
    IndexSets,
    atom_numel,
    index_sets,
    numel,
    row_strides,
    union_all,
)


def prop_slice(eqn: JaxprEqn, deps: Deps) -> None:
    """Slicing extracts a contiguous (possibly strided) subarray.
    Each output element maps to exactly one input element,
    so dependencies pass through unchanged.

    For slice with start indices s, strides t:
        out[i, j, ...] = in[s₀ + i·t₀, s₁ + j·t₁, ...]
    The Jacobian is a selection matrix with exactly one 1 per row.

    Example: x = [a, b, c, d, e], y = x[1:4:2] = [b, d]
        Input deps:  [{0}, {1}, {2}, {3}, {4}]
        Output deps: [{1}, {3}]  (indices 1 and 3 from input)

    Jaxpr:
        invars[0]: input array
        start_indices: tuple of start indices per dimension
        limit_indices: tuple of end indices per dimension
        strides: tuple of step sizes per dimension (default: all 1s)
    """
    in_indices = index_sets(deps, eqn.invars[0])
    start = eqn.params["start_indices"]
    limit = eqn.params["limit_indices"]
    slice_strides = eqn.params.get("strides") or tuple(1 for _ in start)

    in_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))
    out_shape = tuple(
        (limit[d] - start[d] + slice_strides[d] - 1) // slice_strides[d]
        for d in range(len(start))
    )

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

        # Map to input coordinates: in_coord[d] = start[d] + out_coord[d] * slice_strides[d]
        in_flat = sum(
            (start[d] + out_coord[d] * slice_strides[d]) * in_strides[d]
            for d in range(len(start))
        )
        out_indices.append(in_indices[in_flat].copy())

    deps[eqn.outvars[0]] = out_indices


def prop_squeeze(eqn: JaxprEqn, deps: Deps) -> None:
    """Squeeze removes dimensions of size 1 without changing the data.
    Since it's a reshape with the same number of elements,
    dependencies pass through unchanged in flat order.

    For input shape (2, 1, 3) with squeeze on dim 1:
        out[i, k] = in[i, 0, k]
    The Jacobian is the identity matrix (permuted).

    Example: x.shape = (2, 1), y = squeeze(x) with shape (2,)
        Input deps:  [{0}, {1}]
        Output deps: [{0}, {1}]

    Jaxpr:
        invars[0]: input array
        dimensions: axes to squeeze (must have size 1)
    """
    deps[eqn.outvars[0]] = index_sets(deps, eqn.invars[0])


def prop_broadcast_in_dim(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Broadcast replicates input elements across new or expanded dimensions.
    Each output element depends on exactly one input element,
    determined by projecting output coordinates onto input dimensions.

    For broadcast_dimensions mapping input dim i → output dim d[i]:
        out[..., j, ...] = in[..., j mod in_shape[i], ...]
    Size-1 input dims are implicitly broadcast (all outputs read index 0).

    Also tracks const values: if input is a Literal or known const,
    the output value is also recorded for use in gather/scatter handlers.

    Example: x.shape = (3,), y = broadcast(x, shape=(2, 3), dims=(1,))
        Input deps:  [{0}, {1}, {2}]
        Output deps: [{0}, {1}, {2}, {0}, {1}, {2}]  (repeated per row)

    Jaxpr:
        invars[0]: input array
        shape: target output shape
        broadcast_dimensions: maps input dim i to output dim
    """
    in_atom = eqn.invars[0]
    in_indices = index_sets(deps, in_atom)
    out_shape = eqn.params["shape"]
    broadcast_dims = eqn.params["broadcast_dimensions"]
    out_size = numel(out_shape)

    # Track const values for static index tracking in gather/scatter
    out_var = eqn.outvars[0]

    def broadcast_const(in_val: np.ndarray) -> np.ndarray:
        """Broadcast a const value according to broadcast_in_dim semantics."""
        # broadcast_dimensions tells us which output dims come from input dims
        # Create an intermediate shape that can be broadcast to out_shape
        in_shape = in_val.shape or (1,)
        intermediate_shape = [1] * len(out_shape)
        for i, out_dim in enumerate(broadcast_dims):
            intermediate_shape[out_dim] = in_shape[i] if i < len(in_shape) else 1
        reshaped = np.reshape(in_val, intermediate_shape)
        return np.broadcast_to(reshaped, out_shape)

    if isinstance(in_atom, Literal):
        in_val = np.asarray(in_atom.val)
        const_vals[out_var] = broadcast_const(in_val)
    elif isinstance(in_atom, Var) and in_atom in const_vals:
        in_val = np.asarray(const_vals[in_atom])
        const_vals[out_var] = broadcast_const(in_val)

    # Scalar case: single input dependency applies to all outputs
    if len(in_indices) == 1:
        deps[out_var] = [in_indices[0].copy() for _ in range(out_size)]
        return

    in_shape = tuple(getattr(in_atom.aval, "shape", ()))
    in_strides = row_strides(in_shape)
    out_strides = row_strides(out_shape)

    out_indices: IndexSets = []
    for out_flat in range(out_size):
        # Convert flat output index to output coordinates
        out_coord = []
        remaining = out_flat
        for s in out_strides:
            out_coord.append(remaining // s)
            remaining %= s

        # Map to input coordinates using broadcast_dimensions.
        # broadcast_dims[i] = which output dim corresponds to input dim i.
        # Size-1 input dims are replicated: input (3,1) -> output (3,2) means
        # out[i,0] and out[i,1] both come from in[i,0], so we clamp to 0.
        in_flat = sum(
            (out_coord[broadcast_dims[i]] if in_shape[i] > 1 else 0) * in_strides[i]
            for i in range(len(in_shape))
        )
        out_indices.append(in_indices[in_flat].copy())

    deps[out_var] = out_indices


def prop_concatenate(eqn: JaxprEqn, deps: Deps) -> None:
    """Concatenate joins arrays along a specified axis.
    Each output element comes from exactly one input element.

    For concat([A, B], axis=0): output = [A; B] (vertical stack).
    For concat([A, B], axis=1): output = [A | B] (horizontal stack).
    The Jacobian is a permuted identity matrix.

    Example: concat([[a,b], [c,d]], axis=0) → [a,b,c,d]
        Input deps:  [{0}, {1}], [{2}, {3}]
        Output deps: [{0}, {1}, {2}, {3}]

    Jaxpr:
        invars: list of input arrays to concatenate
        dimension: axis along which to concatenate
    """
    dim = eqn.params["dimension"]

    # Concat along dim 0: flat arrays are contiguous, just append
    if dim == 0:
        out_indices: IndexSets = []
        for invar in eqn.invars:
            out_indices.extend(index_sets(deps, invar))
        deps[eqn.outvars[0]] = out_indices
        return

    # Inner dimension: output coord along `dim` determines which input it's from.
    # E.g., concat([A(2x1), B(2x1)], dim=1) -> C(2x2): C[i,0] from A, C[i,1] from B.
    out_shape = tuple(getattr(eqn.outvars[0].aval, "shape", ()))
    in_shapes = [tuple(getattr(iv.aval, "shape", ())) for iv in eqn.invars]
    in_dim_sizes = [s[dim] for s in in_shapes]

    # dim_offsets[i] = starting position of input i along concat dimension
    dim_offsets = [sum(in_dim_sizes[:i]) for i in range(len(in_dim_sizes) + 1)]

    out_strides = row_strides(out_shape)
    all_in_indices = [index_sets(deps, iv) for iv in eqn.invars]
    all_in_strides = [row_strides(s) for s in in_shapes]

    out_indices = []
    for out_flat in range(numel(out_shape)):
        out_coord = []
        remaining = out_flat
        for s in out_strides:
            out_coord.append(remaining // s)
            remaining %= s

        # Find which input owns this position along the concat dimension
        pos_along_dim = out_coord[dim]
        for i in range(len(eqn.invars)):
            if dim_offsets[i] <= pos_along_dim < dim_offsets[i + 1]:
                in_coord = list(out_coord)
                in_coord[dim] = pos_along_dim - dim_offsets[i]
                in_flat = sum(
                    c * s for c, s in zip(in_coord, all_in_strides[i], strict=True)
                )
                out_indices.append(all_in_indices[i][in_flat].copy())
                break

    deps[eqn.outvars[0]] = out_indices


def prop_reshape(eqn: JaxprEqn, deps: Deps) -> None:
    """Reshape changes array shape without changing data or element count.
    Dependencies pass through unchanged in row-major (C) order.
    The Jacobian is the identity matrix.

    Example: reshape([a,b,c,d], (2,2)) → [[a,b],[c,d]]
        Input deps:  [{0}, {1}, {2}, {3}]
        Output deps: [{0}, {1}, {2}, {3}]

    Jaxpr:
        invars[0]: input array
        new_sizes: target shape
        dimensions: optional axis permutation before reshape
    """
    in_indices = index_sets(deps, eqn.invars[0])
    out_size = numel(tuple(getattr(eqn.outvars[0].aval, "shape", ())))
    if len(in_indices) == out_size:
        deps[eqn.outvars[0]] = in_indices
    else:
        # TODO: Investigate when size mismatch occurs and handle precisely.
        # Conservative fallback: union all input dependencies.
        all_deps = union_all(in_indices)
        deps[eqn.outvars[0]] = [all_deps.copy() for _ in range(out_size)]


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


def prop_scatter(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Scatter writes updates into operand at positions specified by indices.

    For static indices (Literal or tracked const), we can precisely track which
    output positions come from the original operand vs which receive scattered
    updates. For dynamic indices, we fall back to conservative.

    For scatter (replace): out[indices[i]] = updates[i], else out[j] = operand[j]
        Output positions NOT in indices: depend on corresponding operand element.
        Output positions in indices: depend on corresponding updates element.

    For scatter-add: out[indices[i]] = operand[indices[i]] + updates[i]
        Output positions in indices: depend on BOTH operand AND updates.

    Example: arr = [a, b, c], arr.at[1].set(x) = [a, x, c]
        operand deps:  [{0}, {1}, {2}]  (from arr)
        updates deps:  [{3}]             (from x, assuming x is input index 3)
        Output deps:   [{0}, {3}, {2}]   (index 1 replaced by x)

    Example with dynamic indices: arr.at[traced_idx].set(x)
        Cannot determine which position receives the update.
        Conservative: all outputs depend on all inputs.

    Jaxpr:
        invars[0]: operand array (base)
        invars[1]: indices specifying scatter positions
        invars[2]: updates to scatter
        dimension_numbers: ScatterDimensionNumbers specifying axis mapping
        update_jaxpr: optional, defines combination function (e.g., add for scatter-add)
    """
    operand_indices = index_sets(deps, eqn.invars[0])
    indices_atom = eqn.invars[1]
    updates_indices = index_sets(deps, eqn.invars[2])

    # Check if we can get static index values
    concrete_indices = _get_static_indices(indices_atom, deps, const_vals)

    if concrete_indices is not None:
        # Static indices - track which positions get updates
        dim_nums = eqn.params["dimension_numbers"]
        update_jaxpr = eqn.params.get("update_jaxpr")
        is_scatter_add = update_jaxpr is not None

        operand_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))
        out_shape = tuple(getattr(eqn.outvars[0].aval, "shape", ()))
        out_size = numel(out_shape)

        # Handle simple 1D case: operand is 1D, indices specify positions
        # This covers arr.at[idx].set(val) and arr.at[indices].set(vals)
        if (
            len(operand_shape) == 1
            and dim_nums.update_window_dims == ()
            and dim_nums.inserted_window_dims == (0,)
            and dim_nums.scatter_dims_to_operand_dims == (0,)
        ):
            # Build mapping from output position to list of update indices
            # (multiple updates can target the same position in scatter-add)
            flat_indices = concrete_indices.flatten()
            scatter_positions: dict[int, list[int]] = {}
            for update_idx, pos in enumerate(flat_indices):
                pos_int = int(pos)
                if 0 <= pos_int < out_size:
                    if pos_int not in scatter_positions:
                        scatter_positions[pos_int] = []
                    scatter_positions[pos_int].append(update_idx)

            out_indices: IndexSets = []
            for out_pos in range(out_size):
                if out_pos in scatter_positions:
                    update_idx_list = scatter_positions[out_pos]
                    if is_scatter_add:
                        # scatter-add: depends on operand AND all updates at this position
                        combined = operand_indices[out_pos].copy()
                        for update_idx in update_idx_list:
                            if update_idx < len(updates_indices):
                                combined |= updates_indices[update_idx]
                            elif updates_indices:
                                combined |= updates_indices[0]
                        out_indices.append(combined)
                    else:
                        # scatter (replace): last update wins, depends only on that update
                        last_update_idx = update_idx_list[-1]
                        if last_update_idx < len(updates_indices):
                            out_indices.append(updates_indices[last_update_idx].copy())
                        elif updates_indices:
                            out_indices.append(updates_indices[0].copy())
                        else:
                            out_indices.append(set())
                else:
                    # Position not in scatter targets - keep operand dependency
                    out_indices.append(operand_indices[out_pos].copy())

            deps[eqn.outvars[0]] = out_indices
            return

        # For more complex scatter patterns, fall through to conservative

    # Dynamic indices or complex scatter - conservative fallback
    all_deps = union_all(operand_indices + updates_indices)
    out_size = atom_numel(eqn.outvars[0])
    deps[eqn.outvars[0]] = [all_deps.copy() for _ in range(out_size)]

"""Propagation rule for scatter operations."""

from jax._src.core import JaxprEqn

from ._commons import (
    ConstVals,
    Deps,
    IndexSets,
    atom_const_val,
    atom_numel,
    atom_shape,
    conservative_deps,
    index_sets,
    numel,
)


def prop_scatter(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Scatter writes updates into operand at positions given by scatter_indices.

    For static scatter_indices (Literal or tracked const),
    we can precisely track which output positions come from the original
    operand vs which receive scattered updates.
    For dynamic scatter_indices, we fall back to conservative.

    Precise path handles the simple 1D case
    (update_window_dims=(), inserted_window_dims=(0,),
    scatter_dims_to_operand_dims=(0,)).
    This is the pattern JAX emits for ``arr.at[idx].set(val)``.

    For scatter (replace): out[idx[i]] = updates[i], else out[j] = operand[j]
        Positions NOT in idx: depend on corresponding operand element.
        Positions in idx: depend on corresponding updates element.

    For scatter-add: out[idx[i]] = operand[idx[i]] + updates[i]
        Positions in idx: depend on BOTH operand AND updates.

    Example: arr = [a, b, c], arr.at[1].set(x) = [a, x, c]
        operand deps:  [{0}, {1}, {2}]  (from arr)
        updates deps:  [{3}]             (from x, assuming x is input index 3)
        Output deps:   [{0}, {3}, {2}]   (index 1 replaced by x)

    Example with dynamic scatter_indices: arr.at[traced_idx].set(x)
        Cannot determine which position receives the update.
        Conservative: all outputs depend on all inputs.

    Jaxpr:
        invars[0]: operand — base array
        invars[1]: scatter_indices — positions to scatter into
        invars[2]: updates — values to write
        dimension_numbers: ScatterDimensionNumbers specifying axis mapping
        update_jaxpr: combination function (e.g., add for scatter-add),
            absent for plain scatter (replace)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter.html
    """
    operand_indices = index_sets(deps, eqn.invars[0])
    indices_atom = eqn.invars[1]
    updates_indices = index_sets(deps, eqn.invars[2])

    # Check if we can get static index values
    concrete_indices = atom_const_val(indices_atom, const_vals)

    if concrete_indices is not None:
        # Static indices - track which positions get updates
        dim_nums = eqn.params["dimension_numbers"]
        update_jaxpr = eqn.params.get("update_jaxpr")
        is_scatter_add = update_jaxpr is not None

        operand_shape = atom_shape(eqn.invars[0])
        out_shape = atom_shape(eqn.outvars[0])
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
    deps[eqn.outvars[0]] = conservative_deps(
        operand_indices + updates_indices, atom_numel(eqn.outvars[0])
    )

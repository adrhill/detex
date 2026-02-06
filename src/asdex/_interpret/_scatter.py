"""Propagation rule for scatter operations."""

import numpy as np
from jax._src.core import JaxprEqn, Literal, Var

from ._commons import (
    ConstVals,
    Deps,
    IndexSets,
    atom_numel,
    index_sets,
    numel,
    union_all,
)


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

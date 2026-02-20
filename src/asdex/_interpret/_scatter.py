"""Propagation rule for scatter operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    ConstVals,
    Deps,
    IndexSets,
    atom_const_val,
    atom_numel,
    atom_shape,
    check_no_index_sets,
    conservative_indices,
    empty_index_set,
    index_sets,
    numel,
)


def prop_scatter(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Scatter writes updates into operand at positions given by scatter_indices.

    For static scatter_indices (Literal or tracked const),
    we can precisely track which output positions come from the original
    operand vs which receive scattered updates.
    For dynamic scatter_indices, we fall back to conservative.

    Three precise patterns are handled:

    1. **Batched scatter along dim 0**: each update row targets a different operand row.
       Pattern: ``inserted_window_dims=(0,)``, ``scatter_dims_to_operand_dims=(0,)``,
       ``update_window_dims=(1, ..., ndim-1)`` with matching trailing shapes.
       This is the pattern JAX emits for ``arr.at[indices].set(vals)``
       and the backward of ``features[indices]`` (scatter-add).

    2. **Full-window scatter along an arbitrary dim**: a single update slice
       is written at one position along an arbitrary dimension.
       Pattern: ``inserted_window_dims=(d,)``, ``scatter_dims_to_operand_dims=(d,)``,
       all update dims are window dims.
       This is the pattern JAX emits for ``arr.at[:, idx, :].set(val)``.

    3. **Multi-index scatter**: each update is a scalar written at a multi-dim coordinate.
       Pattern: ``update_window_dims=()``, all dims inserted, ``scatter_dims_to_operand_dims``
       matches ``inserted_window_dims``, indices shape is ``(N, ndim)``.
       This is the pattern JAX emits for ``mat.at[rows, cols].set(vals)``.

    For scatter (replace): out[idx[i]] = updates[i], else out[j] = operand[j]
        Positions NOT in idx: depend on corresponding operand element.
        Positions in idx: depend on corresponding updates element.

    For scatter-add/mul/min/max: out[idx[i]] = combine(operand[idx[i]], updates[i])
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
    # TODO: include scatter_indices index sets in output dependencies.
    check_no_index_sets(deps, indices_atom, eqn.primitive.name)
    updates_indices = index_sets(deps, eqn.invars[2])

    # Check if we can get static index values
    concrete_indices = atom_const_val(indices_atom, const_vals)

    if concrete_indices is not None:
        # Static indices - track which positions get updates
        dim_nums = eqn.params["dimension_numbers"]
        update_jaxpr = eqn.params.get("update_jaxpr")
        # All combine variants (add, mul, min, max) have the same dependency
        # structure: the output depends on both operand and updates.
        is_combine = update_jaxpr is not None

        operand_shape = atom_shape(eqn.invars[0])
        out_shape = atom_shape(eqn.outvars[0])
        out_size = numel(out_shape)

        # Pattern 1: Batched scatter along dim 0 with trailing window dims.
        # Covers both 1D (arr.at[idx].set(val))
        # and multi-D (arr.at[idx].set(vals) where arr is N-D).
        updates_shape = atom_shape(eqn.invars[2])
        n_update_window = len(dim_nums.update_window_dims)
        expected_window_dims = tuple(range(1, 1 + n_update_window))
        if (
            dim_nums.inserted_window_dims == (0,)
            and dim_nums.scatter_dims_to_operand_dims == (0,)
            and dim_nums.update_window_dims == expected_window_dims
            and updates_shape[1:] == operand_shape[1:]
        ):
            row_size = numel(operand_shape[1:]) if len(operand_shape) > 1 else 1
            n_rows = operand_shape[0]

            # Build mapping from output row to list of update row indices.
            flat_indices = concrete_indices.flatten()
            scatter_positions: dict[int, list[int]] = {}
            for update_row, pos in enumerate(flat_indices):
                pos_int = int(pos)
                if 0 <= pos_int < n_rows:
                    if pos_int not in scatter_positions:
                        scatter_positions[pos_int] = []
                    scatter_positions[pos_int].append(update_row)

            out_indices: IndexSets = []
            for out_row in range(n_rows):
                if out_row in scatter_positions:
                    update_row_list = scatter_positions[out_row]
                    for d in range(row_size):
                        out_flat = out_row * row_size + d
                        if is_combine:
                            combined = operand_indices[out_flat].copy()
                            for update_row in update_row_list:
                                u_flat = update_row * row_size + d
                                if u_flat < len(updates_indices):
                                    combined |= updates_indices[u_flat]
                            out_indices.append(combined)
                        else:
                            last_row = update_row_list[-1]
                            u_flat = last_row * row_size + d
                            if u_flat < len(updates_indices):
                                out_indices.append(updates_indices[u_flat].copy())
                            else:
                                out_indices.append(empty_index_set())
                else:
                    for d in range(row_size):
                        out_indices.append(
                            operand_indices[out_row * row_size + d].copy()
                        )

            deps[eqn.outvars[0]] = out_indices
            return

        # Pattern 2: Full-window scatter along an arbitrary single dimension.
        # All update dims are window dims (no batch dim in updates).
        #
        # Example: (4,4,4).at[:,3,:].set((4,4)):
        #   inserted_window_dims=(1,), scatter_dims_to_operand_dims=(1,),
        #   update_window_dims=(0,1)
        if (
            len(dim_nums.inserted_window_dims) == 1
            and dim_nums.scatter_dims_to_operand_dims == dim_nums.inserted_window_dims
            and not dim_nums.operand_batching_dims
            and dim_nums.update_window_dims == tuple(range(len(updates_shape)))
        ):
            scatter_dim = dim_nums.inserted_window_dims[0]
            ndim = len(operand_shape)

            # Validate: updates shape should be operand with scatter_dim removed.
            expected_updates = tuple(
                s for i, s in enumerate(operand_shape) if i != scatter_dim
            )
            if updates_shape == expected_updates:
                flat_idx = concrete_indices.flatten()
                target_set = {
                    int(k) for k in flat_idx if 0 <= int(k) < operand_shape[scatter_dim]
                }

                # Identify which output positions are scatter targets
                # and map them to the corresponding updates flat index.
                out_coords = np.indices(operand_shape)
                scatter_mask = np.isin(
                    out_coords[scatter_dim], list(target_set)
                ).ravel()
                window_coords = tuple(
                    out_coords[i] for i in range(ndim) if i != scatter_dim
                )
                updates_flat_map = np.ravel_multi_index(
                    window_coords, expected_updates
                ).ravel()

                out_indices = []
                for i in range(out_size):
                    if scatter_mask[i]:
                        u_flat = int(updates_flat_map[i])
                        if is_combine:
                            out_indices.append(
                                operand_indices[i].copy()
                                | updates_indices[u_flat].copy()
                            )
                        else:
                            out_indices.append(updates_indices[u_flat].copy())
                    else:
                        out_indices.append(operand_indices[i].copy())

                deps[eqn.outvars[0]] = out_indices
                return

        # Pattern 3: Multi-index scatter where each update is a scalar
        # written at a multi-dimensional coordinate.
        #
        # Example: mat.at[rows, cols].set(vals) with mat shape (3, 5):
        #   update_window_dims=(), inserted_window_dims=(0, 1),
        #   scatter_dims_to_operand_dims=(0, 1),
        #   scatter_indices shape (N, 2)
        ndim = len(operand_shape)
        if (
            dim_nums.update_window_dims == ()
            and dim_nums.inserted_window_dims == tuple(range(ndim))
            and dim_nums.scatter_dims_to_operand_dims == dim_nums.inserted_window_dims
            and concrete_indices.ndim == 2
            and concrete_indices.shape[1] == ndim
        ):
            # Build mapping from flat output position
            # to list of update indices that scatter into it.
            # Skip OOB indices (JAX drops them in FILL_OR_DROP mode).
            scatter_positions: dict[int, list[int]] = {}
            for update_idx in range(concrete_indices.shape[0]):
                coords = concrete_indices[update_idx]
                if any(
                    coords[d] < 0 or coords[d] >= operand_shape[d] for d in range(ndim)
                ):
                    continue
                flat_pos = int(np.ravel_multi_index(coords, operand_shape))
                if flat_pos not in scatter_positions:
                    scatter_positions[flat_pos] = []
                scatter_positions[flat_pos].append(update_idx)

            out_indices: IndexSets = []
            for i in range(out_size):
                if i in scatter_positions:
                    if is_combine:
                        # Union operand and all updates targeting this position.
                        combined = operand_indices[i].copy()
                        for u_idx in scatter_positions[i]:
                            combined |= updates_indices[u_idx]
                        out_indices.append(combined)
                    else:
                        # Replace: last-write-wins.
                        last_u = scatter_positions[i][-1]
                        out_indices.append(updates_indices[last_u].copy())
                else:
                    out_indices.append(operand_indices[i].copy())

            deps[eqn.outvars[0]] = out_indices
            return

        # For more complex scatter patterns, fall through to conservative

    # Dynamic indices or complex scatter - conservative fallback
    deps[eqn.outvars[0]] = conservative_indices(
        operand_indices + updates_indices, atom_numel(eqn.outvars[0])
    )

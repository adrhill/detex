"""Propagation rule for gather operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    ConstVals,
    Deps,
    atom_const_val,
    atom_numel,
    atom_shape,
    check_no_index_sets,
    conservative_indices,
    index_sets,
    permute_indices,
    position_map,
)


def prop_gather(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Gather extracts slices from operand at positions given by start_indices.

    For static start_indices (Literal or tracked const),
    each output element depends on specific input elements.
    For dynamic (traced) start_indices, falls back to conservative.

    Three precise patterns are handled:

    1. **Single-dim gather** (any dim ``d``):
       selects along exactly one dimension and keeps all others intact.
       ``collapsed_slice_dims=(d,)``, ``start_index_map=(d,)``,
       ``slice_sizes[d]==1``, other slice sizes match operand.
       This is the pattern JAX emits for ``x[indices]``, ``x[:, indices]``,
       ``jnp.take(x, indices, axis=d)``.

    2. **Multi-dim collapse**:
       all collapsed dims have ``slice_sizes==1``,
       ``start_index_map == collapsed_slice_dims``.
       This is the pattern JAX emits for ``x[row_idx, col_idx]``.

    The Jacobian is a selection/permutation matrix:
    each output element reads exactly one input element.

    Example: x = [a, b, c], idx = [2, 0, 1], y = x[idx] = [c, a, b]
        Input index sets:  [{0}, {1}, {2}]
        Output index sets: [{2}, {0}, {1}]  (permuted by index array)

    Example: x.shape = (3, 4), y = x[:, idx] where idx = [2, 0]
        Each output row selects columns 2 and 0 from the corresponding input row.

    Example with dynamic start_indices: x[traced_idx]
        Cannot determine which inputs each output depends on.
        Conservative: all outputs depend on all inputs.

    Jaxpr:
        invars[0]: operand — array to gather from
        invars[1]: start_indices — positions at which slices begin
        dimension_numbers: GatherDimensionNumbers specifying axis mapping
        slice_sizes: shape of each extracted slice (length = ndim(operand))

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html
    """
    operand_indices = index_sets(deps, eqn.invars[0])
    # TODO: include start_indices index sets in output dependencies.
    check_no_index_sets(deps, eqn.invars[1], eqn.primitive.name)
    concrete_indices = atom_const_val(eqn.invars[1], const_vals)

    if concrete_indices is not None:
        dim_nums = eqn.params["dimension_numbers"]
        slice_sizes = eqn.params["slice_sizes"]
        operand_shape = atom_shape(eqn.invars[0])

        # Guard: batching dims are not supported.
        if (
            hasattr(dim_nums, "operand_batching_dims")
            and dim_nums.operand_batching_dims
        ):
            pass  # fall through to conservative

        # Pattern 1 (single-dim) or Pattern 2 (multi-dim collapse).
        elif _try_single_dim_gather(
            eqn,
            deps,
            operand_indices,
            concrete_indices,
            dim_nums,
            slice_sizes,
            operand_shape,
        ) or _try_multi_dim_gather(
            eqn,
            deps,
            operand_indices,
            concrete_indices,
            dim_nums,
            slice_sizes,
            operand_shape,
        ):
            return

    # Conservative fallback: every output depends on every input.
    # Always correct, but marks the full Jacobian as dense.
    # Used when indices are dynamic or the gather pattern isn't one we handle.
    deps[eqn.outvars[0]] = conservative_indices(
        operand_indices, atom_numel(eqn.outvars[0])
    )


def _try_single_dim_gather(
    eqn: JaxprEqn,
    deps: Deps,
    operand_indices,
    concrete_indices: np.ndarray,
    dim_nums,
    slice_sizes,
    operand_shape: tuple[int, ...],
) -> bool:
    """Handle gather along exactly one dimension ``d``, keeping all others intact.

    Returns True if the pattern matched and index sets were written.
    """
    if len(dim_nums.collapsed_slice_dims) != 1:
        return False

    d = dim_nums.collapsed_slice_dims[0]

    if dim_nums.start_index_map != (d,):
        return False
    if slice_sizes[d] != 1:
        # JAX enforces this at trace time:
        # https://github.com/jax-ml/jax/blob/jax-v0.9.0.1/jax/_src/lax/slicing.py#L1961
        raise AssertionError(
            f"collapsed dim {d} has slice_size {slice_sizes[d]}, expected 1"
        )

    # All non-collapsed slice sizes must match the operand shape.
    for i, (ss, os) in enumerate(zip(slice_sizes, operand_shape, strict=True)):
        if i != d and ss != os:
            return False

    # Build a position map of the operand and index along dim d
    # to find which flat input position each output element reads from.
    op_pos = position_map(operand_shape)
    # Select the slices along dim d using the concrete index values.
    # np.take handles arbitrary dim selection cleanly.
    selected = np.take(op_pos, concrete_indices.flatten(), axis=d)

    # The output layout is: batch dims (from start_indices shape)
    # come first at positions NOT in offset_dims,
    # and the kept operand dims fill in at offset_dims positions.
    # We need to transpose `selected` so its axes match the output layout.
    #
    # `selected` currently has shape:
    #   operand dims before d + (n_indices,) + operand dims after d
    # where n_indices = len(concrete_indices.flatten()).
    #
    # The output has ndim = len(offset_dims) + ndim(start_indices_shape).
    # offset_dims tells us where the kept operand dims land in the output.
    # The remaining positions are batch dims from the index array.
    out_ndim = len(atom_shape(eqn.outvars[0]))
    offset_dims = dim_nums.offset_dims
    batch_positions = [i for i in range(out_ndim) if i not in offset_dims]

    # In `selected`, axis d is the index-batch axis
    # and the other axes are the kept operand dims (in order).
    # We need to build a permutation that moves them to match the output layout.
    n_kept = len(operand_shape) - 1  # dims kept from operand
    # `selected` axes: 0..d-1 are kept-before, d is batch, d+1..n_kept are kept-after.
    # Target: offset_dims positions get kept dims, batch_positions get batch dims.
    source_kept_axes = [i for i in range(n_kept + 1) if i != d]
    source_batch_axis = d

    # Build inverse permutation: for each output axis, which selected axis provides it.
    perm = [0] * out_ndim
    for out_ax in batch_positions:
        perm[out_ax] = source_batch_axis
    kept_iter = iter(source_kept_axes)
    for out_ax in offset_dims:
        perm[out_ax] = next(kept_iter)

    # Handle case where start_indices has more than 1 batch dim
    # by reshaping the batch axis into multiple axes.
    idx_shape = concrete_indices.shape
    if len(idx_shape) > 1:
        # start_indices is (batch..., index_vector_dim).
        # After flattening for np.take, the batch axis in `selected`
        # has size prod(idx_shape[:-1]).
        # Reshape it back to the original batch shape before transposing.
        batch_shape = idx_shape[:-1] if idx_shape[-1] == 1 else idx_shape
        new_shape = list(selected.shape)
        new_shape[d : d + 1] = list(batch_shape)
        selected = selected.reshape(new_shape)

        # Recompute permutation for the expanded batch dims.
        out_ndim = len(atom_shape(eqn.outvars[0]))
        offset_dims = dim_nums.offset_dims
        batch_positions = [i for i in range(out_ndim) if i not in offset_dims]
        n_batch = len(batch_positions)
        # selected axes: kept-before (0..d-1), batch (d..d+n_batch-1), kept-after
        source_batch_axes = list(range(d, d + n_batch))
        source_kept_axes = list(range(d)) + list(
            range(d + n_batch, d + n_batch + n_kept - d)
        )
        perm = [0] * out_ndim
        for out_ax, src_ax in zip(batch_positions, source_batch_axes, strict=True):
            perm[out_ax] = src_ax
        kept_iter = iter(source_kept_axes)
        for out_ax in offset_dims:
            perm[out_ax] = next(kept_iter)

    permutation_map = selected.transpose(perm).flatten()
    deps[eqn.outvars[0]] = permute_indices(operand_indices, permutation_map)
    return True


def _try_multi_dim_gather(
    eqn: JaxprEqn,
    deps: Deps,
    operand_indices,
    concrete_indices: np.ndarray,
    dim_nums,
    slice_sizes,
    operand_shape: tuple[int, ...],
) -> bool:
    """Handle multi-dim collapse gather (e.g. ``x[row_idx, col_idx]``).

    All collapsed dims must have ``slice_sizes==1``
    and ``start_index_map == collapsed_slice_dims``.

    Returns True if the pattern matched and index sets were written.
    """
    collapsed = dim_nums.collapsed_slice_dims
    if len(collapsed) < 2:
        return False
    if dim_nums.start_index_map != collapsed:
        return False

    # JAX enforces collapsed dims have slice_size == 1:
    # https://github.com/jax-ml/jax/blob/jax-v0.9.0.1/jax/_src/lax/slicing.py#L1961
    if not all(slice_sizes[d] == 1 for d in collapsed):
        raise AssertionError(
            f"collapsed dims {collapsed} have slice_sizes {[slice_sizes[d] for d in collapsed]}"
        )

    # Non-collapsed slice sizes must match the operand shape.
    for i, (ss, os) in enumerate(zip(slice_sizes, operand_shape, strict=True)):
        if i not in collapsed and ss != os:
            return False

    # start_indices has shape (N, len(collapsed))
    # where each row is a coordinate into the collapsed dims.
    # Use ravel_multi_index to convert to flat operand positions.
    collapsed_shape = tuple(operand_shape[d] for d in collapsed)

    # concrete_indices may be 1D (single index) or 2D (N x n_collapsed).
    if concrete_indices.ndim == 1:
        # Single multi-dim index.
        coords = tuple(concrete_indices[i] for i in range(len(collapsed)))
    else:
        # (N, n_collapsed) — each column is one dimension's indices.
        coords = tuple(concrete_indices[:, i] for i in range(len(collapsed)))

    flat_operand_positions = np.ravel_multi_index(coords, collapsed_shape)

    # If there are non-collapsed dims, each gathered position
    # expands to a full slice over those dims.
    non_collapsed = [i for i in range(len(operand_shape)) if i not in collapsed]
    if non_collapsed:
        # Build a position map and index into it.
        op_pos = position_map(operand_shape)

        # Transpose so collapsed dims come first, then reshape to merge them.
        # This lets us index with flat_operand_positions directly.
        perm_to_front = list(collapsed) + non_collapsed
        op_pos_t = op_pos.transpose(perm_to_front)
        # Shape: (collapsed_shape..., non_collapsed_shape...)
        n_collapsed_elems = int(np.prod(collapsed_shape))
        non_collapsed_shape = tuple(operand_shape[d] for d in non_collapsed)
        op_pos_flat = op_pos_t.reshape(n_collapsed_elems, -1)
        # Index with the flat positions to get (N, non_collapsed_flat) map.
        selected = op_pos_flat[flat_operand_positions.flatten()]
        # selected shape: (N, prod(non_collapsed_shape))

        # Output layout: batch dims from index array come at positions
        # NOT in offset_dims; kept dims fill offset_dims.
        out_shape = atom_shape(eqn.outvars[0])
        out_ndim = len(out_shape)
        offset_dims = dim_nums.offset_dims

        if offset_dims:
            # Reshape selected to (batch_shape..., non_collapsed_shape...)
            idx_shape = (
                concrete_indices.shape[:-1]
                if concrete_indices.ndim > 1
                else (len(flat_operand_positions),)
            )
            intermediate_shape = tuple(idx_shape) + non_collapsed_shape
            selected = selected.reshape(intermediate_shape)

            # Transpose to match output layout.
            batch_positions = [i for i in range(out_ndim) if i not in offset_dims]
            n_batch = len(batch_positions)
            n_offset = len(offset_dims)
            # selected axes: 0..n_batch-1 are batch, n_batch..end are kept.
            perm = [0] * out_ndim
            for target, src in zip(batch_positions, range(n_batch), strict=True):
                perm[target] = src
            for target, src in zip(
                offset_dims, range(n_batch, n_batch + n_offset), strict=True
            ):
                perm[target] = src
            selected = selected.transpose(perm)

        permutation_map = selected.flatten()
    else:
        # All dims collapsed — output is just a selection of scalar elements.
        permutation_map = flat_operand_positions.flatten()

    deps[eqn.outvars[0]] = permute_indices(operand_indices, permutation_map)
    return True

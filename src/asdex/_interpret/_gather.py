"""Propagation rule for gather operations."""

from jax._src.core import JaxprEqn

from ._commons import (
    ConstVals,
    Deps,
    atom_const_val,
    atom_numel,
    atom_shape,
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

    Precise path handles the common pattern where the gather selects
    along dim 0 and keeps all other dims intact
    (collapsed_slice_dims=(0,), start_index_map=(0,), slice_sizes[0]=1).
    This is the pattern JAX emits for ``x[indices]`` on any-rank operand.

    For simple 1D gather: out[i] = operand[start_indices[i]]
        Each output depends on exactly one input.
    The Jacobian is a selection/permutation matrix.

    Example: x = [a, b, c], idx = [2, 0, 1], y = x[idx] = [c, a, b]
        Input deps:  [{0}, {1}, {2}]
        Output deps: [{2}, {0}, {1}]  (permuted by index array)

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
    concrete_indices = atom_const_val(eqn.invars[1], const_vals)

    if concrete_indices is not None:
        dim_nums = eqn.params["dimension_numbers"]
        slice_sizes = eqn.params["slice_sizes"]
        operand_shape = atom_shape(eqn.invars[0])

        # We can compute a precise mapping when the gather selects along
        # exactly one dimension (dim 0) and keeps all others intact.
        # This is the pattern JAX emits for x[indices] on any-rank operand.
        #
        # Unsupported patterns (e.g. gathering along a non-leading dim, or taking partial slices)
        # fall through to the conservative fallback, which is always correct but imprecise.
        if (
            dim_nums.collapsed_slice_dims == (0,)
            and dim_nums.start_index_map == (0,)
            and slice_sizes[0] == 1
            and slice_sizes[1:] == operand_shape[1:]
        ):
            # Map each output element to the flat operand element it reads from.
            # Fancy-indexing a position map does this without manual stride math:
            # position_map[k] gives the flat indices of the k-th slice along dim 0.
            permutation_map = position_map(operand_shape)[
                concrete_indices.flatten()
            ].flatten()
            deps[eqn.outvars[0]] = permute_indices(operand_indices, permutation_map)
            return

    # Conservative fallback: every output depends on every input.
    # Always correct (never misses a dependency), but marks the full Jacobian as dense.
    # Used when indices are dynamic or the gather pattern isn't one we handle precisely.
    deps[eqn.outvars[0]] = conservative_indices(
        operand_indices, atom_numel(eqn.outvars[0])
    )

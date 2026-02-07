"""Propagation rule for gather operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    ConstVals,
    Deps,
    atom_const_val,
    atom_numel,
    index_sets,
    numel,
    union_all,
)


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
    concrete_indices = atom_const_val(eqn.invars[1], const_vals)

    if concrete_indices is not None:
        dim_nums = eqn.params["dimension_numbers"]
        slice_sizes = eqn.params["slice_sizes"]
        operand_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))

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
            # Fancy-indexing an iota array does this without manual stride math:
            # iota[k] gives the flat indices of the k-th slice along dim 0.
            iota = np.arange(numel(operand_shape)).reshape(operand_shape)
            flat_map = iota[concrete_indices.flatten()].flatten()
            deps[eqn.outvars[0]] = [operand_indices[i].copy() for i in flat_map]
            return

    # Conservative fallback: every output depends on every input.
    # Always correct (never misses a dependency), but marks the full Jacobian as dense.
    # Used when indices are dynamic or the gather pattern isn't one we handle precisely.
    all_deps = union_all(operand_indices)
    out_size = atom_numel(eqn.outvars[0])
    deps[eqn.outvars[0]] = [all_deps.copy() for _ in range(out_size)]

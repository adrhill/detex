"""Propagation rule for sort.

Sorting along one dimension mixes elements within slices along that dimension,
but not across other dimensions.
"""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    Deps,
    IndexSets,
    atom_numel,
    atom_shape,
    index_sets,
    numel,
    union_all,
)


def prop_sort(eqn: JaxprEqn, deps: Deps) -> None:
    """Sort reorders elements along one dimension.

    Each output element depends on all input elements in its slice
    along the sort dimension,
    since any input could end up at any position after sorting.

    For input shape (*batch, n) sorted along the last axis:
        out[*b, j] depends on all in[*b, :]

    For multiple operands (multi-key sort via ``lax.sort``),
    the permutation is determined by all key inputs jointly,
    so deps from all operands in the same slice are unioned.

    Example: y = sort(x, dimension=1) where x.shape = (2, 3)
        Input deps:  [{0}, {1}, {2}, {3}, {4}, {5}]
        Output deps: [{0, 1, 2}, {0, 1, 2}, {0, 1, 2},
                      {3, 4, 5}, {3, 4, 5}, {3, 4, 5}]

    Jaxpr:
        invars: one or more operand arrays (all same shape)
        dimension: axis along which to sort
        is_stable: whether sort is stable (irrelevant for sparsity)
        num_keys: number of key operands (irrelevant for sparsity)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.sort.html
    """
    dimension = eqn.params["dimension"]
    in_shape = atom_shape(eqn.invars[0])
    ndim = len(in_shape)

    # Canonicalize negative dimension
    if dimension < 0:
        dimension += ndim

    kept_dims = [d for d in range(ndim) if d != dimension]
    batch_shape = tuple(in_shape[d] for d in kept_dims)
    n_batches = numel(batch_shape)

    # Collect and union input deps from all operands,
    # grouped by their non-sort coordinates.
    group_deps: list[set[int]] = [set() for _ in range(n_batches)]

    for invar in eqn.invars:
        in_indices = index_sets(deps, invar)
        if not kept_dims:
            # 1D: single group containing all elements
            group_deps[0] |= union_all(in_indices)
        else:
            in_coords = np.indices(in_shape)
            batch_coords = tuple(in_coords[d] for d in kept_dims)
            group_map = np.ravel_multi_index(batch_coords, batch_shape).ravel()
            for in_flat, elem_deps in enumerate(in_indices):
                group_deps[group_map[in_flat]] |= elem_deps

    # Each output element gets the deps of its batch group.
    # Output shape matches input shape for each operand.
    for outvar in eqn.outvars:
        out_size = atom_numel(outvar)
        if not kept_dims:
            # 1D: all outputs share the single group
            out_indices: IndexSets = [group_deps[0].copy() for _ in range(out_size)]
        else:
            out_coords = np.indices(in_shape)
            batch_coords = tuple(out_coords[d] for d in kept_dims)
            group_map = np.ravel_multi_index(batch_coords, batch_shape).ravel()
            out_indices = [group_deps[group_map[i]].copy() for i in range(out_size)]
        deps[outvar] = out_indices

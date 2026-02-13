"""Propagation rule for sort.

Sorting along one dimension mixes elements within slices along that dimension,
but not across other dimensions.
"""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    Deps,
    atom_numel,
    atom_shape,
    index_sets,
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
    total = atom_numel(eqn.invars[0])
    n = in_shape[dimension]

    # Group flat indices by batch coordinates (all dims except sort dim).
    # After moveaxis + reshape, groups[b] holds the flat indices for batch b.
    groups = np.moveaxis(np.arange(total).reshape(in_shape), dimension, -1).reshape(
        -1, n
    )

    # Union deps from all operands within each batch group.
    group_deps: list[set[int]] = [set() for _ in range(len(groups))]
    for invar in eqn.invars:
        in_indices = index_sets(deps, invar)
        for b, flat_indices in enumerate(groups):
            for idx in flat_indices:
                group_deps[b] |= in_indices[idx]

    # Each output element gets the deps of its batch group.
    for outvar in eqn.outvars:
        out = [set[int]()] * total
        for b, flat_indices in enumerate(groups):
            for idx in flat_indices:
                out[idx] = group_deps[b].copy()
        deps[outvar] = out

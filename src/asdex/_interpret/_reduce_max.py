"""Propagation rule for reduce_max operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    Deps,
    IndexSets,
    atom_shape,
    index_sets,
    numel,
    union_all,
)


def prop_reduce_max(eqn: JaxprEqn, deps: Deps) -> None:
    """Max reduction selects one element per reduction group.

    Any input in the group could be the maximum,
    so sparsity-wise this is identical to reduce_sum:
    each output depends on all inputs that reduce into it.

    Full reduction (all axes):
        out = max(x)  →  out depends on all inputs
    Partial reduction along axis k:
        out[i] = max_j x[i, j]  →  out[i] depends on all x[i, :]

    Example: y = max(x, axis=1) where x.shape = (2, 3)
        Input deps:  [{0}, {1}, {2}, {3}, {4}, {5}]
        Output deps: [{0, 1, 2}, {3, 4, 5}]  (one set per row)

    Jaxpr:
        invars[0]: input array
        axes: tuple of axes to reduce

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_max.html
    """
    in_indices = index_sets(deps, eqn.invars[0])
    axes = eqn.params.get("axes", ())
    in_shape = atom_shape(eqn.invars[0])

    # Full reduction: single output depends on all inputs
    if not axes or len(axes) == len(in_shape):
        deps[eqn.outvars[0]] = [union_all(in_indices)]
        return

    # Partial reduction: group input elements by their non-reduced coordinates.
    # Build a flat map from each input element to its output group,
    # then union input deps into the corresponding output set.
    kept_dims = [d for d in range(len(in_shape)) if d not in axes]
    out_shape = tuple(in_shape[d] for d in kept_dims)
    out_size = numel(out_shape)

    # For each input element, project to output coordinates (drop reduced dims)
    in_coords = np.indices(in_shape)
    out_coords = tuple(in_coords[d] for d in kept_dims)
    group_map = np.ravel_multi_index(out_coords, out_shape).ravel()

    out_indices: IndexSets = [set() for _ in range(out_size)]
    for in_flat, elem_deps in enumerate(in_indices):
        out_indices[group_map[in_flat]] |= elem_deps

    deps[eqn.outvars[0]] = out_indices

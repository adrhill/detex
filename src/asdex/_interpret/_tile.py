"""Propagation rule for tile operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import Deps, IndexSets, atom_shape, index_sets


def prop_tile(eqn: JaxprEqn, deps: Deps) -> None:
    """Tile repeats an array along each dimension.

    Each output element depends on exactly one input element
    via modular indexing: ``out[i, j, ...] = in[i % s0, j % s1, ...]``.

    The Jacobian has exactly one 1 per row.

    Example: x = [a, b], tile(x, reps=(2,))
        Input deps:  [{0}, {1}]
        Output deps: [{0}, {1}, {0}, {1}]

    Jaxpr:
        invars[0]: input array
        reps: tuple of repetition counts per dimension

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.tile.html
    """
    in_indices = index_sets(deps, eqn.invars[0])
    in_shape = atom_shape(eqn.invars[0])
    reps = eqn.params["reps"]

    out_shape = tuple(s * r for s, r in zip(in_shape, reps, strict=True))
    out_size = int(np.prod(out_shape))

    # Build output coordinates and map back to input via modular arithmetic.
    out_coords = np.indices(out_shape)  # (ndim, *out_shape)
    in_coords = tuple(out_coords[d] % in_shape[d] for d in range(len(in_shape)))
    in_flat = np.ravel_multi_index(in_coords, in_shape).ravel()

    out_indices: IndexSets = [in_indices[in_flat[i]].copy() for i in range(out_size)]
    deps[eqn.outvars[0]] = out_indices

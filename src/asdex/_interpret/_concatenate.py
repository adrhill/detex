"""Propagation rule for concatenate operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import Deps, IndexSets, atom_shape, index_sets


def prop_concatenate(eqn: JaxprEqn, deps: Deps) -> None:
    """Concatenate joins arrays along a specified axis.

    Each output element comes from exactly one input element.

    For concat([A, B], axis=0): output = [A; B] (vertical stack).
    For concat([A, B], axis=1): output = [A | B] (horizontal stack).
    The Jacobian is a permuted identity matrix.

    Example: concat([[a,b], [c,d]], axis=0) → [a,b,c,d]
        Input deps:  [{0}, {1}], [{2}, {3}]
        Output deps: [{0}, {1}, {2}, {3}]

    Jaxpr:
        invars: list of input arrays to concatenate
        dimension: axis along which to concatenate

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.concatenate.html
    """
    dim = eqn.params["dimension"]

    # Pool every input's flat deps into one list.
    # For each input, build a shaped array whose values are positions in that pool.
    # np.concatenate on these index arrays mirrors the real op's element shuffling,
    # giving a flat mapping from each output element to the pool position it came from.
    all_deps: IndexSets = []
    index_arrays = []
    for invar in eqn.invars:
        in_deps = index_sets(deps, invar)
        offset = len(all_deps)
        all_deps.extend(in_deps)
        shape = atom_shape(invar)
        index_arrays.append(np.arange(offset, offset + len(in_deps)).reshape(shape))

    mapping = np.concatenate(index_arrays, axis=dim).ravel()
    deps[eqn.outvars[0]] = [all_deps[i] for i in mapping]

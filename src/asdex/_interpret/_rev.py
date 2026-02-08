"""Propagation rule for rev (reverse) operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    Deps,
    IndexSets,
    atom_shape,
    index_sets,
)


def prop_rev(eqn: JaxprEqn, deps: Deps) -> None:
    """Rev reverses an array along specified dimensions.

    Each output element maps to exactly one input element
    by flipping coordinates along the reversed dimensions.
    The Jacobian is a permutation matrix.

    For dimensions d in reversed_dims,
    output[..., i_d, ...] = input[..., (shape[d]-1-i_d), ...].

    Example: x = [a, b, c], rev(x, dimensions=[0])
        Input deps:  [{0}, {1}, {2}]
        Output deps: [{2}, {1}, {0}]

    Jaxpr:
        invars[0]: input array
        dimensions: sequence of ints specifying which axes to reverse

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.rev.html
    """
    in_indices = index_sets(deps, eqn.invars[0])
    in_shape = atom_shape(eqn.invars[0])
    dimensions = eqn.params["dimensions"]

    out_size = int(np.prod(in_shape))
    perm = np.flip(np.arange(out_size).reshape(in_shape), axis=dimensions).ravel()

    out_indices: IndexSets = [in_indices[perm[i]].copy() for i in range(out_size)]

    deps[eqn.outvars[0]] = out_indices

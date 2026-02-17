"""Propagation rule for rev (reverse) operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    Deps,
    atom_shape,
    index_sets,
    permute_indices,
    position_map,
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

    permutation_map = np.flip(position_map(in_shape), axis=dimensions).ravel()

    deps[eqn.outvars[0]] = permute_indices(in_indices, permutation_map)

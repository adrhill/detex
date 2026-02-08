"""Propagation rule for rev (reverse) operations."""

from jax._src.core import JaxprEqn

from ._commons import (
    Deps,
    IndexSets,
    atom_shape,
    flat_to_coords,
    index_sets,
    numel,
    row_strides,
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
    ndim = len(in_shape)

    in_strides = row_strides(in_shape)
    out_size = numel(in_shape)

    out_indices: IndexSets = []
    for out_flat in range(out_size):
        out_coord = flat_to_coords(out_flat, in_strides)

        # Flip coordinates along reversed dimensions.
        in_flat = sum(
            (in_shape[d] - 1 - out_coord[d] if d in dimensions else out_coord[d])
            * in_strides[d]
            for d in range(ndim)
        )

        out_indices.append(in_indices[in_flat].copy())

    deps[eqn.outvars[0]] = out_indices

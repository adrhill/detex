"""Propagation rule for transpose operations."""

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


def prop_transpose(eqn: JaxprEqn, deps: Deps) -> None:
    """Transpose permutes the dimensions of an array.

    Each output element maps to exactly one input element
    via the inverse permutation of coordinates.
    The Jacobian is a permutation matrix.

    For permutation perm, output[i0, i1, ...] = input[inv(i0), inv(i1), ...],
    where inv_perm[perm[d]] = d.

    Example: x = [[a, b, c], [d, e, f]], transpose(x, (1, 0))
        Input deps:  [{0}, {1}, {2}, {3}, {4}, {5}]
        Output deps: [{0}, {3}, {1}, {4}, {2}, {5}]

    Jaxpr:
        invars[0]: input array
        permutation: tuple of ints specifying the dimension order

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.transpose.html
    """
    in_indices = index_sets(deps, eqn.invars[0])
    in_shape = atom_shape(eqn.invars[0])
    permutation = eqn.params["permutation"]
    ndim = len(in_shape)

    # Compute output shape: out_shape[d] = in_shape[permutation[d]].
    out_shape = tuple(in_shape[permutation[d]] for d in range(ndim))

    # Build inverse permutation: inv_perm[perm[d]] = d.
    inv_perm = [0] * ndim
    for d in range(ndim):
        inv_perm[permutation[d]] = d

    in_strides = row_strides(in_shape)
    out_strides = row_strides(out_shape)
    out_size = numel(out_shape)

    out_indices: IndexSets = []
    for out_flat in range(out_size):
        out_coord = flat_to_coords(out_flat, out_strides)

        # Map output coordinate back to input coordinate via inverse permutation.
        in_flat = sum(out_coord[inv_perm[d]] * in_strides[d] for d in range(ndim))

        out_indices.append(in_indices[in_flat].copy())

    deps[eqn.outvars[0]] = out_indices

"""Propagation rule for transpose operations."""

from jax._src.core import JaxprEqn

from ._commons import Deps, atom_shape, index_sets, permute_indices, position_map


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

    permutation_map = position_map(in_shape).transpose(permutation).ravel()

    deps[eqn.outvars[0]] = permute_indices(in_indices, permutation_map)

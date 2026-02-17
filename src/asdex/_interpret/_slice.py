"""Propagation rule for slice operations."""

from jax._src.core import JaxprEqn

from ._commons import Deps, atom_shape, index_sets, permute_indices, position_map


def prop_slice(eqn: JaxprEqn, deps: Deps) -> None:
    """Slicing extracts a contiguous (possibly strided) subarray.

    Each output element maps to exactly one input element,
    so dependencies pass through unchanged.

    For slice with start indices s, strides t:
        out[i, j, ...] = in[s₀ + i·t₀, s₁ + j·t₁, ...]
    The Jacobian is a selection matrix with exactly one 1 per row.

    Example: x = [a, b, c, d, e], y = x[1:4:2] = [b, d]
        Input deps:  [{0}, {1}, {2}, {3}, {4}]
        Output deps: [{1}, {3}]  (indices 1 and 3 from input)

    Jaxpr:
        invars[0]: input array
        start_indices: tuple of start indices per dimension
        limit_indices: tuple of end indices per dimension
        strides: tuple of step sizes per dimension (default: all 1s)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.slice.html
    """
    in_indices = index_sets(deps, eqn.invars[0])
    start = eqn.params["start_indices"]
    limit = eqn.params["limit_indices"]
    slice_strides = eqn.params.get("strides") or tuple(1 for _ in start)

    in_shape = atom_shape(eqn.invars[0])
    slices = tuple(
        slice(start[d], limit[d], slice_strides[d]) for d in range(len(start))
    )
    permutation_map = position_map(in_shape)[slices].ravel()

    deps[eqn.outvars[0]] = permute_indices(in_indices, permutation_map)

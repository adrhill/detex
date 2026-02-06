"""Propagation rule for concatenate operations."""

from jax._src.core import JaxprEqn

from ._commons import Deps, IndexSets, index_sets, numel, row_strides


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
    """
    dim = eqn.params["dimension"]

    # Concat along dim 0: flat arrays are contiguous, just append
    if dim == 0:
        out_indices: IndexSets = []
        for invar in eqn.invars:
            out_indices.extend(index_sets(deps, invar))
        deps[eqn.outvars[0]] = out_indices
        return

    # Inner dimension: output coord along `dim` determines which input it's from.
    out_shape = tuple(getattr(eqn.outvars[0].aval, "shape", ()))
    in_shapes = [tuple(getattr(iv.aval, "shape", ())) for iv in eqn.invars]
    in_dim_sizes = [s[dim] for s in in_shapes]

    # dim_offsets[i] = starting position of input i along concat dimension
    dim_offsets = [sum(in_dim_sizes[:i]) for i in range(len(in_dim_sizes) + 1)]

    out_strides = row_strides(out_shape)
    all_in_indices = [index_sets(deps, iv) for iv in eqn.invars]
    all_in_strides = [row_strides(s) for s in in_shapes]

    out_indices = []
    for out_flat in range(numel(out_shape)):
        out_coord = []
        remaining = out_flat
        for s in out_strides:
            out_coord.append(remaining // s)
            remaining %= s

        # Find which input owns this position along the concat dimension
        pos_along_dim = out_coord[dim]
        for i in range(len(eqn.invars)):
            if dim_offsets[i] <= pos_along_dim < dim_offsets[i + 1]:
                in_coord = list(out_coord)
                in_coord[dim] = pos_along_dim - dim_offsets[i]
                in_flat = sum(
                    c * s for c, s in zip(in_coord, all_in_strides[i], strict=True)
                )
                out_indices.append(all_in_indices[i][in_flat].copy())
                break

    deps[eqn.outvars[0]] = out_indices

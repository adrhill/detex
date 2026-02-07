"""Propagation rules for indexing and shape manipulation operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    ConstVals,
    Deps,
    IndexSets,
    atom_const_val,
    index_sets,
    numel,
    row_strides,
)


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
    """
    in_indices = index_sets(deps, eqn.invars[0])
    start = eqn.params["start_indices"]
    limit = eqn.params["limit_indices"]
    slice_strides = eqn.params.get("strides") or tuple(1 for _ in start)

    in_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))
    out_shape = tuple(
        (limit[d] - start[d] + slice_strides[d] - 1) // slice_strides[d]
        for d in range(len(start))
    )

    in_strides = row_strides(in_shape)
    out_strides = row_strides(out_shape)
    out_size = numel(out_shape)

    out_indices: IndexSets = []
    for out_flat in range(out_size):
        # Convert flat output index to output coordinates
        out_coord = []
        remaining = out_flat
        for s in out_strides:
            out_coord.append(remaining // s)
            remaining %= s

        # Map to input coordinates: in_coord[d] = start[d] + out_coord[d] * slice_strides[d]
        in_flat = sum(
            (start[d] + out_coord[d] * slice_strides[d]) * in_strides[d]
            for d in range(len(start))
        )
        out_indices.append(in_indices[in_flat].copy())

    deps[eqn.outvars[0]] = out_indices


def prop_squeeze(eqn: JaxprEqn, deps: Deps) -> None:
    """Squeeze removes dimensions of size 1 without changing the data.
    Since it's a reshape with the same number of elements,
    dependencies pass through unchanged in flat order.

    For input shape (2, 1, 3) with squeeze on dim 1:
        out[i, k] = in[i, 0, k]
    The Jacobian is the identity matrix (permuted).

    Example: x.shape = (2, 1), y = squeeze(x) with shape (2,)
        Input deps:  [{0}, {1}]
        Output deps: [{0}, {1}]

    Jaxpr:
        invars[0]: input array
        dimensions: axes to squeeze (must have size 1)
    """
    deps[eqn.outvars[0]] = index_sets(deps, eqn.invars[0])


def prop_broadcast_in_dim(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Broadcast replicates input elements across new or expanded dimensions.
    Each output element depends on exactly one input element,
    determined by projecting output coordinates onto input dimensions.

    For broadcast_dimensions mapping input dim i → output dim d[i]:
        out[..., j, ...] = in[..., j mod in_shape[i], ...]
    Size-1 input dims are implicitly broadcast (all outputs read index 0).

    Also tracks const values: if input is a Literal or known const,
    the output value is also recorded for use in gather/scatter handlers.

    Example: x.shape = (3,), y = broadcast(x, shape=(2, 3), dims=(1,))
        Input deps:  [{0}, {1}, {2}]
        Output deps: [{0}, {1}, {2}, {0}, {1}, {2}]  (repeated per row)

    Jaxpr:
        invars[0]: input array
        shape: target output shape
        broadcast_dimensions: maps input dim i to output dim
    """

    in_atom = eqn.invars[0]
    in_indices = index_sets(deps, in_atom)
    out_shape = eqn.params["shape"]
    broadcast_dims = eqn.params["broadcast_dimensions"]
    out_var = eqn.outvars[0]

    # Gather/scatter handlers need concrete index arrays to resolve which input elements are accessed.
    # When the broadcast input is statically known (literal or traced from constants),
    # propagate its value so downstream handlers can use it
    # instead of falling back to conservative all-to-all dependencies.
    in_val = atom_const_val(in_atom, const_vals)
    if in_val is not None:
        intermediate_shape = [1] * len(out_shape)
        for i, out_dim in enumerate(broadcast_dims):
            intermediate_shape[out_dim] = (in_val.shape or (1,))[i]
        const_vals[out_var] = np.broadcast_to(
            np.reshape(in_val, intermediate_shape), out_shape
        )

    # Scalars have a single dependency set shared by all output elements,
    # so we can skip the coordinate mapping below and just replicate it.
    # Early return avoids building the np.indices grid for this common case.
    out_size = numel(out_shape)
    if len(in_indices) == 1:
        deps[out_var] = [in_indices[0].copy() for _ in range(out_size)]
        return

    # General case: map each output element back to the input element it reads.
    # np.indices gives all output coordinates.
    # We select the output dim corresponding to each input dim via broadcast_dims.
    # Size-1 input dims are broadcast (every output reads index 0), so we clamp to 0.
    in_shape = tuple(getattr(in_atom.aval, "shape", ()))
    out_coords = np.indices(out_shape)
    in_coords = tuple(
        out_coords[broadcast_dims[i]] if in_shape[i] > 1 else 0
        for i in range(len(in_shape))
    )
    flat_map = np.ravel_multi_index(in_coords, in_shape).ravel()

    deps[out_var] = [in_indices[j].copy() for j in flat_map]


def prop_reshape(eqn: JaxprEqn, deps: Deps) -> None:
    """Reshape changes array shape without changing data or element count.
    Dependencies pass through unchanged in row-major (C) order.
    The Jacobian is the identity matrix.

    Example: reshape([a,b,c,d], (2,2)) → [[a,b],[c,d]]
        Input deps:  [{0}, {1}, {2}, {3}]
        Output deps: [{0}, {1}, {2}, {3}]

    Jaxpr:
        invars[0]: input array
        new_sizes: target shape
        dimensions: optional axis permutation before reshape
    """
    in_indices = index_sets(deps, eqn.invars[0])
    out_size = numel(tuple(getattr(eqn.outvars[0].aval, "shape", ())))
    if len(in_indices) == out_size:
        deps[eqn.outvars[0]] = in_indices
    else:
        # TODO: Investigate when size mismatch occurs and handle precisely.
        # Conservative fallback: union all input dependencies.
        from ._commons import union_all

        all_deps = union_all(in_indices)
        deps[eqn.outvars[0]] = [all_deps.copy() for _ in range(out_size)]

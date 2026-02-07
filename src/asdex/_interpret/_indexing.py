"""Propagation rules for indexing and shape manipulation operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import Deps, IndexSets, index_sets, numel, row_strides


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

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.squeeze.html
    """
    deps[eqn.outvars[0]] = index_sets(deps, eqn.invars[0])


def prop_reshape(eqn: JaxprEqn, deps: Deps) -> None:
    """Reshape changes array shape without changing data or element count.
    Dependencies pass through unchanged in row-major (C) order.
    The Jacobian is the identity matrix.

    When ``dimensions`` is not None, JAX transposes the input axes
    before reshaping (e.g. ``ravel(order='F')`` emits ``dimensions=(1, 0)``).
    The permutation reorders which flat input each flat output reads from.

    Example: reshape([a,b,c,d], (2,2)) → [[a,b],[c,d]]
        Input deps:  [{0}, {1}, {2}, {3}]
        Output deps: [{0}, {1}, {2}, {3}]

    Example: reshape([[a,b,c],[d,e,f]], (6,), dimensions=(1,0))
        Transpose first → [[a,d],[b,e],[c,f]], then flatten → [a,d,b,e,c,f]
        Input deps:  [{0}, {1}, {2}, {3}, {4}, {5}]
        Output deps: [{0}, {3}, {1}, {4}, {2}, {5}]

    Jaxpr:
        invars[0]: operand — array to reshape
        new_sizes: target shape
        dimensions: optional axis permutation applied before reshape

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reshape.html
    """
    in_indices = index_sets(deps, eqn.invars[0])
    out_size = numel(tuple(getattr(eqn.outvars[0].aval, "shape", ())))
    if len(in_indices) != out_size:
        # Defensive fallback for unexpected size mismatches.
        from ._commons import union_all

        all_deps = union_all(in_indices)
        deps[eqn.outvars[0]] = [all_deps.copy() for _ in range(out_size)]
        return

    dimensions = eqn.params.get("dimensions")
    if dimensions is not None:
        # dimensions is a permutation applied before the reshape.
        # Build the flat index mapping: iota transposed then raveled
        # tells us which original flat index each output position reads.
        in_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))
        perm = (
            np.arange(len(in_indices)).reshape(in_shape).transpose(dimensions).ravel()
        )
        deps[eqn.outvars[0]] = [in_indices[j] for j in perm]
    else:
        deps[eqn.outvars[0]] = in_indices

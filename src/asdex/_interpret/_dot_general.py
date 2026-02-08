"""Propagation rule for dot_general (generalized matrix multiply)."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    Deps,
    IndexSets,
    atom_shape,
    index_sets,
    union_all,
)


def prop_dot_general(eqn: JaxprEqn, deps: Deps) -> None:
    """Dot_general contracts and batches two arrays.

    Each output element is a sum of products over the contracting dimensions,
    so it depends on a slice of lhs and a slice of rhs.
    Batch dimensions are preserved one-to-one.

    For out[b..., i..., j...] = sum_k lhs[b..., i..., k...] * rhs[b..., k..., j...]:
        deps(out[b,i,j]) = deps(lhs[b, i, :]) | deps(rhs[b, :, j])
    where b are batch dims, i are lhs-free dims, j are rhs-free dims,
    and k are contracting dims.

    Example: matrix multiply A(2,3) @ B(3,4) -> C(2,4)
        contracting: lhs_dim=1, rhs_dim=0
        out[i,j] depends on lhs[i,:] and rhs[:,j]
        Input lhs deps:  [{0},{1},{2},{3},{4},{5}]  (shape 2x3)
        Input rhs deps:  [{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17}]
        Output deps[0,0] = {0,1,2} | {6,10,14} = {0,1,2,6,10,14}

    Jaxpr:
        invars[0]: lhs array
        invars[1]: rhs array
        dimension_numbers: ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot_general.html
    """
    lhs_indices = index_sets(deps, eqn.invars[0])
    rhs_indices = index_sets(deps, eqn.invars[1])

    lhs_shape = atom_shape(eqn.invars[0])
    rhs_shape = atom_shape(eqn.invars[1])

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params[
        "dimension_numbers"
    ]
    lhs_contract = tuple(lhs_contract)
    rhs_contract = tuple(rhs_contract)
    lhs_batch = tuple(lhs_batch)
    rhs_batch = tuple(rhs_batch)

    lhs_free = tuple(
        d for d in range(len(lhs_shape)) if d not in lhs_contract and d not in lhs_batch
    )
    rhs_free = tuple(
        d for d in range(len(rhs_shape)) if d not in rhs_contract and d not in rhs_batch
    )

    # Output dim order: batch, lhs_free, rhs_free.
    out_shape = (
        tuple(lhs_shape[d] for d in lhs_batch)
        + tuple(lhs_shape[d] for d in lhs_free)
        + tuple(rhs_shape[d] for d in rhs_free)
    )

    if not out_shape:
        # Scalar output (e.g., vector dot product).
        deps[eqn.outvars[0]] = [union_all(lhs_indices) | union_all(rhs_indices)]
        return

    n_batch = len(lhs_batch)
    n_lhs_free = len(lhs_free)
    out_coords = np.indices(out_shape)
    out_size = int(np.prod(out_shape))

    # Map output coordinates to lhs/rhs fixed (non-contracting) coordinates.
    lhs_fixed = {}
    for i, d in enumerate(lhs_batch):
        lhs_fixed[d] = out_coords[i]
    for i, d in enumerate(lhs_free):
        lhs_fixed[d] = out_coords[n_batch + i]

    rhs_fixed = {}
    for i, d in enumerate(rhs_batch):
        rhs_fixed[d] = out_coords[i]
    for i, d in enumerate(rhs_free):
        rhs_fixed[d] = out_coords[n_batch + n_lhs_free + i]

    # Iterate over all contracting index combinations.
    # For each, compute lhs and rhs flat indices for every output element.
    contract_sizes = tuple(lhs_shape[d] for d in lhs_contract)
    n_contract = int(np.prod(contract_sizes)) if contract_sizes else 1
    contract_coords = (
        np.indices(contract_sizes).reshape(len(contract_sizes), -1)
        if contract_sizes
        else np.empty((0, 1), dtype=int)
    )

    out_indices: IndexSets = [set() for _ in range(out_size)]

    for c_idx in range(n_contract):
        lhs_coord = tuple(
            lhs_fixed[d]
            if d in lhs_fixed
            else np.full(out_shape, contract_coords[lhs_contract.index(d), c_idx])
            for d in range(len(lhs_shape))
        )
        rhs_coord = tuple(
            rhs_fixed[d]
            if d in rhs_fixed
            else np.full(out_shape, contract_coords[rhs_contract.index(d), c_idx])
            for d in range(len(rhs_shape))
        )
        lhs_flat = np.ravel_multi_index(lhs_coord, lhs_shape).ravel()
        rhs_flat = np.ravel_multi_index(rhs_coord, rhs_shape).ravel()

        for o in range(out_size):
            out_indices[o] |= lhs_indices[lhs_flat[o]]
            out_indices[o] |= rhs_indices[rhs_flat[o]]

    deps[eqn.outvars[0]] = out_indices

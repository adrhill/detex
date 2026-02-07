"""Propagation rule for select_n operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    ConstVals,
    Deps,
    IndexSets,
    atom_const_val,
    atom_numel,
    index_sets,
)


def prop_select_n(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """select_n(pred, x, y) selects x where pred is False, y where pred is True.

    This is element-wise: out[i] depends on pred[i], on_false[i], and on_true[i].
    The predicate has zero derivative,
    so only value-branch deps contribute to the sparsity pattern.

    For scalar predicate broadcast over array branches,
    the predicate deps are empty (no input dependency from a boolean).

    Jaxpr:
        invars[0]: predicate (boolean, scalar or array)
        invars[1:]: value branches (on_false, on_true, ...)
    """
    out_var = eqn.outvars[0]
    out_size = atom_numel(out_var)
    branches = eqn.invars[1:]  # value branches (pred is invars[0])

    # Element-wise union across value branches.
    # Predicate is boolean with zero derivative, so we skip it.
    branch_indices = [index_sets(deps, b) for b in branches]

    out_indices: IndexSets = []
    for i in range(out_size):
        merged: set[int] = set()
        for b_idx in branch_indices:
            # Handle scalar broadcast: scalar has 1 element, reuse index 0
            j = i if len(b_idx) > 1 else 0
            merged |= b_idx[j]
        out_indices.append(merged)

    deps[out_var] = out_indices

    # When all inputs are statically known, compute the concrete result
    # so const_vals tracking isn't broken by this op.
    pred = eqn.invars[0]
    on_false = eqn.invars[1]
    on_true = eqn.invars[2]

    pred_val = atom_const_val(pred, const_vals)
    on_false_val = atom_const_val(on_false, const_vals)
    on_true_val = atom_const_val(on_true, const_vals)

    if pred_val is not None and on_false_val is not None and on_true_val is not None:
        const_vals[out_var] = np.where(pred_val, on_true_val, on_false_val)

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
    union_all,
)


def prop_select_n(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """select_n(pred, x, y) selects x where pred is False, y where pred is True.

    For sparsity, this is conservative: each output depends on both alternatives
    since we don't know at trace time which path will be taken.

    However, for const value tracking (used in gather/scatter), if all inputs
    are tracked consts, we can compute the concrete output value.

    Jaxpr:
        invars[0]: boolean predicate array
        invars[1]: values selected when pred is False (on_false)
        invars[2]: values selected when pred is True (on_true)
    """
    # We can't know at trace time which branch the predicate selects,
    # so every output must conservatively depend on all inputs (pred + both branches).
    all_inputs: IndexSets = []
    for invar in eqn.invars:
        all_inputs.extend(index_sets(deps, invar))
    all_deps = union_all(all_inputs)
    for outvar in eqn.outvars:
        deps[outvar] = [all_deps.copy() for _ in range(atom_numel(outvar))]

    # When all three inputs are statically known, compute the concrete result
    # so const_vals tracking isn't broken by this op.
    pred = eqn.invars[0]
    on_false = eqn.invars[1]
    on_true = eqn.invars[2]
    out_var = eqn.outvars[0]

    pred_val = atom_const_val(pred, const_vals)
    on_false_val = atom_const_val(on_false, const_vals)
    on_true_val = atom_const_val(on_true, const_vals)

    if pred_val is not None and on_false_val is not None and on_true_val is not None:
        const_vals[out_var] = np.where(pred_val, on_true_val, on_false_val)

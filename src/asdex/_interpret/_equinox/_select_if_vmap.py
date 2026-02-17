"""Propagation rule for select_if_vmap (Equinox primitive)."""

import numpy as np
from jax._src.core import JaxprEqn

from .._commons import (
    ConstVals,
    Deps,
    IndexSets,
    atom_const_val,
    atom_numel,
    check_no_index_sets,
    index_sets,
)


def prop_select_if_vmap(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """select_if_vmap(pred, on_true, on_false) picks values element-wise.

    Equinox emits this when vmapping ``lax.cond``.
    Both branches are traced and the result is selected element-wise,
    identical to ``select_n`` with two cases.
    The predicate has zero derivative,
    so only the branch values contribute to the sparsity pattern.

    Jaxpr:
        invars[0]: pred (boolean, scalar or array)
        invars[1]: on_true (value when pred is True)
        invars[2]: on_false (value when pred is False)

    https://github.com/patrick-kidger/equinox/blob/main/equinox/internal/_loop/common.py
    """
    check_no_index_sets(deps, eqn.invars[0], eqn.primitive.name)

    out_var = eqn.outvars[0]
    out_size = atom_numel(out_var)
    on_true, on_false = eqn.invars[1], eqn.invars[2]
    true_indices = index_sets(deps, on_true)
    false_indices = index_sets(deps, on_false)

    out_indices: IndexSets = []
    for i in range(out_size):
        out_indices.append(true_indices[i] | false_indices[i])

    deps[out_var] = out_indices

    # Propagate concrete values when both branches are statically known.
    pred_val = atom_const_val(eqn.invars[0], const_vals)
    true_val = atom_const_val(on_true, const_vals)
    false_val = atom_const_val(on_false, const_vals)
    if pred_val is not None and true_val is not None and false_val is not None:
        const_vals[out_var] = np.where(pred_val, true_val, false_val)

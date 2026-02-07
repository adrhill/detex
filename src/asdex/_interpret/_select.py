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
    """select_n(which, *cases) picks case values element-wise.

    ``which`` is a boolean or integer selector (scalar or array).
    All cases must have identical shapes.
    The selector has zero derivative,
    so only value-case deps contribute to the sparsity pattern.

    Jaxpr:
        invars[0]: which (boolean or integer, scalar or array)
        invars[1:]: value cases (on_false, on_true, ...)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.select_n.html
    """
    out_var = eqn.outvars[0]
    out_size = atom_numel(out_var)
    cases = eqn.invars[1:]  # value cases (which is invars[0])

    # Element-wise union across value cases.
    # The selector has zero derivative, so we skip it.
    case_indices = [index_sets(deps, c) for c in cases]

    out_indices: IndexSets = []
    for i in range(out_size):
        merged: set[int] = set()
        for c_idx in case_indices:
            merged |= c_idx[i]
        out_indices.append(merged)

    deps[out_var] = out_indices

    # When all inputs are statically known, compute the concrete result
    # so const_vals tracking isn't broken by this op.
    which_val = atom_const_val(eqn.invars[0], const_vals)
    case_vals = [atom_const_val(c, const_vals) for c in cases]
    if which_val is not None and all(v is not None for v in case_vals):
        const_vals[out_var] = np.choose(
            which_val, [v for v in case_vals if v is not None]
        )

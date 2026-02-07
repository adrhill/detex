"""Propagation rule for cond (conditional branching)."""

from jax._src.core import JaxprEqn

from ._commons import ConstVals, Deps, IndexSets, index_sets


def prop_cond(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """cond selects one of several branches based on an integer index.
    Since we don't know which branch executes at trace time,
    output deps are the union across all branches.

    Layout:
        invars: [index_scalar, operands...]
        outvars: [results...]
        params: branches (tuple of ClosedJaxpr)

    Example: cond(pred, true_fn, false_fn, x)
        true_fn:  out = x[:2]  → deps [{0}, {1}]
        false_fn: out = x[1:]  → deps [{1}, {2}]
        union:    [{0, 1}, {1, 2}]
    """
    from . import prop_jaxpr

    branches = eqn.params["branches"]
    # First invar is the branch index (integer), rest are operands
    operand_deps: list[IndexSets] = [index_sets(deps, v) for v in eqn.invars[1:]]

    n_out = len(eqn.outvars)

    # Propagate each branch and collect per-branch output deps
    branch_outputs: list[list[IndexSets]] = []
    for branch in branches:
        branch_jaxpr = branch.jaxpr
        out = prop_jaxpr(branch_jaxpr, operand_deps, const_vals)
        branch_outputs.append(out)

    # Union across branches for each output variable
    for i in range(n_out):
        outvar = eqn.outvars[i]
        # Start from first branch, union with the rest
        merged: IndexSets = [s.copy() for s in branch_outputs[0][i]]
        for branch_out in branch_outputs[1:]:
            for j in range(len(merged)):
                merged[j] |= branch_out[i][j]
        deps[outvar] = merged

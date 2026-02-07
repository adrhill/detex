"""Propagation rule for cond (conditional branching)."""

from jax._src.core import JaxprEqn

from ._commons import (
    ConstVals,
    Deps,
    IndexSets,
    forward_const_vals,
    index_sets,
    seed_const_vals,
)


def prop_cond(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """cond/switch selects one of several branches based on an integer index.
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

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html
    """
    from . import prop_jaxpr

    branches = eqn.params["branches"]
    operands = eqn.invars[1:]
    operand_deps: list[IndexSets] = [index_sets(deps, v) for v in operands]

    n_out = len(eqn.outvars)

    # Propagate each branch and collect per-branch output deps
    branch_outputs: list[list[IndexSets]] = []
    for branch in branches:
        seed_const_vals(const_vals, branch.jaxpr.constvars, branch.consts)
        forward_const_vals(const_vals, operands, branch.jaxpr.invars)
        out = prop_jaxpr(branch.jaxpr, operand_deps, const_vals)
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

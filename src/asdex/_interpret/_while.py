"""Propagation rule for while_loop."""

from jax._src.core import JaxprEqn

from ._commons import ConstVals, Deps, IndexSets, index_sets


def prop_while(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """while_loop iterates a body until a condition becomes false.
    The carry variables may accumulate dependencies across iterations,
    so we iterate propagation to a fixed point.

    The cond jaxpr only produces a boolean and doesn't contribute to carry deps.

    Layout:
        invars: [body_consts..., cond_consts..., carry_init...]
        outvars: [carry_final...]
        params: body_jaxpr, body_nconsts, cond_jaxpr, cond_nconsts

    Example: carry = carry + const (accumulation)
        Input deps:  carry=[{0}, {1}], const=[{}, {}]
        After 1 iter: carry=[{0}, {1}] (stable immediately since const deps are empty)
    """
    from . import prop_jaxpr

    body_jaxpr = eqn.params["body_jaxpr"].jaxpr
    body_nconsts = eqn.params["body_nconsts"]
    cond_nconsts = eqn.params["cond_nconsts"]

    # Split invars: [body_consts | cond_consts | carry_init]
    n_carry = len(eqn.outvars)
    body_consts = eqn.invars[:body_nconsts]
    carry_init = eqn.invars[body_nconsts + cond_nconsts :]
    assert len(carry_init) == n_carry

    # Initialize carry deps from the initial values
    carry_deps: list[IndexSets] = [index_sets(deps, v) for v in carry_init]

    # body_jaxpr invars: [body_consts..., carry...]
    const_input: list[IndexSets] = [index_sets(deps, v) for v in body_consts]

    # Fixed-point iteration:
    # each iteration propagates the body and unions the result with current carry.
    # Since deps only grow (monotone on a finite lattice), this converges.
    max_iters = 100
    for _ in range(max_iters):
        body_input = const_input + carry_deps
        body_output = prop_jaxpr(body_jaxpr, body_input, const_vals)

        # Union body output deps into carry deps
        changed = False
        for i in range(n_carry):
            for j in range(len(carry_deps[i])):
                before = len(carry_deps[i][j])
                carry_deps[i][j] |= body_output[i][j]
                if len(carry_deps[i][j]) > before:
                    changed = True

        if not changed:
            break

    # Write final carry deps to outvars
    for outvar, out_deps in zip(eqn.outvars, carry_deps, strict=True):
        deps[outvar] = out_deps

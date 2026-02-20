"""Propagation rule for while_loop."""

from jax._src.core import JaxprEqn

from ._commons import (
    ConstVals,
    Deps,
    IndexSet,
    PropJaxprFn,
    fixed_point_loop,
    forward_const_vals,
    index_sets,
    seed_const_vals,
)


def prop_while(
    eqn: JaxprEqn,
    deps: Deps,
    const_vals: ConstVals,
    prop_jaxpr: PropJaxprFn,
) -> None:
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

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html
    """
    body_closed = eqn.params["body_jaxpr"]
    body_jaxpr = body_closed.jaxpr
    body_nconsts = eqn.params["body_nconsts"]
    cond_nconsts = eqn.params["cond_nconsts"]

    # Split invars: [body_consts | cond_consts | carry_init]
    n_carry = len(eqn.outvars)
    body_consts = eqn.invars[:body_nconsts]
    carry_init = eqn.invars[body_nconsts + cond_nconsts :]
    assert len(carry_init) == n_carry

    seed_const_vals(const_vals, body_jaxpr.constvars, body_closed.consts)
    # Only forward const_vals for body consts, not carry (carry changes each iteration)
    forward_const_vals(const_vals, body_consts, body_jaxpr.invars[:body_nconsts])

    # Initialize carry deps from the initial values
    carry_indices: list[list[IndexSet]] = [index_sets(deps, v) for v in carry_init]

    # body_jaxpr invars: [body_consts..., carry...]
    const_inputs: list[list[IndexSet]] = [index_sets(deps, v) for v in body_consts]

    def iterate(carry: list[list[IndexSet]]) -> list[list[IndexSet]]:
        return prop_jaxpr(body_jaxpr, const_inputs + carry, const_vals)

    fixed_point_loop(iterate, carry_indices, n_carry)

    # Write final carry deps to outvars
    for outvar, out_indices in zip(eqn.outvars, carry_indices, strict=True):
        deps[outvar] = out_indices

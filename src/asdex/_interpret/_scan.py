"""Propagation rule for scan."""

from jax._src.core import JaxprEqn

from ._commons import (
    ConstVals,
    Deps,
    IndexSets,
    PropJaxprFn,
    fixed_point_loop,
    forward_const_vals,
    index_sets,
    seed_const_vals,
)


def prop_scan(
    eqn: JaxprEqn,
    deps: Deps,
    const_vals: ConstVals,
    prop_jaxpr: PropJaxprFn,
) -> None:
    """Scan applies a body jaxpr iteratively, threading carry across iterations.

    Dependencies are propagated via fixed-point iteration on the carry,
    same as ``while_loop``.
    The body is identical every iteration,
    so deps grow monotonically on a finite lattice and converge fast.

    Layout:
        invars:  [consts..., carry_init..., xs...]
        outvars: [carry_final..., ys...]
        body jaxpr invars:  [consts..., carry..., x_slice...]
        body jaxpr outvars: [carry_new..., y_slice...]
        params: jaxpr, num_consts, num_carry, length, reverse, linear, unroll

    xs arrays have an extra leading dimension of size ``length``
    compared to their body counterparts x_slice.
    Similarly for ys vs y_slice.

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
    """
    body_closed = eqn.params["jaxpr"]
    body_jaxpr = body_closed.jaxpr
    num_consts = eqn.params["num_consts"]
    num_carry = eqn.params["num_carry"]

    # Split invars: [consts | carry_init | xs]
    consts = eqn.invars[:num_consts]
    carry_init = eqn.invars[num_consts : num_consts + num_carry]
    xs = eqn.invars[num_consts + num_carry :]

    # Split outvars: [carry_final | ys]
    carry_final = eqn.outvars[:num_carry]
    ys = eqn.outvars[num_carry:]

    seed_const_vals(const_vals, body_jaxpr.constvars, body_closed.consts)
    forward_const_vals(const_vals, consts, body_jaxpr.invars[:num_consts])

    # Prepare const deps for the body
    const_deps: list[IndexSets] = [index_sets(deps, v) for v in consts]

    # Initialize carry deps from initial values
    carry_deps: list[IndexSets] = [index_sets(deps, v) for v in carry_init]

    # Prepare x_slice deps by unioning across the leading (length) dimension.
    # Each xs[i] has shape (length, *rest), and the body sees x_slice with shape rest.
    # We overapproximate by unioning all slices.
    x_slice_deps: list[IndexSets] = []
    for x_var in xs:
        x_deps = index_sets(deps, x_var)
        x_shape = tuple(getattr(x_var.aval, "shape", ()))
        if len(x_shape) == 0:
            # Scalar xs — shouldn't happen but handle gracefully
            x_slice_deps.append(x_deps)
            continue
        length = x_shape[0]
        slice_numel = len(x_deps) // length
        # Union deps across all length slices for each element position
        merged: IndexSets = [set() for _ in range(slice_numel)]
        for t in range(length):
            for j in range(slice_numel):
                merged[j] |= x_deps[t * slice_numel + j]
        x_slice_deps.append(merged)

    # Fixed-point iteration on carry deps
    def iterate(carry: list[IndexSets]) -> list[IndexSets]:
        return prop_jaxpr(body_jaxpr, const_deps + carry + x_slice_deps, const_vals)

    body_output = fixed_point_loop(iterate, carry_deps, num_carry)

    # Write carry_final deps
    for outvar, out_deps in zip(carry_final, carry_deps, strict=True):
        deps[outvar] = out_deps

    # Write ys deps by tiling each y_slice across the length dimension.
    # Every iteration slice gets the same (overapproximate) deps.
    y_slice_outputs = body_output[num_carry:]
    for outvar, slice_deps in zip(ys, y_slice_outputs, strict=True):
        y_shape = tuple(getattr(outvar.aval, "shape", ()))
        if len(y_shape) == 0:
            deps[outvar] = slice_deps
            continue
        length = y_shape[0]
        # Tile: repeat the slice deps for each time step
        tiled: IndexSets = []
        for _ in range(length):
            tiled.extend([s.copy() for s in slice_deps])
        deps[outvar] = tiled

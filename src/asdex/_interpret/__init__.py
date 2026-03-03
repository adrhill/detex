"""Propagate index sets through a jaxpr to determine sparsity.

Each JAX primitive has a handler that maps input index sets to output index sets.
For example, element-wise ops preserve per-element dependencies, while reductions
union all input dependencies into a single output.

The main entry point is `prop_jaxpr`, which walks the computation graph
and applies the appropriate handler for each equation.
"""

import numpy as np
from jax._src.core import Jaxpr, JaxprEqn, Var

from ._broadcast import prop_broadcast_in_dim
from ._commons import (
    ConstVals,
    Deps,
    IndexSet,
    ValueBounds,
    atom_const_val,
    atom_numel,
    atom_shape,
    atom_value_bounds,
    conservative_indices,
    empty_index_sets,
    forward_const_vals,
    forward_value_bounds,
    index_sets,
    seed_const_vals,
)
from ._concatenate import prop_concatenate
from ._cond import prop_cond
from ._conv import prop_conv_general_dilated
from ._cumsum import prop_cumsum
from ._dot_general import prop_dot_general
from ._dynamic_slice import prop_dynamic_slice, prop_dynamic_update_slice
from ._elementwise import (
    prop_binary_elementwise,
    prop_convert_element_type,
    prop_integer_pow,
    prop_unary_elementwise,
    prop_zero_derivative,
    propagate_const_elementwise,
)
from ._equinox._select_if_vmap import prop_select_if_vmap
from ._gather import prop_gather
from ._mul import prop_mul
from ._pad import prop_pad
from ._platform_index import prop_platform_index
from ._reduce import prop_reduce
from ._reshape import prop_reshape
from ._rev import prop_rev
from ._scan import prop_scan
from ._scatter import prop_scatter
from ._select import prop_select_n
from ._slice import prop_slice
from ._sort import prop_sort
from ._split import prop_split
from ._squeeze import prop_squeeze
from ._tile import prop_tile
from ._top_k import prop_top_k
from ._transpose import prop_transpose
from ._while import prop_while


def prop_jaxpr(
    jaxpr: Jaxpr,
    input_indices: list[list[IndexSet]],
    const_vals: ConstVals | None = None,
    value_bounds: ValueBounds | None = None,
) -> list[list[IndexSet]]:
    """Propagate index sets through a jaxpr.

    Args:
        jaxpr: The jaxpr to analyze
        input_indices: List of per-element index set lists, one per input variable
        const_vals: Optional mapping of constant variables to their values.
            Used for precise tracking of static indices in gather/scatter.
        value_bounds: Optional pre-seeded value bounds from an outer scope.
            Used to forward bounded-but-not-constant values into nested jaxprs.

    Returns:
        List of per-element index set lists, one per output variable
    """
    deps: Deps = {}
    if const_vals is None:
        const_vals = {}
    if value_bounds is None:
        value_bounds = {}

    # Initialize input variables
    for var, indices in zip(jaxpr.invars, input_indices, strict=False):
        deps[var] = indices

    # Initialize constant variables (no input dependencies)
    for var in jaxpr.constvars:
        deps[var] = empty_index_sets(atom_numel(var))

    # Process each equation
    for eqn in jaxpr.eqns:
        prop_dispatch(eqn, deps, const_vals, value_bounds)

    # Return output dependencies
    return [index_sets(deps, outvar) for outvar in jaxpr.outvars]


def prop_nested_jaxpr(
    eqn: JaxprEqn,
    deps: Deps,
    const_vals: ConstVals,
    value_bounds: ValueBounds,
) -> None:
    """Handle primitives with nested jaxprs by recursively tracing."""
    closed = eqn.params.get("jaxpr")
    if closed is None:
        msg = (
            f"Primitive '{eqn.primitive.name}' has no 'jaxpr' parameter. "
            "Please help out asdex's development by reporting this at "
            "https://github.com/adrhill/asdex/issues"
        )
        raise ValueError(msg)

    # Unwrap ClosedJaxpr, seeding const_vals for captured constants
    if hasattr(closed, "jaxpr"):
        seed_const_vals(const_vals, closed.jaxpr.constvars, closed.consts)
        closed = closed.jaxpr

    forward_const_vals(const_vals, eqn.invars, closed.invars)
    forward_value_bounds(value_bounds, eqn.invars, closed.invars)
    input_indices = [index_sets(deps, invar) for invar in eqn.invars]
    output_indices = prop_jaxpr(closed, input_indices, const_vals, value_bounds)

    for outvar, indices, inner_outvar in zip(
        eqn.outvars,
        output_indices,
        closed.outvars,
        strict=False,
    ):
        deps[outvar] = indices
        # Propagate value bounds from inner output vars to outer output vars.
        if isinstance(inner_outvar, Var) and inner_outvar in value_bounds:
            value_bounds[outvar] = value_bounds[inner_outvar]


def prop_custom_call(
    eqn: JaxprEqn,
    deps: Deps,
    const_vals: ConstVals,
    value_bounds: ValueBounds,
) -> None:
    """Custom differentiation wrappers delegate to their forward jaxpr.

    JAX's `custom_jvp` and `custom_vjp` allow users to define custom derivative rules.
    For sparsity detection, we only need the forward pass behavior,
    which is stored in the `call_jaxpr` parameter.

    The custom derivative rules don't affect which outputs depend on which
    inputs — they only change how derivatives are computed.
    """
    closed = eqn.params.get("call_jaxpr")
    if closed is None:
        msg = (
            f"Primitive '{eqn.primitive.name}' has no 'call_jaxpr' parameter. "
            "Please help out asdex's development by reporting this at "
            "https://github.com/adrhill/asdex/issues"
        )
        raise ValueError(msg)

    # Unwrap ClosedJaxpr, seeding const_vals for captured constants
    if hasattr(closed, "jaxpr"):
        seed_const_vals(const_vals, closed.jaxpr.constvars, closed.consts)
        closed = closed.jaxpr

    forward_const_vals(const_vals, eqn.invars, closed.invars)
    forward_value_bounds(value_bounds, eqn.invars, closed.invars)
    input_indices = [index_sets(deps, invar) for invar in eqn.invars]
    output_indices = prop_jaxpr(closed, input_indices, const_vals, value_bounds)

    for outvar, indices, inner_outvar in zip(
        eqn.outvars,
        output_indices,
        closed.outvars,
        strict=False,
    ):
        deps[outvar] = indices
        if isinstance(inner_outvar, Var) and inner_outvar in value_bounds:
            value_bounds[outvar] = value_bounds[inner_outvar]


def prop_dispatch(
    eqn: JaxprEqn,
    deps: Deps,
    const_vals: ConstVals,
    value_bounds: ValueBounds,
) -> None:
    """Propagate dependencies through a single equation."""
    match eqn.primitive.name:
        case "argmax" | "argmin":
            prop_zero_derivative(eqn, deps)
            # Source value bounds: output is in [0, axis_size - 1].
            axes = eqn.params["axes"]
            in_shape = atom_shape(eqn.invars[0])
            axis_size = in_shape[axes[0]]
            out_var = eqn.outvars[0]
            lo = np.zeros(atom_shape(out_var), dtype=np.int64)
            hi = np.full(atom_shape(out_var), axis_size - 1, dtype=np.int64)
            value_bounds[out_var] = (lo, hi)
        case (
            "floor"
            | "ceil"
            | "round"
            | "sign"
            | "is_finite"
            | "clz"
            | "clamp"
            | "population_count"
            | "reduce_and"
            | "reduce_or"
            | "reduce_xor"
        ):
            prop_zero_derivative(eqn, deps)
        case "eq" | "ne" | "lt" | "le" | "gt" | "ge" | "lt_to" | "le_to":
            prop_zero_derivative(eqn, deps)
            propagate_const_elementwise(eqn, const_vals)
            _propagate_const_comparison_from_bounds(eqn, const_vals, value_bounds)
        case "and" | "or" | "xor":
            prop_zero_derivative(eqn, deps)
            propagate_const_elementwise(eqn, const_vals)
        case "jit" | "pjit" | "xla_call" | "named_call":
            prop_nested_jaxpr(eqn, deps, const_vals, value_bounds)
        case "slice":
            prop_slice(eqn, deps, const_vals)
        case "pad":
            prop_pad(eqn, deps)
        case "squeeze":
            prop_squeeze(eqn, deps)
        case "broadcast_in_dim":
            prop_broadcast_in_dim(eqn, deps, const_vals)
            _propagate_bounds_broadcast(eqn, value_bounds)
        case "concatenate":
            prop_concatenate(eqn, deps, const_vals)
        case "reshape":
            prop_reshape(eqn, deps, const_vals)
        case "transpose":
            prop_transpose(eqn, deps, const_vals)
        case "rev":
            prop_rev(eqn, deps)
        case "integer_pow":
            prop_integer_pow(eqn, deps)
        case "mul":
            prop_mul(eqn, deps, const_vals)
            propagate_const_elementwise(eqn, const_vals)
        case (
            "add"
            | "sub"
            | "div"
            | "pow"
            | "max"
            | "min"
            | "add_any"
            | "atan2"
            | "rem"
            | "nextafter"
            | "complex"
        ):
            prop_binary_elementwise(eqn, deps)
            propagate_const_elementwise(eqn, const_vals)
            _propagate_bounds_binary(eqn, const_vals, value_bounds)
        case (
            "neg"
            | "exp"
            | "log"
            | "sin"
            | "cos"
            | "tan"
            | "sqrt"
            | "abs"
            | "sinh"
            | "cosh"
            | "tanh"
            | "log1p"
            | "expm1"
            | "acos"
            | "acosh"
            | "asin"
            | "asinh"
            | "atan"
            | "atanh"
            | "cbrt"
            | "conj"
            | "copy"
            | "exp2"
            | "logistic"
            | "real"
            | "imag"
            | "rsqrt"
            | "erf"
            | "square"
        ):
            prop_unary_elementwise(eqn, deps)
        case "reduce_sum" | "reduce_max" | "reduce_min" | "reduce_prod":
            prop_reduce(eqn, deps)
        case "convert_element_type" | "bitcast_convert_type" | "reduce_precision":
            prop_convert_element_type(eqn, deps, const_vals)
            _propagate_bounds_convert(eqn, value_bounds)
        case "stop_gradient":
            prop_convert_element_type(eqn, deps, const_vals)
            _propagate_bounds_convert(eqn, value_bounds)
        case "conv_general_dilated":
            prop_conv_general_dilated(eqn, deps)
        case "custom_jvp_call" | "custom_vjp_call":
            prop_custom_call(eqn, deps, const_vals, value_bounds)
        case "gather":
            prop_gather(eqn, deps, const_vals, value_bounds)
        case "scatter" | "scatter-add" | "scatter-mul" | "scatter-min" | "scatter-max":
            prop_scatter(eqn, deps, const_vals, value_bounds)
        case "select_n":
            prop_select_n(eqn, deps, const_vals)
            _propagate_bounds_select_n(eqn, const_vals, value_bounds)
        case "select_if_vmap":
            prop_select_if_vmap(eqn, deps, const_vals)
        case "iota":
            _prop_iota(eqn, deps, const_vals)
        case "while":
            prop_while(eqn, deps, const_vals, prop_jaxpr)
        case "cond":
            prop_cond(eqn, deps, const_vals, prop_jaxpr)
        case "platform_index":
            prop_platform_index(eqn, deps)
        case "dynamic_slice":
            prop_dynamic_slice(eqn, deps, const_vals, value_bounds)
        case "dynamic_update_slice":
            prop_dynamic_update_slice(eqn, deps, const_vals, value_bounds)
        case "top_k":
            prop_top_k(eqn, deps)
        case "not":
            prop_zero_derivative(eqn, deps)
        # TODO: add precise handlers for remaining control flow operators.
        # https://docs.jax.dev/en/latest/jax.lax.html#control-flow-operators
        case "scan":
            prop_scan(eqn, deps, const_vals, prop_jaxpr)
        case "dot_general":
            prop_dot_general(eqn, deps, const_vals)
        case "split":
            prop_split(eqn, deps)
        case "tile":
            prop_tile(eqn, deps, const_vals)
        case "sort":
            prop_sort(eqn, deps)
        case "cumsum":
            prop_cumsum(eqn, deps)
        # Conservative fallback: all outputs depend on all inputs.
        case (
            "nonbatchable"
            | "unvmap_any"  # from Equinox
            | "unvmap_max"  # from Equinox
            | "pure_callback"
        ):
            prop_conservative_fallback(eqn, deps)
        case _:
            prop_throw_error(eqn, deps)


# Value bounds propagation


def _propagate_const_comparison_from_bounds(
    eqn: JaxprEqn,
    const_vals: ConstVals,
    value_bounds: ValueBounds,
) -> None:
    """Derive const comparison results from value bounds.

    When bounds prove the comparison is always True or always False,
    store the result as a const value.
    This enables ``select_n`` to pick the correct branch.
    """
    # Skip if already resolved as const.
    if eqn.outvars[0] in const_vals:
        return

    b1 = atom_value_bounds(eqn.invars[0], const_vals, value_bounds)
    b2 = atom_value_bounds(eqn.invars[1], const_vals, value_bounds)
    if b1 is None or b2 is None:
        return

    lo1, hi1 = b1
    lo2, hi2 = b2
    out_shape = atom_shape(eqn.outvars[0])

    result: np.ndarray | None = None
    match eqn.primitive.name:
        case "lt":
            if np.all(hi1 < lo2):
                result = np.ones(out_shape, dtype=bool)
            elif np.all(lo1 >= hi2):
                result = np.zeros(out_shape, dtype=bool)
        case "le":
            if np.all(hi1 <= lo2):
                result = np.ones(out_shape, dtype=bool)
            elif np.all(lo1 > hi2):
                result = np.zeros(out_shape, dtype=bool)
        case "gt":
            if np.all(lo1 > hi2):
                result = np.ones(out_shape, dtype=bool)
            elif np.all(hi1 <= lo2):
                result = np.zeros(out_shape, dtype=bool)
        case "ge":
            if np.all(lo1 >= hi2):
                result = np.ones(out_shape, dtype=bool)
            elif np.all(hi1 < lo2):
                result = np.zeros(out_shape, dtype=bool)

    if result is not None:
        const_vals[eqn.outvars[0]] = result


def _propagate_bounds_select_n(
    eqn: JaxprEqn,
    const_vals: ConstVals,
    value_bounds: ValueBounds,
) -> None:
    """Propagate value bounds through select_n when the predicate is const.

    ``select_n(pred, x, y)`` returns ``x`` when pred is False, ``y`` when True.
    If pred is a known constant, forward the bounds of the selected branch.
    """
    pred_val = atom_const_val(eqn.invars[0], const_vals)
    if pred_val is None:
        return

    # Boolean select_n with two value branches.
    if len(eqn.invars) == 3 and pred_val.dtype == bool:
        if np.all(pred_val == False):  # noqa: E712
            # Selects first value branch (invars[1]).
            selected = eqn.invars[1]
        elif np.all(pred_val == True):  # noqa: E712
            # Selects second value branch (invars[2]).
            selected = eqn.invars[2]
        else:
            return

        # Forward bounds from the selected branch.
        if isinstance(selected, Var) and selected in value_bounds:
            value_bounds[eqn.outvars[0]] = value_bounds[selected]


def _propagate_bounds_binary(
    eqn: JaxprEqn,
    const_vals: ConstVals,
    value_bounds: ValueBounds,
) -> None:
    """Propagate value bounds through add, sub, and add_any.

    Uses interval arithmetic: [a,b] + [c,d] = [a+c, b+d],
    [a,b] - [c,d] = [a-d, b-c].
    Only propagates when at least one input has known bounds.
    """
    name = eqn.primitive.name
    if name not in ("add", "sub", "add_any"):
        return

    b1 = atom_value_bounds(eqn.invars[0], const_vals, value_bounds)
    b2 = atom_value_bounds(eqn.invars[1], const_vals, value_bounds)
    if b1 is None or b2 is None:
        return

    lo1, hi1 = b1
    lo2, hi2 = b2
    if name in ("add", "add_any"):
        value_bounds[eqn.outvars[0]] = (lo1 + lo2, hi1 + hi2)
    else:  # sub
        value_bounds[eqn.outvars[0]] = (lo1 - hi2, hi1 - lo2)


def _propagate_bounds_convert(eqn: JaxprEqn, value_bounds: ValueBounds) -> None:
    """Propagate value bounds through convert_element_type."""
    in_var = eqn.invars[0]
    if not isinstance(in_var, Var) or in_var not in value_bounds:
        return
    lo, hi = value_bounds[in_var]
    new_dtype = eqn.params.get("new_dtype")
    if new_dtype is not None:
        value_bounds[eqn.outvars[0]] = (lo.astype(new_dtype), hi.astype(new_dtype))
    else:
        value_bounds[eqn.outvars[0]] = (lo, hi)


def _propagate_bounds_broadcast(eqn: JaxprEqn, value_bounds: ValueBounds) -> None:
    """Propagate value bounds through broadcast_in_dim.

    Broadcasting replicates values without changing them,
    so bounds are broadcast to the output shape.
    """
    in_var = eqn.invars[0]
    if not isinstance(in_var, Var) or in_var not in value_bounds:
        return
    lo, hi = value_bounds[in_var]
    out_shape = eqn.params["shape"]
    broadcast_dims = eqn.params["broadcast_dimensions"]
    in_shape = lo.shape or (1,)

    intermediate_shape = [1] * len(out_shape)
    for i, out_dim in enumerate(broadcast_dims):
        intermediate_shape[out_dim] = in_shape[i]

    value_bounds[eqn.outvars[0]] = (
        np.broadcast_to(np.reshape(lo, intermediate_shape), out_shape),
        np.broadcast_to(np.reshape(hi, intermediate_shape), out_shape),
    )


def _prop_iota(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Iota generates a constant index array with no input dependencies.

    The output is fully determined by the parameters (shape, dtype, dimension),
    so all dependency sets are empty.
    We also track the concrete values for downstream gather/scatter precision.

    Jaxpr:
        invars: [] (no inputs)
        shape: output shape
        dtype: output dtype
        dimension: axis along which indices increase
    """
    shape = eqn.params["shape"]
    numel = int(np.prod(shape))
    deps[eqn.outvars[0]] = empty_index_sets(numel)

    dtype = eqn.params["dtype"]
    dim = eqn.params["dimension"]
    const_vals[eqn.outvars[0]] = np.broadcast_to(
        np.arange(shape[dim], dtype=dtype).reshape(
            [shape[dim] if i == dim else 1 for i in range(len(shape))]
        ),
        shape,
    )


def prop_conservative_fallback(eqn: JaxprEqn, deps: Deps) -> None:
    """Conservative fallback for primitives without precise handlers.

    Assumes worst-case: every output element may depend on every input element.
    This is correct but may overestimate sparsity (more nonzeros than necessary).

    Used for primitives without precise handlers.
    """
    all_inputs: list[IndexSet] = []
    for invar in eqn.invars:
        all_inputs.extend(index_sets(deps, invar))
    for outvar in eqn.outvars:
        deps[outvar] = conservative_indices(all_inputs, atom_numel(outvar))


def prop_throw_error(eqn: JaxprEqn, deps: Deps) -> None:
    """Raise an error for unknown primitives.

    This ensures we don't silently produce incorrect sparsity patterns.
    """
    msg = (
        f"No handler for primitive '{eqn.primitive.name}'. "
        "Please help out asdex's development by reporting this at "
        "https://github.com/adrhill/asdex/issues"
    )
    raise NotImplementedError(msg)

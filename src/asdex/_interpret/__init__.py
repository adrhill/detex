"""Propagate index sets through a jaxpr to determine sparsity.

Each JAX primitive has a handler that maps input index sets to output index sets.
For example, element-wise ops preserve per-element dependencies, while reductions
union all input dependencies into a single output.

The main entry point is `prop_jaxpr`, which walks the computation graph
and applies the appropriate handler for each equation.
"""

import numpy as np
from jax._src.core import Jaxpr, JaxprEqn

from ._broadcast import prop_broadcast_in_dim
from ._commons import (
    ConstVals,
    Deps,
    IndexSet,
    atom_numel,
    conservative_indices,
    empty_index_sets,
    forward_const_vals,
    index_sets,
    seed_const_vals,
)
from ._concatenate import prop_concatenate
from ._cond import prop_cond
from ._conv import prop_conv_general_dilated
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
) -> list[list[IndexSet]]:
    """Propagate index sets through a jaxpr.

    Args:
        jaxpr: The jaxpr to analyze
        input_indices: List of per-element index set lists, one per input variable
        const_vals: Optional mapping of constant variables to their values.
            Used for precise tracking of static indices in gather/scatter.

    Returns:
        List of per-element index set lists, one per output variable
    """
    deps: Deps = {}
    if const_vals is None:
        const_vals = {}

    # Initialize input variables
    for var, indices in zip(jaxpr.invars, input_indices, strict=False):
        deps[var] = indices

    # Initialize constant variables (no input dependencies)
    for var in jaxpr.constvars:
        deps[var] = empty_index_sets(atom_numel(var))

    # Process each equation
    for eqn in jaxpr.eqns:
        prop_dispatch(eqn, deps, const_vals)

    # Return output dependencies
    return [index_sets(deps, outvar) for outvar in jaxpr.outvars]


def prop_nested_jaxpr(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
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
    input_indices = [index_sets(deps, invar) for invar in eqn.invars]
    output_indices = prop_jaxpr(closed, input_indices, const_vals)

    for outvar, indices in zip(eqn.outvars, output_indices, strict=False):
        deps[outvar] = indices


def prop_custom_call(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
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
    input_indices = [index_sets(deps, invar) for invar in eqn.invars]
    output_indices = prop_jaxpr(closed, input_indices, const_vals)

    for outvar, indices in zip(eqn.outvars, output_indices, strict=False):
        deps[outvar] = indices


def prop_dispatch(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Propagate dependencies through a single equation."""
    match eqn.primitive.name:
        case (
            "floor"
            | "ceil"
            | "round"
            | "sign"
            | "is_finite"
            | "argmax"
            | "argmin"
            | "clz"
            | "clamp"
            | "population_count"
            | "reduce_and"
            | "reduce_or"
            | "reduce_xor"
        ):
            prop_zero_derivative(eqn, deps)
        case "eq" | "ne" | "lt" | "le" | "gt" | "ge":
            prop_zero_derivative(eqn, deps)
            propagate_const_elementwise(eqn, const_vals)
        case "and" | "or" | "xor":
            prop_zero_derivative(eqn, deps)
            propagate_const_elementwise(eqn, const_vals)
        case "jit" | "pjit" | "xla_call" | "named_call":
            prop_nested_jaxpr(eqn, deps, const_vals)
        case "slice":
            prop_slice(eqn, deps, const_vals)
        case "pad":
            prop_pad(eqn, deps)
        case "squeeze":
            prop_squeeze(eqn, deps)
        case "broadcast_in_dim":
            prop_broadcast_in_dim(eqn, deps, const_vals)
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
            | "square"
        ):
            prop_unary_elementwise(eqn, deps)
        case "reduce_sum" | "reduce_max" | "reduce_min" | "reduce_prod":
            prop_reduce(eqn, deps)
        case "convert_element_type" | "bitcast_convert_type" | "reduce_precision":
            prop_convert_element_type(eqn, deps, const_vals)
        case "stop_gradient":
            prop_convert_element_type(eqn, deps, const_vals)
        case "conv_general_dilated":
            prop_conv_general_dilated(eqn, deps)
        case "custom_jvp_call" | "custom_vjp_call":
            prop_custom_call(eqn, deps, const_vals)
        case "gather":
            prop_gather(eqn, deps, const_vals)
        case "scatter" | "scatter-add" | "scatter-mul" | "scatter-min" | "scatter-max":
            prop_scatter(eqn, deps, const_vals)
        case "select_n":
            prop_select_n(eqn, deps, const_vals)
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
            prop_dynamic_slice(eqn, deps, const_vals)
        case "dynamic_update_slice":
            prop_dynamic_update_slice(eqn, deps, const_vals)
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

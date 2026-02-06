"""Propagate index sets through a jaxpr to determine sparsity.

Each JAX primitive has a handler that maps input index sets to output index sets.
For example, element-wise ops preserve per-element dependencies, while reductions
union all input dependencies into a single output.

The main entry point is `prop_jaxpr`, which walks the computation graph
and applies the appropriate handler for each equation.
"""

import numpy as np
from jax._src.core import Jaxpr, JaxprEqn

from ._commons import ConstVals, Deps, IndexSets, atom_numel, index_sets, union_all
from ._concatenate import prop_concatenate
from ._conv import prop_conv_general_dilated
from ._elementwise import (
    prop_binary_elementwise,
    prop_convert_element_type,
    prop_integer_pow,
    prop_unary_elementwise,
    prop_zero_derivative,
    propagate_const_binary,
)

# Ufuncs for evaluating constant values during tracing.
# Used to propagate static index values through arithmetic to gather/scatter.
_ARITHMETIC_UFUNCS: dict[str, np.ufunc] = {
    "add": np.add,
    "add_any": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "pow": np.power,
    "max": np.maximum,
    "min": np.minimum,
}

_COMPARISON_UFUNCS: dict[str, np.ufunc] = {
    "lt": np.less,
    "le": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
    "eq": np.equal,
    "ne": np.not_equal,
}
from ._gather import prop_gather
from ._indexing import prop_broadcast_in_dim, prop_reshape, prop_slice, prop_squeeze
from ._reduction import prop_reduce_sum
from ._scatter import prop_scatter
from ._select import prop_select_n


def prop_jaxpr(
    jaxpr: Jaxpr,
    input_indices: list[IndexSets],
    const_vals: ConstVals | None = None,
) -> list[IndexSets]:
    """
    Propagate index sets through a jaxpr.

    Args:
        jaxpr: The jaxpr to analyze
        input_indices: List of IndexSets, one per input variable
        const_vals: Optional mapping of constant variables to their values.
            Used for precise tracking of static indices in gather/scatter.

    Returns:
        List of IndexSets, one per output variable
    """
    deps: Deps = {}
    if const_vals is None:
        const_vals = {}

    # Initialize input variables
    for var, indices in zip(jaxpr.invars, input_indices, strict=False):
        deps[var] = indices

    # Initialize constant variables (no input dependencies)
    for var in jaxpr.constvars:
        deps[var] = [set() for _ in range(atom_numel(var))]

    # Process each equation
    for eqn in jaxpr.eqns:
        prop_dispatch(eqn, deps, const_vals)

    # Return output dependencies
    return [index_sets(deps, outvar) for outvar in jaxpr.outvars]


def prop_nested_jaxpr(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Handle primitives with nested jaxprs by recursively tracing."""
    nested_jaxpr = eqn.params.get("jaxpr")
    if nested_jaxpr is None:
        msg = (
            f"Primitive '{eqn.primitive.name}' has no 'jaxpr' parameter. "
            "Please report this at https://github.com/adrhill/asdex/issues"
        )
        raise ValueError(msg)

    # Handle ClosedJaxpr wrapper
    if hasattr(nested_jaxpr, "jaxpr"):
        nested_jaxpr = nested_jaxpr.jaxpr

    input_indices = [index_sets(deps, invar) for invar in eqn.invars]
    output_indices = prop_jaxpr(nested_jaxpr, input_indices, const_vals)

    for outvar, indices in zip(eqn.outvars, output_indices, strict=False):
        deps[outvar] = indices


def prop_custom_call(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Custom differentiation wrappers delegate to their forward jaxpr.

    JAX's `custom_jvp` and `custom_vjp` allow users to define custom derivative rules.
    For sparsity detection, we only need the forward pass behavior,
    which is stored in the `call_jaxpr` parameter.

    The custom derivative rules don't affect which outputs depend on which
    inputs - they only change how derivatives are computed.
    """
    call_jaxpr = eqn.params.get("call_jaxpr")
    if call_jaxpr is None:
        msg = (
            f"Primitive '{eqn.primitive.name}' has no 'call_jaxpr' parameter. "
            "Please report this at https://github.com/adrhill/asdex/issues"
        )
        raise ValueError(msg)

    # Handle ClosedJaxpr wrapper
    if hasattr(call_jaxpr, "jaxpr"):
        call_jaxpr = call_jaxpr.jaxpr

    input_indices = [index_sets(deps, invar) for invar in eqn.invars]
    output_indices = prop_jaxpr(call_jaxpr, input_indices, const_vals)

    for outvar, indices in zip(eqn.outvars, output_indices, strict=False):
        deps[outvar] = indices


def prop_dispatch(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Propagate dependencies through a single equation."""
    match eqn.primitive.name:
        case "floor" | "ceil" | "round" | "sign" | "is_finite":
            prop_zero_derivative(eqn, deps)
        case "eq" | "ne" | "lt" | "le" | "gt" | "ge":
            prop_zero_derivative(eqn, deps)
            propagate_const_binary(eqn, const_vals, _COMPARISON_UFUNCS)
        case "jit" | "pjit" | "xla_call" | "named_call":
            prop_nested_jaxpr(eqn, deps, const_vals)
        case "slice":
            prop_slice(eqn, deps)
        case "squeeze":
            prop_squeeze(eqn, deps)
        case "broadcast_in_dim":
            prop_broadcast_in_dim(eqn, deps, const_vals)
        case "concatenate":
            prop_concatenate(eqn, deps)
        case "reshape":
            prop_reshape(eqn, deps)
        case "integer_pow":
            prop_integer_pow(eqn, deps)
        case "add" | "sub" | "mul" | "div" | "pow" | "max" | "min" | "add_any":
            prop_binary_elementwise(eqn, deps)
            propagate_const_binary(eqn, const_vals, _ARITHMETIC_UFUNCS)
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
        ):
            prop_unary_elementwise(eqn, deps)
        case "reduce_sum":
            prop_reduce_sum(eqn, deps)
        case "convert_element_type":
            prop_convert_element_type(eqn, deps)
        case "conv_general_dilated":
            prop_conv_general_dilated(eqn, deps)
        case "custom_jvp_call" | "custom_vjp_call":
            prop_custom_call(eqn, deps, const_vals)
        case "gather":
            prop_gather(eqn, deps, const_vals)
        case "scatter" | "scatter-add":
            prop_scatter(eqn, deps, const_vals)
        case "select_n":
            prop_select_n(eqn, deps, const_vals)
        # TODO: implement precise handlers for these primitives.
        # Currently uses conservative fallback (all outputs depend on all inputs).
        case (
            "argmax"
            | "dot_general"
            | "iota"
            | "pad"
            | "reduce_max"
            | "reduce_prod"
            | "rev"
            | "sort"
            | "split"
            | "tile"
            | "transpose"
        ):
            prop_conservative_fallback(eqn, deps)
        case _:
            prop_throw_error(eqn, deps)


def prop_conservative_fallback(eqn: JaxprEqn, deps: Deps) -> None:
    """Conservative fallback for primitives without precise handlers.
    Assumes worst-case: every output element may depend on every input element.
    This is correct but may overestimate sparsity (more nonzeros than necessary).

    Used for: dot_general, transpose, sort, etc.
    """
    all_inputs: IndexSets = []
    for invar in eqn.invars:
        all_inputs.extend(index_sets(deps, invar))
    all_deps = union_all(all_inputs)
    for outvar in eqn.outvars:
        deps[outvar] = [all_deps.copy() for _ in range(atom_numel(outvar))]


def prop_throw_error(eqn: JaxprEqn, deps: Deps) -> None:
    """Raise an error for unknown primitives.
    This ensures we don't silently produce incorrect sparsity patterns.
    """
    msg = (
        f"No handler for primitive '{eqn.primitive.name}'. "
        "Please report this at https://github.com/adrhill/asdex/issues"
    )
    raise NotImplementedError(msg)

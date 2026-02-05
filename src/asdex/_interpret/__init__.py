"""Propagate index sets through a jaxpr to determine sparsity.

Each JAX primitive has a handler that maps input index sets to output index sets.
For example, element-wise ops preserve per-element dependencies, while reductions
union all input dependencies into a single output.

The main entry point is `prop_jaxpr`, which walks the computation graph
and applies the appropriate handler for each equation.
"""

from jax._src.core import Jaxpr, JaxprEqn

from ._commons import (
    NESTED_JAXPR_PRIMITIVES,
    ZERO_DERIVATIVE_PRIMITIVES,
    Deps,
    IndexSets,
    atom_numel,
    index_sets,
    union_all,
)
from ._conv import prop_conv_general_dilated
from ._elementwise import (
    prop_binary_elementwise,
    prop_convert_element_type,
    prop_integer_pow,
    prop_unary_elementwise,
    prop_zero_derivative,
)
from ._indexing import (
    prop_broadcast_in_dim,
    prop_concatenate,
    prop_reshape,
    prop_slice,
    prop_squeeze,
)
from ._reduction import prop_reduce_sum


def prop_jaxpr(jaxpr: Jaxpr, input_indices: list[IndexSets]) -> list[IndexSets]:
    """
    Propagate index sets through a jaxpr.

    Args:
        jaxpr: The jaxpr to analyze
        input_indices: List of IndexSets, one per input variable

    Returns:
        List of IndexSets, one per output variable
    """
    deps: Deps = {}

    # Initialize input variables
    for var, indices in zip(jaxpr.invars, input_indices, strict=False):
        deps[var] = indices

    # Initialize constant variables (no input dependencies)
    for var in jaxpr.constvars:
        deps[var] = [set() for _ in range(atom_numel(var))]

    # Process each equation
    for eqn in jaxpr.eqns:
        prop_equation(eqn, deps)

    # Return output dependencies
    return [index_sets(deps, outvar) for outvar in jaxpr.outvars]


def prop_nested_jaxpr(eqn: JaxprEqn, deps: Deps) -> None:
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
    output_indices = prop_jaxpr(nested_jaxpr, input_indices)

    for outvar, indices in zip(eqn.outvars, output_indices, strict=False):
        deps[outvar] = indices


def prop_equation(eqn: JaxprEqn, deps: Deps) -> None:
    """Propagate dependencies through a single equation."""
    match eqn.primitive.name:
        case prim if prim in ZERO_DERIVATIVE_PRIMITIVES:
            prop_zero_derivative(eqn, deps)
        case prim if prim in NESTED_JAXPR_PRIMITIVES:
            prop_nested_jaxpr(eqn, deps)
        case "slice":
            prop_slice(eqn, deps)
        case "squeeze":
            prop_squeeze(eqn, deps)
        case "broadcast_in_dim":
            prop_broadcast_in_dim(eqn, deps)
        case "concatenate":
            prop_concatenate(eqn, deps)
        case "reshape":
            prop_reshape(eqn, deps)
        case "integer_pow":
            prop_integer_pow(eqn, deps)
        case "add" | "sub" | "mul" | "div" | "pow" | "max" | "min" | "add_any":
            prop_binary_elementwise(eqn, deps)
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
        # TODO: implement precise handlers for these primitives.
        # Currently uses conservative fallback (all outputs depend on all inputs).
        case (
            "argmax"
            | "dot_general"
            | "gather"
            | "iota"
            | "pad"
            | "reduce_max"
            | "reduce_prod"
            | "rev"
            | "scatter"
            | "select_n"
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

    Used for: dot_general, gather, scatter, transpose, sort, etc.
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

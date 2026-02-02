"""Global Jacobian sparsity detection via jaxpr graph analysis."""

import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax._src.core import Jaxpr, JaxprEqn, Literal, Var
from scipy.sparse import coo_matrix

# Type aliases for readability
IndexSets = list[set[int]]
ReadFn = Callable[[Var | Literal], IndexSets]
WriteFn = Callable[[Var, IndexSets], None]
GetSizeFn = Callable[[Var | Literal], int]

# Primitives with zero derivatives (output doesn't depend on input)
ZERO_DERIVATIVE_PRIMITIVES = frozenset(
    [
        "floor",
        "ceil",
        "round",
        "sign",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "is_finite",
    ]
)

# Primitives that contain nested jaxprs we should trace into
NESTED_JAXPR_PRIMITIVES = frozenset(["jit", "pjit", "xla_call", "named_call"])


def _shape_size(shape: Sequence[int]) -> int:
    """Compute the total number of elements from a shape tuple."""
    return math.prod(shape) if shape else 1


def _propagate_zero_derivative(
    eqn: JaxprEqn, read: ReadFn, write: WriteFn, get_var_size: GetSizeFn
) -> None:
    """Zero-derivative ops: output has no dependence on inputs."""
    for outvar in eqn.outvars:
        size = get_var_size(outvar)
        write(outvar, [set() for _ in range(size)])


def _propagate_slice(
    eqn: JaxprEqn, read: ReadFn, write: WriteFn, get_var_size: GetSizeFn
) -> None:
    """Slice extracts elements [start:limit] - preserve element structure."""
    in_indices = read(eqn.invars[0])
    start = eqn.params["start_indices"]
    limit = eqn.params["limit_indices"]
    if len(start) == 1:
        out_indices = in_indices[start[0] : limit[0]]
    else:
        # Multi-dimensional: conservative fallback
        all_deps = set().union(*in_indices)
        out_size = get_var_size(eqn.outvars[0])
        out_indices = [all_deps.copy() for _ in range(out_size)]
    write(eqn.outvars[0], out_indices)


def _propagate_squeeze(
    eqn: JaxprEqn, read: ReadFn, write: WriteFn, get_var_size: GetSizeFn
) -> None:
    """Squeeze removes size-1 dims, preserves element dependencies."""
    write(eqn.outvars[0], read(eqn.invars[0]))


def _propagate_broadcast_in_dim(
    eqn: JaxprEqn, read: ReadFn, write: WriteFn, get_var_size: GetSizeFn
) -> None:
    """Broadcast: replicate dependencies to match output shape."""
    in_indices = read(eqn.invars[0])
    out_shape = eqn.params["shape"]
    out_size = _shape_size(out_shape)
    if len(in_indices) == 1:
        write(eqn.outvars[0], [in_indices[0].copy() for _ in range(out_size)])
    else:
        # Array broadcast: conservative (could be smarter)
        all_deps = set().union(*in_indices)
        write(eqn.outvars[0], [all_deps.copy() for _ in range(out_size)])


def _propagate_concatenate(
    eqn: JaxprEqn, read: ReadFn, write: WriteFn, get_var_size: GetSizeFn
) -> None:
    """Concatenate: join element lists in order."""
    out_indices: IndexSets = []
    for invar in eqn.invars:
        out_indices.extend(read(invar))
    write(eqn.outvars[0], out_indices)


def _propagate_reshape(
    eqn: JaxprEqn, read: ReadFn, write: WriteFn, get_var_size: GetSizeFn
) -> None:
    """Reshape preserves total elements and their dependencies."""
    in_indices = read(eqn.invars[0])
    out_size = get_var_size(eqn.outvars[0])
    if len(in_indices) == out_size:
        write(eqn.outvars[0], in_indices)
    else:
        # Size mismatch: conservative
        all_deps = set().union(*in_indices)
        write(eqn.outvars[0], [all_deps.copy() for _ in range(out_size)])


def _propagate_integer_pow(
    eqn: JaxprEqn, read: ReadFn, write: WriteFn, get_var_size: GetSizeFn
) -> None:
    """x^n: element-wise, preserves structure (unless n=0)."""
    power = eqn.params.get("y", 1)
    in_indices = read(eqn.invars[0])
    if power == 0:
        write(eqn.outvars[0], [set() for _ in range(len(in_indices))])
    else:
        write(eqn.outvars[0], [s.copy() for s in in_indices])


def _propagate_binary_elementwise(
    eqn: JaxprEqn, read: ReadFn, write: WriteFn, get_var_size: GetSizeFn
) -> None:
    """Binary element-wise ops: merge corresponding elements."""
    in1 = read(eqn.invars[0])
    in2 = read(eqn.invars[1])
    out_size = max(len(in1), len(in2))
    out_indices: IndexSets = []
    for i in range(out_size):
        deps: set[int] = set()
        # Handle broadcasting: scalars apply to all
        if len(in1) == 1:
            deps |= in1[0]
        elif i < len(in1):
            deps |= in1[i]
        if len(in2) == 1:
            deps |= in2[0]
        elif i < len(in2):
            deps |= in2[i]
        out_indices.append(deps)
    write(eqn.outvars[0], out_indices)


def _propagate_unary_elementwise(
    eqn: JaxprEqn, read: ReadFn, write: WriteFn, get_var_size: GetSizeFn
) -> None:
    """Unary element-wise ops: preserve element structure."""
    in_indices = read(eqn.invars[0])
    write(eqn.outvars[0], [s.copy() for s in in_indices])


def _propagate_reduce_sum(
    eqn: JaxprEqn, read: ReadFn, write: WriteFn, get_var_size: GetSizeFn
) -> None:
    """Reduction: output depends on all input elements."""
    in_indices = read(eqn.invars[0])
    all_deps = set().union(*in_indices)
    write(eqn.outvars[0], [all_deps])


def _propagate_convert_element_type(
    eqn: JaxprEqn, read: ReadFn, write: WriteFn, get_var_size: GetSizeFn
) -> None:
    """Type conversion: preserve dependencies."""
    in_indices = read(eqn.invars[0])
    write(eqn.outvars[0], [s.copy() for s in in_indices])


def _propagate_default(
    eqn: JaxprEqn, read: ReadFn, write: WriteFn, get_var_size: GetSizeFn
) -> None:
    """Default fallback: union all input deps for all outputs."""
    all_deps: set[int] = set()
    for invar in eqn.invars:
        for s in read(invar):
            all_deps |= s
    for outvar in eqn.outvars:
        out_size = get_var_size(outvar)
        write(outvar, [all_deps.copy() for _ in range(out_size)])


def _get_var_size(v: Var | Literal) -> int:
    """Get the total number of elements in a variable."""
    if isinstance(v, Literal):
        val = v.val
        shape = getattr(val, "shape", ())
        return _shape_size(tuple(shape)) if shape else 1
    aval = v.aval
    shape = getattr(aval, "shape", ())
    return _shape_size(tuple(shape)) if shape else 1


def _propagate_jaxpr(jaxpr: Jaxpr, input_indices: list[IndexSets]) -> list[IndexSets]:
    """
    Propagate index sets through a jaxpr.

    Args:
        jaxpr: The jaxpr to analyze
        input_indices: List of IndexSets, one per input variable

    Returns:
        List of IndexSets, one per output variable
    """
    env: dict[Var, IndexSets] = {}

    def read(v: Var | Literal) -> IndexSets:
        if isinstance(v, Literal):
            size = _get_var_size(v)
            return [set() for _ in range(size)]
        return env.get(v, [set()])

    def write(v: Var, indices: IndexSets) -> None:
        env[v] = indices

    # Initialize input variables
    for var, indices in zip(jaxpr.invars, input_indices, strict=False):
        write(var, indices)

    # Process each equation
    for eqn in jaxpr.eqns:
        _propagate_equation(eqn, read, write)

    # Return output dependencies
    return [read(outvar) for outvar in jaxpr.outvars]


def _propagate_nested_jaxpr(
    eqn: JaxprEqn, read: ReadFn, write: WriteFn, get_var_size: GetSizeFn
) -> None:
    """Handle primitives with nested jaxprs by recursively tracing."""
    # Get the nested jaxpr from params
    nested_jaxpr = eqn.params.get("jaxpr")
    if nested_jaxpr is None:
        # Fallback if we can't find the jaxpr
        _propagate_default(eqn, read, write, get_var_size)
        return

    # Handle ClosedJaxpr wrapper
    if hasattr(nested_jaxpr, "jaxpr"):
        nested_jaxpr = nested_jaxpr.jaxpr

    # Gather input dependencies
    input_indices = [read(invar) for invar in eqn.invars]

    # Recursively propagate through nested jaxpr
    output_indices = _propagate_jaxpr(nested_jaxpr, input_indices)

    # Write output dependencies
    for outvar, indices in zip(eqn.outvars, output_indices, strict=False):
        write(outvar, indices)


def _propagate_equation(eqn: JaxprEqn, read: ReadFn, write: WriteFn) -> None:
    """Propagate dependencies through a single equation."""
    match eqn.primitive.name:
        case prim if prim in ZERO_DERIVATIVE_PRIMITIVES:
            _propagate_zero_derivative(eqn, read, write, _get_var_size)
        case prim if prim in NESTED_JAXPR_PRIMITIVES:
            _propagate_nested_jaxpr(eqn, read, write, _get_var_size)
        case "slice":
            _propagate_slice(eqn, read, write, _get_var_size)
        case "squeeze":
            _propagate_squeeze(eqn, read, write, _get_var_size)
        case "broadcast_in_dim":
            _propagate_broadcast_in_dim(eqn, read, write, _get_var_size)
        case "concatenate":
            _propagate_concatenate(eqn, read, write, _get_var_size)
        case "reshape":
            _propagate_reshape(eqn, read, write, _get_var_size)
        case "integer_pow":
            _propagate_integer_pow(eqn, read, write, _get_var_size)
        case "add" | "sub" | "mul" | "div" | "pow" | "max" | "min":
            _propagate_binary_elementwise(eqn, read, write, _get_var_size)
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
            _propagate_unary_elementwise(eqn, read, write, _get_var_size)
        case "reduce_sum":
            _propagate_reduce_sum(eqn, read, write, _get_var_size)
        case "convert_element_type":
            _propagate_convert_element_type(eqn, read, write, _get_var_size)
        case _:
            _propagate_default(eqn, read, write, _get_var_size)


def jacobian_sparsity(f, n: int) -> coo_matrix:
    """
    Detect GLOBAL Jacobian sparsity pattern for f: R^n -> R^m.

    This analyzes the computation graph structure directly, without evaluating
    any derivatives. The result is valid for ALL inputs (global sparsity).

    The approach:
    1. Get the jaxpr (computation graph) for the function
    2. Propagate per-element index sets through each primitive
    3. Build the sparsity pattern from output dependencies

    Args:
        f: Function taking a 1D array of length n
        n: Input dimension

    Returns:
        Sparse boolean matrix of shape (m, n) where entry (i,j) is True
        if output i depends on input j
    """
    dummy_input = jnp.zeros(n)
    closed_jaxpr = jax.make_jaxpr(f)(dummy_input)
    jaxpr = closed_jaxpr.jaxpr
    m = int(jax.eval_shape(f, dummy_input).size)

    # Initialize: input element i depends on input index i
    input_indices = [[{i} for i in range(n)]]

    # Propagate through the jaxpr
    output_indices_list = _propagate_jaxpr(jaxpr, input_indices)

    # Extract output dependencies (first output variable)
    out_indices = output_indices_list[0] if output_indices_list else []

    # Build sparse matrix
    rows = []
    cols = []
    for i, deps in enumerate(out_indices):
        for j in deps:
            rows.append(i)
            cols.append(j)

    return coo_matrix(([True] * len(rows), (rows, cols)), shape=(m, n), dtype=bool)

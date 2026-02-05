"""Propagate index sets through a jaxpr to determine sparsity.

Each JAX primitive has a handler that maps input index sets to output index sets.
For example, element-wise ops preserve per-element dependencies, while reductions
union all input dependencies into a single output.

The main entry point is `prop_jaxpr`, which walks the computation graph
and applies the appropriate handler for each equation.
"""

import math
from collections.abc import Sequence
from itertools import product

from jax._src.core import Jaxpr, JaxprEqn, Literal, Var

# Type aliases
IndexSets = list[set[int]]
# Maps each variable to its per-element dependency index sets
Deps = dict[Var, IndexSets]
# In jaxpressions, "atom"s are the atomic elements that can appear as inputs to equations:
#  * Var: a named intermediate value
#  * Literal: a constant
Atom = Var | Literal


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


def union_all(sets: Sequence[set[int]]) -> set[int]:
    """Union all sets together, returning a new set."""
    if not sets:
        return set()
    result: set[int] = set()
    for s in sets:
        result |= s
    return result


def numel(shape: Sequence[int]) -> int:
    """Compute the total number of elements from a shape tuple."""
    return math.prod(shape) if shape else 1


def atom_numel(atom: Atom) -> int:
    """Get the total number of elements in a variable or literal."""
    if isinstance(atom, Literal):
        shape = getattr(atom.val, "shape", ())
        return numel(tuple(shape)) if shape else 1
    shape = getattr(atom.aval, "shape", ())
    return numel(tuple(shape)) if shape else 1


def index_sets(deps: Deps, atom: Atom) -> IndexSets:
    """Get the index sets for a variable or literal."""
    if isinstance(atom, Literal):
        return [set() for _ in range(atom_numel(atom))]
    return deps.get(atom, [set()])


def row_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """Compute row-major strides for multi-dimensional index tracking.

    Used to convert between flat indices and coordinates when propagating
    dependencies through slice and broadcast_in_dim. Each stride tells how
    many flat elements to skip when incrementing one coordinate position.

    For shape (2, 3, 4): row_strides = (12, 4, 1) since moving one step in dim 0
    skips 3*4=12 elements, dim 1 skips 4 elements, and dim 2 skips 1 element.
    """
    result: list[int] = []
    stride = 1
    for dim in reversed(shape):
        result.append(stride)
        stride *= dim
    return tuple(reversed(result))


# =============================================================================
# Handle Jaxpr
# =============================================================================


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
            "Please report this at https://github.com/adrhill/detex/issues"
        )
        raise ValueError(msg)

    # Handle ClosedJaxpr wrapper
    if hasattr(nested_jaxpr, "jaxpr"):
        nested_jaxpr = nested_jaxpr.jaxpr

    input_indices = [index_sets(deps, invar) for invar in eqn.invars]
    output_indices = prop_jaxpr(nested_jaxpr, input_indices)

    for outvar, indices in zip(eqn.outvars, output_indices, strict=False):
        deps[outvar] = indices


# =============================================================================
# Dispatch on JaxprEqn
# =============================================================================


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
        case "add" | "sub" | "mul" | "div" | "pow" | "max" | "min":
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
        "Please report this at https://github.com/adrhill/detex/issues"
    )
    raise NotImplementedError(msg)


# =============================================================================
# Propagation rules
# =============================================================================


def prop_zero_derivative(eqn: JaxprEqn, deps: Deps) -> None:
    """Propagate dependencies through zero-derivative primitives.

    Operations like floor, ceil, comparisons (eq, lt, gt, ...), and sign
    have zero derivative almost everywhere.
    Their outputs are piecewise constant,
    so infinitesimal input changes don't affect outputs.

    Mathematically, for f in {floor, ceil, sign, eq, ...}:
        ∂f/∂x = 0  (almost everywhere)
    Therefore, output elements have no dependencies on input elements.

    Example: y = floor(x) where x = [1.7, 2.3, 3.9]
        Input deps:  [{0}, {1}, {2}]
        Output deps: [{}, {}, {}]  (empty sets, no dependence)
    """
    for outvar in eqn.outvars:
        deps[outvar] = [set() for _ in range(atom_numel(outvar))]


def prop_slice(eqn: JaxprEqn, deps: Deps) -> None:
    """Slicing extracts a contiguous (possibly strided) subarray.
    Each output element maps to exactly one input element,
    so dependencies pass through unchanged.

    For slice with start indices s, strides t:
        out[i, j, ...] = in[s₀ + i·t₀, s₁ + j·t₁, ...]
    The Jacobian is a selection matrix with exactly one 1 per row.

    Example: x = [a, b, c, d, e], y = x[1:4:2] = [b, d]
        Input deps:  [{0}, {1}, {2}, {3}, {4}]
        Output deps: [{1}, {3}]  (indices 1 and 3 from input)

    Jaxpr:
        invars[0]: input array
        start_indices: tuple of start indices per dimension
        limit_indices: tuple of end indices per dimension
        strides: tuple of step sizes per dimension (default: all 1s)
    """
    in_indices = index_sets(deps, eqn.invars[0])
    start = eqn.params["start_indices"]
    limit = eqn.params["limit_indices"]
    slice_strides = eqn.params.get("strides") or tuple(1 for _ in start)

    in_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))
    out_shape = tuple(
        (limit[d] - start[d] + slice_strides[d] - 1) // slice_strides[d]
        for d in range(len(start))
    )

    in_strides = row_strides(in_shape)
    out_strides = row_strides(out_shape)
    out_size = numel(out_shape)

    out_indices: IndexSets = []
    for out_flat in range(out_size):
        # Convert flat output index to output coordinates
        out_coord = []
        remaining = out_flat
        for s in out_strides:
            out_coord.append(remaining // s)
            remaining %= s

        # Map to input coordinates: in_coord[d] = start[d] + out_coord[d] * slice_strides[d]
        in_flat = sum(
            (start[d] + out_coord[d] * slice_strides[d]) * in_strides[d]
            for d in range(len(start))
        )
        out_indices.append(in_indices[in_flat].copy())

    deps[eqn.outvars[0]] = out_indices


def prop_squeeze(eqn: JaxprEqn, deps: Deps) -> None:
    """Squeeze removes dimensions of size 1 without changing the data.
    Since it's a reshape with the same number of elements,
    dependencies pass through unchanged in flat order.

    For input shape (2, 1, 3) with squeeze on dim 1:
        out[i, k] = in[i, 0, k]
    The Jacobian is the identity matrix (permuted).

    Example: x.shape = (2, 1), y = squeeze(x) with shape (2,)
        Input deps:  [{0}, {1}]
        Output deps: [{0}, {1}]

    Jaxpr:
        invars[0]: input array
        dimensions: axes to squeeze (must have size 1)
    """
    deps[eqn.outvars[0]] = index_sets(deps, eqn.invars[0])


def prop_broadcast_in_dim(eqn: JaxprEqn, deps: Deps) -> None:
    """Broadcast replicates input elements across new or expanded dimensions.
    Each output element depends on exactly one input element,
    determined by projecting output coordinates onto input dimensions.

    For broadcast_dimensions mapping input dim i → output dim d[i]:
        out[..., j, ...] = in[..., j mod in_shape[i], ...]
    Size-1 input dims are implicitly broadcast (all outputs read index 0).

    Example: x.shape = (3,), y = broadcast(x, shape=(2, 3), dims=(1,))
        Input deps:  [{0}, {1}, {2}]
        Output deps: [{0}, {1}, {2}, {0}, {1}, {2}]  (repeated per row)

    Jaxpr:
        invars[0]: input array
        shape: target output shape
        broadcast_dimensions: maps input dim i to output dim
    """
    in_indices = index_sets(deps, eqn.invars[0])
    out_shape = eqn.params["shape"]
    broadcast_dims = eqn.params["broadcast_dimensions"]
    out_size = numel(out_shape)

    # Scalar case: single input dependency applies to all outputs
    if len(in_indices) == 1:
        deps[eqn.outvars[0]] = [in_indices[0].copy() for _ in range(out_size)]
        return

    in_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))
    in_strides = row_strides(in_shape)
    out_strides = row_strides(out_shape)

    out_indices: IndexSets = []
    for out_flat in range(out_size):
        # Convert flat output index to output coordinates
        out_coord = []
        remaining = out_flat
        for s in out_strides:
            out_coord.append(remaining // s)
            remaining %= s

        # Map to input coordinates using broadcast_dimensions.
        # broadcast_dims[i] = which output dim corresponds to input dim i.
        # Size-1 input dims are replicated: input (3,1) -> output (3,2) means
        # out[i,0] and out[i,1] both come from in[i,0], so we clamp to 0.
        in_flat = sum(
            (out_coord[broadcast_dims[i]] if in_shape[i] > 1 else 0) * in_strides[i]
            for i in range(len(in_shape))
        )
        out_indices.append(in_indices[in_flat].copy())

    deps[eqn.outvars[0]] = out_indices


def prop_concatenate(eqn: JaxprEqn, deps: Deps) -> None:
    """Concatenate joins arrays along a specified axis.
    Each output element comes from exactly one input element.

    For concat([A, B], axis=0): output = [A; B] (vertical stack).
    For concat([A, B], axis=1): output = [A | B] (horizontal stack).
    The Jacobian is a permuted identity matrix.

    Example: concat([[a,b], [c,d]], axis=0) → [a,b,c,d]
        Input deps:  [{0}, {1}], [{2}, {3}]
        Output deps: [{0}, {1}, {2}, {3}]

    Jaxpr:
        invars: list of input arrays to concatenate
        dimension: axis along which to concatenate
    """
    dim = eqn.params["dimension"]

    # Concat along dim 0: flat arrays are contiguous, just append
    if dim == 0:
        out_indices: IndexSets = []
        for invar in eqn.invars:
            out_indices.extend(index_sets(deps, invar))
        deps[eqn.outvars[0]] = out_indices
        return

    # Inner dimension: output coord along `dim` determines which input it's from.
    # E.g., concat([A(2x1), B(2x1)], dim=1) -> C(2x2): C[i,0] from A, C[i,1] from B.
    out_shape = tuple(getattr(eqn.outvars[0].aval, "shape", ()))
    in_shapes = [tuple(getattr(iv.aval, "shape", ())) for iv in eqn.invars]
    in_dim_sizes = [s[dim] for s in in_shapes]

    # dim_offsets[i] = starting position of input i along concat dimension
    dim_offsets = [sum(in_dim_sizes[:i]) for i in range(len(in_dim_sizes) + 1)]

    out_strides = row_strides(out_shape)
    all_in_indices = [index_sets(deps, iv) for iv in eqn.invars]
    all_in_strides = [row_strides(s) for s in in_shapes]

    out_indices = []
    for out_flat in range(numel(out_shape)):
        out_coord = []
        remaining = out_flat
        for s in out_strides:
            out_coord.append(remaining // s)
            remaining %= s

        # Find which input owns this position along the concat dimension
        pos_along_dim = out_coord[dim]
        for i in range(len(eqn.invars)):
            if dim_offsets[i] <= pos_along_dim < dim_offsets[i + 1]:
                in_coord = list(out_coord)
                in_coord[dim] = pos_along_dim - dim_offsets[i]
                in_flat = sum(
                    c * s for c, s in zip(in_coord, all_in_strides[i], strict=True)
                )
                out_indices.append(all_in_indices[i][in_flat].copy())
                break

    deps[eqn.outvars[0]] = out_indices


def prop_reshape(eqn: JaxprEqn, deps: Deps) -> None:
    """Reshape changes array shape without changing data or element count.
    Dependencies pass through unchanged in row-major (C) order.
    The Jacobian is the identity matrix.

    Example: reshape([a,b,c,d], (2,2)) → [[a,b],[c,d]]
        Input deps:  [{0}, {1}, {2}, {3}]
        Output deps: [{0}, {1}, {2}, {3}]

    Jaxpr:
        invars[0]: input array
        new_sizes: target shape
        dimensions: optional axis permutation before reshape
    """
    in_indices = index_sets(deps, eqn.invars[0])
    out_size = atom_numel(eqn.outvars[0])
    if len(in_indices) == out_size:
        deps[eqn.outvars[0]] = in_indices
    else:
        # TODO: Investigate when size mismatch occurs and handle precisely.
        # Conservative fallback: union all input dependencies.
        all_deps = union_all(in_indices)
        deps[eqn.outvars[0]] = [all_deps.copy() for _ in range(out_size)]


def prop_integer_pow(eqn: JaxprEqn, deps: Deps) -> None:
    """Integer power x^n is element-wise.
    Each output depends only on the corresponding input element.
    Special case: x^0 = 1 has zero derivative, so no dependencies.

    ∂(x^n)/∂x = n·x^(n-1), which is zero iff n = 0.

    Example: y = x^2 where x = [a, b, c]
        Input deps:  [{0}, {1}, {2}]
        Output deps: [{0}, {1}, {2}]  (or [{}, {}, {}] if n=0)

    Jaxpr:
        invars[0]: input array
        y: the integer exponent
    """
    in_indices = index_sets(deps, eqn.invars[0])
    if eqn.params.get("y", 1) == 0:
        deps[eqn.outvars[0]] = [set() for _ in range(len(in_indices))]
    else:
        deps[eqn.outvars[0]] = [s.copy() for s in in_indices]


def prop_binary_elementwise(eqn: JaxprEqn, deps: Deps) -> None:
    """Binary element-wise ops (add, mul, etc.) combine two arrays element-wise.
    Each output element depends on the corresponding elements from both inputs.
    Broadcasting is handled: scalars contribute to all output elements.

    For f(x, y) element-wise:
        ∂f/∂x[i] and ∂f/∂y[i] are generally nonzero
    So out[i] depends on {x[i], y[i]} (union of dependencies).

    Example: z = x + y where x = [a, b], y = [c, d]
        Input deps:  [{0}, {1}], [{2}, {3}]
        Output deps: [{0, 2}, {1, 3}]

    Jaxpr:
        invars[0]: first input array
        invars[1]: second input array
    """
    in1 = index_sets(deps, eqn.invars[0])
    in2 = index_sets(deps, eqn.invars[1])
    out_size = max(len(in1), len(in2))
    out_indices: IndexSets = []
    for i in range(out_size):
        to_merge: IndexSets = []
        # Handle broadcasting: scalars apply to all
        if len(in1) == 1:
            to_merge.append(in1[0])
        elif i < len(in1):
            to_merge.append(in1[i])
        if len(in2) == 1:
            to_merge.append(in2[0])
        elif i < len(in2):
            to_merge.append(in2[i])
        out_indices.append(union_all(to_merge))
    deps[eqn.outvars[0]] = out_indices


def prop_unary_elementwise(eqn: JaxprEqn, deps: Deps) -> None:
    """Unary element-wise ops (exp, sin, etc.) apply a function to each element.
    Each output depends only on the corresponding input element.
    The Jacobian is diagonal.

    For f(x) element-wise:
        ∂f[i]/∂x[j] = f'(x[i]) if i = j, else 0

    Example: y = exp(x) where x = [a, b, c]
        Input deps:  [{0}, {1}, {2}]
        Output deps: [{0}, {1}, {2}]

    Jaxpr:
        invars[0]: input array
    """
    deps[eqn.outvars[0]] = [s.copy() for s in index_sets(deps, eqn.invars[0])]


def prop_reduce_sum(eqn: JaxprEqn, deps: Deps) -> None:
    """Sum reduction aggregates elements along specified axes.
    Each output depends on all input elements that were summed into it.

    Full reduction (no axes or all axes):
        out = Σᵢ x[i]  →  out depends on all inputs
    Partial reduction along axis k:
        out[i] = Σⱼ x[i, j]  →  out[i] depends on row i of input

    Example: y = sum(x, axis=1) where x.shape = (2, 3)
        Input deps:  [{0}, {1}, {2}, {3}, {4}, {5}]
        Output deps: [{0, 1, 2}, {3, 4, 5}]  (one set per row)

    Jaxpr:
        invars[0]: input array
        axes: tuple of axes to reduce (empty = full reduction)
    """
    in_indices = index_sets(deps, eqn.invars[0])
    axes = eqn.params.get("axes", ())
    in_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))

    # Full reduction: single output depends on all inputs
    if not axes or len(axes) == len(in_shape):
        deps[eqn.outvars[0]] = [union_all(in_indices)]
        return

    # Partial reduction: group input elements by their non-reduced coordinates
    out_shape = tuple(s for i, s in enumerate(in_shape) if i not in axes)
    in_strides = row_strides(in_shape)
    out_strides = row_strides(out_shape)
    out_size = numel(out_shape)

    out_indices: IndexSets = [set() for _ in range(out_size)]

    for in_flat, elem_deps in enumerate(in_indices):
        # Convert to input coordinates
        in_coord = []
        remaining = in_flat
        for s in in_strides:
            in_coord.append(remaining // s)
            remaining %= s

        # Project to output coordinates (drop reduced dimensions)
        out_coord = [c for i, c in enumerate(in_coord) if i not in axes]
        out_flat = sum(c * s for c, s in zip(out_coord, out_strides, strict=True))
        out_indices[out_flat] |= elem_deps

    deps[eqn.outvars[0]] = out_indices


def prop_convert_element_type(eqn: JaxprEqn, deps: Deps) -> None:
    """Type conversion (e.g., float32 → float64) changes dtype without changing values.
    Dependencies pass through unchanged.
    The Jacobian is the identity matrix.

    Example: y = x.astype(float64) where x = [a, b, c]
        Input deps:  [{0}, {1}, {2}]
        Output deps: [{0}, {1}, {2}]

    Jaxpr:
        invars[0]: input array
        new_dtype: target dtype
    """
    deps[eqn.outvars[0]] = [s.copy() for s in index_sets(deps, eqn.invars[0])]


def prop_conv_general_dilated(eqn: JaxprEqn, deps: Deps) -> None:
    """Convolution slides a kernel over the input, computing weighted sums.
    Each output element depends on a local spatial window of input elements
    across all input channels.

    For 2D conv with kernel size (kH, kW), stride s, and C_in input channels:
        out[n, h, w, c_out] = Σ_{kh, kw, c_in} in[n, h·s + kh, w·s + kw, c_in] · W[...]
    So out[n, h, w, :] depends on in[n, h·s : h·s+kH, w·s : w·s+kW, :].

    Example: 1D conv, kernel size 2, input [a, b, c, d]
        out[0] = a·w0 + b·w1  →  deps {0, 1}
        out[1] = b·w0 + c·w1  →  deps {1, 2}
        out[2] = c·w0 + d·w1  →  deps {2, 3}

    Jaxpr:
        invars[0]: input (lhs), invars[1]: kernel (rhs)
        dimension_numbers: specifies layout (batch, feature, spatial dims)
        window_strides, padding, lhs_dilation, rhs_dilation: conv parameters
    """
    lhs_indices = index_sets(deps, eqn.invars[0])  # Input image dependencies

    # Get shapes from avals
    lhs_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))
    rhs_shape = tuple(getattr(eqn.invars[1].aval, "shape", ()))
    out_shape = tuple(getattr(eqn.outvars[0].aval, "shape", ()))

    # Parse dimension numbers
    dim_nums = eqn.params["dimension_numbers"]
    lhs_spec, rhs_spec, out_spec = (
        dim_nums.lhs_spec,
        dim_nums.rhs_spec,
        dim_nums.out_spec,
    )

    # Extract dimension indices
    lhs_batch_dim, lhs_feature_dim = lhs_spec[0], lhs_spec[1]
    lhs_spatial_dims = lhs_spec[2:]
    out_batch_dim = out_spec[0]
    out_spatial_dims = out_spec[2:]
    rhs_spatial_dims = rhs_spec[2:]

    # Get parameters
    n_spatial = len(lhs_spatial_dims)
    window_strides = eqn.params.get("window_strides", (1,) * n_spatial)
    lhs_dilation = eqn.params.get("lhs_dilation", (1,) * n_spatial)
    rhs_dilation = eqn.params.get("rhs_dilation", (1,) * n_spatial)
    padding = eqn.params.get("padding", ((0, 0),) * n_spatial)

    lhs_strides = row_strides(lhs_shape)
    out_strides = row_strides(out_shape)

    # Get spatial sizes
    lhs_spatial_sizes = [lhs_shape[d] for d in lhs_spatial_dims]
    kernel_spatial_sizes = [rhs_shape[d] for d in rhs_spatial_dims]
    n_in_features = lhs_shape[lhs_feature_dim]

    out_indices: IndexSets = []

    for out_flat in range(numel(out_shape)):
        # Convert flat output index to coordinates
        out_coord = []
        remaining = out_flat
        for s in out_strides:
            out_coord.append(remaining // s)
            remaining %= s

        batch_idx = out_coord[out_batch_dim]
        out_spatial_coord = [out_coord[d] for d in out_spatial_dims]

        # Collect dependencies from input
        elem_deps: set[int] = set()

        # For each position in the kernel window
        for kernel_offsets in product(*[range(k) for k in kernel_spatial_sizes]):
            # Compute input spatial coordinates
            in_spatial_coord = []
            valid = True
            for i in range(n_spatial):
                in_c = (
                    out_spatial_coord[i] * window_strides[i]
                    + kernel_offsets[i] * rhs_dilation[i]
                    - padding[i][0]
                )
                if in_c < 0 or in_c >= lhs_spatial_sizes[i] * lhs_dilation[i]:
                    valid = False
                    break
                if lhs_dilation[i] > 1 and in_c % lhs_dilation[i] != 0:
                    valid = False
                    break
                in_spatial_coord.append(in_c // lhs_dilation[i])

            if not valid:
                continue

            # For each input feature channel
            for in_feature_idx in range(n_in_features):
                in_coord = [0] * len(lhs_shape)
                in_coord[lhs_batch_dim] = batch_idx
                in_coord[lhs_feature_dim] = in_feature_idx
                for i, d in enumerate(lhs_spatial_dims):
                    in_coord[d] = in_spatial_coord[i]

                in_flat = sum(c * s for c, s in zip(in_coord, lhs_strides, strict=True))
                if in_flat < len(lhs_indices):
                    elem_deps |= lhs_indices[in_flat]

        out_indices.append(elem_deps)

    deps[eqn.outvars[0]] = out_indices

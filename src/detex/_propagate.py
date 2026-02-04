"""Propagate index sets through a jaxpr to determine sparsity.

Each JAX primitive has a handler that maps input index sets to output index sets.
For example, element-wise ops preserve per-element dependencies, while reductions
union all input dependencies into a single output.

The main entry point is `_propagate_jaxpr`, which walks the computation graph
and applies the appropriate handler for each equation.
"""

import math
from collections.abc import Sequence
from itertools import product

from jax._src.core import Jaxpr, JaxprEqn, Literal, Var

# Type aliases
IndexSets = list[set[int]]
Env = dict[Var, IndexSets]  # Maps each variable to its per-element dependency sets


def _union_all(sets: Sequence[set[int]]) -> set[int]:
    """Union all sets together, returning a new set."""
    if not sets:
        return set()
    result: set[int] = set()
    for s in sets:
        result |= s
    return result


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


def _compute_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """Compute row-major strides for multi-dimensional index tracking.

    Used to convert between flat indices and coordinates when propagating
    dependencies through slice and broadcast_in_dim. Each stride tells how
    many flat elements to skip when incrementing one coordinate position.

    For shape (2, 3, 4): strides = (12, 4, 1) since moving one step in dim 0
    skips 3*4=12 elements, dim 1 skips 4 elements, and dim 2 skips 1 element.
    """
    result: list[int] = []
    stride = 1
    for dim in reversed(shape):
        result.append(stride)
        stride *= dim
    return tuple(reversed(result))


def _get_size(atom: Var | Literal) -> int:
    """Get the total number of elements in a variable or literal."""
    if isinstance(atom, Literal):
        shape = getattr(atom.val, "shape", ())
        return _shape_size(tuple(shape)) if shape else 1
    shape = getattr(atom.aval, "shape", ())
    return _shape_size(tuple(shape)) if shape else 1


def _get_idxs(env: Env, atom: Var | Literal) -> IndexSets:
    """Get the index sets for a variable or literal."""
    if isinstance(atom, Literal):
        return [set() for _ in range(_get_size(atom))]
    return env.get(atom, [set()])


def _propagate_zero_derivative(eqn: JaxprEqn, env: Env) -> None:
    """Zero-derivative ops: output has no dependence on inputs."""
    for outvar in eqn.outvars:
        env[outvar] = [set() for _ in range(_get_size(outvar))]


def _propagate_slice(eqn: JaxprEqn, env: Env) -> None:
    """Propagate dependencies through the slice primitive.

    Slice extracts arr[start:limit:stride] along each dimension. Each output
    element depends on exactly one input element, so we compute the flat index
    mapping: output[i,j,...] <- input[start + i*stride, start + j*stride, ...].
    """
    in_indices = _get_idxs(env, eqn.invars[0])
    start = eqn.params["start_indices"]
    limit = eqn.params["limit_indices"]
    strides = eqn.params.get("strides") or tuple(1 for _ in start)

    in_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))
    out_shape = tuple(
        (limit[d] - start[d] + strides[d] - 1) // strides[d] for d in range(len(start))
    )

    in_strides = _compute_strides(in_shape)
    out_strides = _compute_strides(out_shape)
    out_size = _shape_size(out_shape)

    out_indices: IndexSets = []
    for out_flat in range(out_size):
        # Convert flat output index to output coordinates
        out_coord = []
        remaining = out_flat
        for s in out_strides:
            out_coord.append(remaining // s)
            remaining %= s

        # Map to input coordinates: in_coord[d] = start[d] + out_coord[d] * strides[d]
        in_flat = sum(
            (start[d] + out_coord[d] * strides[d]) * in_strides[d]
            for d in range(len(start))
        )
        out_indices.append(in_indices[in_flat].copy())

    env[eqn.outvars[0]] = out_indices


def _propagate_squeeze(eqn: JaxprEqn, env: Env) -> None:
    """Squeeze removes size-1 dims, preserves element dependencies."""
    env[eqn.outvars[0]] = _get_idxs(env, eqn.invars[0])


def _propagate_broadcast_in_dim(eqn: JaxprEqn, env: Env) -> None:
    """Propagate dependencies through the broadcast_in_dim primitive.

    Broadcast replicates input elements across new or expanded dimensions.
    The broadcast_dimensions param maps input dim i to output dim broadcast_dims[i].
    Each output element depends on one input element determined by projecting
    the output coordinates onto the input dimensions.
    """
    in_indices = _get_idxs(env, eqn.invars[0])
    out_shape = eqn.params["shape"]
    broadcast_dims = eqn.params["broadcast_dimensions"]
    out_size = _shape_size(out_shape)

    # Scalar case: single input dependency applies to all outputs
    if len(in_indices) == 1:
        env[eqn.outvars[0]] = [in_indices[0].copy() for _ in range(out_size)]
        return

    in_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))
    in_strides = _compute_strides(in_shape)
    out_strides = _compute_strides(out_shape)

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

    env[eqn.outvars[0]] = out_indices


def _propagate_concatenate(eqn: JaxprEqn, env: Env) -> None:
    """Propagate dependencies through the concatenate primitive.

    Concat along dim 0 is simple append. For inner dims, we track which input
    each output coordinate comes from based on position along the concat axis.
    """
    dim = eqn.params["dimension"]

    # Concat along dim 0: flat arrays are contiguous, just append
    if dim == 0:
        out_indices: IndexSets = []
        for invar in eqn.invars:
            out_indices.extend(_get_idxs(env, invar))
        env[eqn.outvars[0]] = out_indices
        return

    # Inner dimension: output coord along `dim` determines which input it's from.
    # E.g., concat([A(2x1), B(2x1)], dim=1) -> C(2x2): C[i,0] from A, C[i,1] from B.
    out_shape = tuple(getattr(eqn.outvars[0].aval, "shape", ()))
    in_shapes = [tuple(getattr(iv.aval, "shape", ())) for iv in eqn.invars]
    in_dim_sizes = [s[dim] for s in in_shapes]

    # dim_offsets[i] = starting position of input i along concat dimension
    dim_offsets = [sum(in_dim_sizes[:i]) for i in range(len(in_dim_sizes) + 1)]

    out_strides = _compute_strides(out_shape)
    all_in_indices = [_get_idxs(env, iv) for iv in eqn.invars]
    all_in_strides = [_compute_strides(s) for s in in_shapes]

    out_indices = []
    for out_flat in range(_shape_size(out_shape)):
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

    env[eqn.outvars[0]] = out_indices


def _propagate_reshape(eqn: JaxprEqn, env: Env) -> None:
    """Reshape preserves total elements and their dependencies."""
    in_indices = _get_idxs(env, eqn.invars[0])
    out_size = _get_size(eqn.outvars[0])
    if len(in_indices) == out_size:
        env[eqn.outvars[0]] = in_indices
    else:
        # TODO: Investigate when size mismatch occurs and handle precisely.
        # Conservative fallback: union all input dependencies.
        all_deps = _union_all(in_indices)
        env[eqn.outvars[0]] = [all_deps.copy() for _ in range(out_size)]


def _propagate_integer_pow(eqn: JaxprEqn, env: Env) -> None:
    """x^n: element-wise, preserves structure (unless n=0)."""
    in_indices = _get_idxs(env, eqn.invars[0])
    if eqn.params.get("y", 1) == 0:
        env[eqn.outvars[0]] = [set() for _ in range(len(in_indices))]
    else:
        env[eqn.outvars[0]] = [s.copy() for s in in_indices]


def _propagate_binary_elementwise(eqn: JaxprEqn, env: Env) -> None:
    """Binary element-wise ops: merge corresponding elements."""
    in1 = _get_idxs(env, eqn.invars[0])
    in2 = _get_idxs(env, eqn.invars[1])
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
        out_indices.append(_union_all(to_merge))
    env[eqn.outvars[0]] = out_indices


def _propagate_unary_elementwise(eqn: JaxprEqn, env: Env) -> None:
    """Unary element-wise ops: preserve element structure."""
    env[eqn.outvars[0]] = [s.copy() for s in _get_idxs(env, eqn.invars[0])]


def _propagate_reduce_sum(eqn: JaxprEqn, env: Env) -> None:
    """Propagate dependencies through the reduce_sum primitive.

    Full reduction: jnp.sum(x) reduces all elements to a scalar. The single
    output depends on all inputs.

    Partial reduction: jnp.sum(x, axis=1) on shape (4,3) produces shape (4,).
    Each output row is the sum of that input row: out[i] depends on in[i,:].
    """
    in_indices = _get_idxs(env, eqn.invars[0])
    axes = eqn.params.get("axes", ())
    in_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))

    # Full reduction: single output depends on all inputs
    if not axes or len(axes) == len(in_shape):
        env[eqn.outvars[0]] = [_union_all(in_indices)]
        return

    # Partial reduction: group input elements by their non-reduced coordinates
    out_shape = tuple(s for i, s in enumerate(in_shape) if i not in axes)
    in_strides = _compute_strides(in_shape)
    out_strides = _compute_strides(out_shape)
    out_size = _shape_size(out_shape)

    out_indices: IndexSets = [set() for _ in range(out_size)]

    for in_flat, deps in enumerate(in_indices):
        # Convert to input coordinates
        in_coord = []
        remaining = in_flat
        for s in in_strides:
            in_coord.append(remaining // s)
            remaining %= s

        # Project to output coordinates (drop reduced dimensions)
        out_coord = [c for i, c in enumerate(in_coord) if i not in axes]
        out_flat = sum(c * s for c, s in zip(out_coord, out_strides, strict=True))
        out_indices[out_flat] |= deps

    env[eqn.outvars[0]] = out_indices


def _propagate_convert_element_type(eqn: JaxprEqn, env: Env) -> None:
    """Type conversion: preserve dependencies."""
    env[eqn.outvars[0]] = [s.copy() for s in _get_idxs(env, eqn.invars[0])]


def _propagate_conv_general_dilated(eqn: JaxprEqn, env: Env) -> None:
    """Convolution: each output depends on a spatial window of inputs.

    For a 2D conv with kernel (kH, kW) and C_in input channels:
    - Output at (n, h, w, c_out) depends on inputs at
      (n, h*stride_h + kh, w*stride_w + kw, c_in) for all kh, kw, c_in
    """
    lhs_indices = _get_idxs(env, eqn.invars[0])  # Input image dependencies

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
    strides = eqn.params.get("window_strides", (1,) * n_spatial)
    lhs_dilation = eqn.params.get("lhs_dilation", (1,) * n_spatial)
    rhs_dilation = eqn.params.get("rhs_dilation", (1,) * n_spatial)
    padding = eqn.params.get("padding", ((0, 0),) * n_spatial)

    # Compute strides for flat indexing
    def compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        result = []
        stride = 1
        for dim in reversed(shape):
            result.append(stride)
            stride *= dim
        return tuple(reversed(result))

    lhs_strides = compute_strides(lhs_shape)
    out_strides = compute_strides(out_shape)

    # Get spatial sizes
    lhs_spatial_sizes = [lhs_shape[d] for d in lhs_spatial_dims]
    kernel_spatial_sizes = [rhs_shape[d] for d in rhs_spatial_dims]
    n_in_features = lhs_shape[lhs_feature_dim]

    out_indices: IndexSets = []

    for out_flat in range(_shape_size(out_shape)):
        # Convert flat output index to coordinates
        out_coords = []
        remaining = out_flat
        for s in out_strides:
            out_coords.append(remaining // s)
            remaining %= s

        batch_idx = out_coords[out_batch_dim]
        out_spatial_coords = [out_coords[d] for d in out_spatial_dims]

        # Collect dependencies from input
        deps = set()

        # For each position in the kernel window
        for kernel_offsets in product(*[range(k) for k in kernel_spatial_sizes]):
            # Compute input spatial coordinates
            in_spatial_coords = []
            valid = True
            for i in range(n_spatial):
                in_coord = (
                    out_spatial_coords[i] * strides[i]
                    + kernel_offsets[i] * rhs_dilation[i]
                    - padding[i][0]
                )
                if in_coord < 0 or in_coord >= lhs_spatial_sizes[i] * lhs_dilation[i]:
                    valid = False
                    break
                if lhs_dilation[i] > 1 and in_coord % lhs_dilation[i] != 0:
                    valid = False
                    break
                in_spatial_coords.append(in_coord // lhs_dilation[i])

            if not valid:
                continue

            # For each input feature channel
            for in_feature_idx in range(n_in_features):
                in_coords = [0] * len(lhs_shape)
                in_coords[lhs_batch_dim] = batch_idx
                in_coords[lhs_feature_dim] = in_feature_idx
                for i, d in enumerate(lhs_spatial_dims):
                    in_coords[d] = in_spatial_coords[i]

                in_flat = sum(
                    c * s for c, s in zip(in_coords, lhs_strides, strict=True)
                )
                if in_flat < len(lhs_indices):
                    deps |= lhs_indices[in_flat]

        out_indices.append(deps)

    env[eqn.outvars[0]] = out_indices


def _propagate_default(eqn: JaxprEqn, env: Env) -> None:
    """Default conservative fallback for unhandled primitives.

    TODO: Add precise handlers for common primitives that hit this fallback,
    such as dot_general, gather, scatter, dynamic_slice, transpose, etc.
    """
    all_inputs: IndexSets = []
    for invar in eqn.invars:
        all_inputs.extend(_get_idxs(env, invar))
    all_deps = _union_all(all_inputs)
    for outvar in eqn.outvars:
        env[outvar] = [all_deps.copy() for _ in range(_get_size(outvar))]


def _propagate_nested_jaxpr(eqn: JaxprEqn, env: Env) -> None:
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

    input_indices = [_get_idxs(env, invar) for invar in eqn.invars]
    output_indices = _propagate_jaxpr(nested_jaxpr, input_indices)

    for outvar, indices in zip(eqn.outvars, output_indices, strict=False):
        env[outvar] = indices


def _propagate_equation(eqn: JaxprEqn, env: Env) -> None:
    """Propagate dependencies through a single equation."""
    match eqn.primitive.name:
        case prim if prim in ZERO_DERIVATIVE_PRIMITIVES:
            _propagate_zero_derivative(eqn, env)
        case prim if prim in NESTED_JAXPR_PRIMITIVES:
            _propagate_nested_jaxpr(eqn, env)
        case "slice":
            _propagate_slice(eqn, env)
        case "squeeze":
            _propagate_squeeze(eqn, env)
        case "broadcast_in_dim":
            _propagate_broadcast_in_dim(eqn, env)
        case "concatenate":
            _propagate_concatenate(eqn, env)
        case "reshape":
            _propagate_reshape(eqn, env)
        case "integer_pow":
            _propagate_integer_pow(eqn, env)
        case "add" | "sub" | "mul" | "div" | "pow" | "max" | "min":
            _propagate_binary_elementwise(eqn, env)
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
            _propagate_unary_elementwise(eqn, env)
        case "reduce_sum":
            _propagate_reduce_sum(eqn, env)
        case "convert_element_type":
            _propagate_convert_element_type(eqn, env)
        case "conv_general_dilated":
            _propagate_conv_general_dilated(eqn, env)
        case _:
            _propagate_default(eqn, env)


def _propagate_jaxpr(jaxpr: Jaxpr, input_indices: list[IndexSets]) -> list[IndexSets]:
    """
    Propagate index sets through a jaxpr.

    Args:
        jaxpr: The jaxpr to analyze
        input_indices: List of IndexSets, one per input variable

    Returns:
        List of IndexSets, one per output variable
    """
    env: Env = {}

    # Initialize input variables
    for var, indices in zip(jaxpr.invars, input_indices, strict=False):
        env[var] = indices

    # Process each equation
    for eqn in jaxpr.eqns:
        _propagate_equation(eqn, env)

    # Return output dependencies
    return [_get_idxs(env, outvar) for outvar in jaxpr.outvars]

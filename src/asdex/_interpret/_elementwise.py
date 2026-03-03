"""Propagation rules for element-wise operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    ConstVals,
    Deps,
    ValueBounds,
    atom_const_val,
    atom_numel,
    atom_shape,
    atom_value_bounds,
    copy_index_sets,
    empty_index_sets,
    index_sets,
    numel,
    propagate_const_binary,
)

# Ufuncs for evaluating constant values during tracing.
# Used to propagate static index values through arithmetic to gather/scatter.
_BINARY_CONST_UFUNCS: dict[str, np.ufunc] = {
    # arithmetic
    "add": np.add,
    "add_any": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "pow": np.power,
    "max": np.maximum,
    "min": np.minimum,
    "atan2": np.arctan2,
    "rem": np.remainder,
    "nextafter": np.nextafter,
    # comparison
    "eq": np.equal,
    "ne": np.not_equal,
    # bitwise
    "and": np.bitwise_and,
    "or": np.bitwise_or,
    "xor": np.bitwise_xor,
}


def _propagate_const(eqn: JaxprEqn, const_vals: ConstVals) -> None:
    """Propagate a const value through a binary elementwise op.

    If both inputs are statically known,
    apply the matching numpy ufunc and store the result.
    Without this, downstream handlers (e.g. ``gather``, ``scatter``) cannot resolve
    static index arrays and fall back to conservative.
    """
    ufunc = _BINARY_CONST_UFUNCS.get(eqn.primitive.name)
    if ufunc is not None:
        propagate_const_binary(eqn, const_vals, ufunc)


# Building blocks (private)


def _zero_derivative(eqn: JaxprEqn, deps: Deps) -> None:
    """Set empty index sets for zero-derivative outputs."""
    for outvar in eqn.outvars:
        deps[outvar] = empty_index_sets(atom_numel(outvar))


def _binary_elementwise(eqn: JaxprEqn, deps: Deps) -> None:
    """Union per-element index sets from two inputs."""
    in1 = index_sets(deps, eqn.invars[0])
    in2 = index_sets(deps, eqn.invars[1])
    out_size = 0 if len(in1) == 0 or len(in2) == 0 else max(len(in1), len(in2))

    in1_shape = atom_shape(eqn.invars[0])
    in2_shape = atom_shape(eqn.invars[1])

    # Fast path: same shape or scalar.
    # Modular indexing handles both correctly:
    # i % len == i for same size, i % 1 == 0 for scalar.
    if in1_shape == in2_shape or len(in1) <= 1 or len(in2) <= 1:
        deps[eqn.outvars[0]] = [
            in1[i % len(in1)] | in2[i % len(in2)] for i in range(out_size)
        ]
        return

    # General broadcast: map output coordinates to input coordinates
    # respecting numpy-style broadcasting (size-1 dims read index 0).
    # Example: mul of (16,16) * (16,1) → (16,16).
    # out[p,d] depends on in1[p,d] and in2[p,0], not in2[d].
    out_shape = atom_shape(eqn.outvars[0])
    out_size = numel(out_shape)
    out_coords = np.indices(out_shape)
    ndim = len(out_shape)

    def _broadcast_flat(in_shape: tuple[int, ...]) -> np.ndarray:
        # Left-pad with 1s to match output ndim (numpy broadcasting rule).
        pad = ndim - len(in_shape)
        padded = (1,) * pad + in_shape
        coords = tuple(out_coords[d] if padded[d] > 1 else 0 for d in range(ndim))
        return np.ravel_multi_index(coords, padded).ravel()

    in1_flat = _broadcast_flat(in1_shape)
    in2_flat = _broadcast_flat(in2_shape)

    deps[eqn.outvars[0]] = [
        in1[in1_flat[i]] | in2[in2_flat[i]] for i in range(out_size)
    ]


def _propagate_bounds_add(
    eqn: JaxprEqn,
    const_vals: ConstVals,
    value_bounds: ValueBounds,
) -> None:
    """Propagate value bounds through ``add`` or ``add_any`` via interval arithmetic."""
    b1 = atom_value_bounds(eqn.invars[0], const_vals, value_bounds)
    b2 = atom_value_bounds(eqn.invars[1], const_vals, value_bounds)
    if b1 is None or b2 is None:
        return
    lo1, hi1 = b1
    lo2, hi2 = b2
    value_bounds[eqn.outvars[0]] = (lo1 + lo2, hi1 + hi2)


def _propagate_bounds_sub(
    eqn: JaxprEqn,
    const_vals: ConstVals,
    value_bounds: ValueBounds,
) -> None:
    """Propagate value bounds through ``sub`` via interval arithmetic."""
    b1 = atom_value_bounds(eqn.invars[0], const_vals, value_bounds)
    b2 = atom_value_bounds(eqn.invars[1], const_vals, value_bounds)
    if b1 is None or b2 is None:
        return
    lo1, hi1 = b1
    lo2, hi2 = b2
    value_bounds[eqn.outvars[0]] = (lo1 - hi2, hi1 - lo2)


# Composite handlers (public)
# Each corresponds to exactly one dispatch case in prop_dispatch.


def prop_zero_derivative(eqn: JaxprEqn, deps: Deps) -> None:
    """Zero-derivative primitives (floor, ceil, sign, ...).

    Operations with zero derivative almost everywhere.
    Their outputs are piecewise constant,
    so infinitesimal input changes don't affect outputs.

    Mathematically, for f in {floor, ceil, sign, ...}:
        ∂f/∂x = 0  (almost everywhere)
    Therefore, output elements have no dependencies on input elements.

    Example: y = floor(x) where x = [1.7, 2.3, 3.9]
        Input deps:  [{0}, {1}, {2}]
        Output deps: [{}, {}, {}]  (empty sets, no dependence)
    """
    _zero_derivative(eqn, deps)


def prop_zero_derivative_const(
    eqn: JaxprEqn, deps: Deps, const_vals: ConstVals
) -> None:
    """Zero-derivative primitives that also propagate const values.

    Used for comparisons (eq, ne) and bitwise ops (and, or, xor)
    where the output is zero-derivative
    but the concrete result may be needed by downstream handlers.
    """
    _zero_derivative(eqn, deps)
    _propagate_const(eqn, const_vals)


def prop_binary_const(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Binary elementwise primitives (div, pow, max, min, ...) with const propagation.

    Each output element depends on the corresponding elements from both inputs.
    Also propagates const values for downstream index resolution.

    For f(x, y) element-wise:
        ∂f/∂x[i] and ∂f/∂y[i] are generally nonzero.
    So out[i] depends on {x[i], y[i]} (union of dependencies).

    Example: z = x + y where x = [a, b], y = [c, d]
        Input deps:  [{0}, {1}], [{2}, {3}]
        Output deps: [{0, 2}, {1, 3}]

    Jaxpr:
        invars[0]: first input array
        invars[1]: second input array
    """
    _binary_elementwise(eqn, deps)
    _propagate_const(eqn, const_vals)


def prop_add(
    eqn: JaxprEqn,
    deps: Deps,
    const_vals: ConstVals,
    value_bounds: ValueBounds,
) -> None:
    """Add / add_any: binary elementwise with interval arithmetic bounds.

    ``[a,b] + [c,d] = [a+c, b+d]``.
    """
    _binary_elementwise(eqn, deps)
    _propagate_const(eqn, const_vals)
    _propagate_bounds_add(eqn, const_vals, value_bounds)


def prop_sub(
    eqn: JaxprEqn,
    deps: Deps,
    const_vals: ConstVals,
    value_bounds: ValueBounds,
) -> None:
    """Sub: binary elementwise with interval arithmetic bounds.

    ``[a,b] - [c,d] = [a-d, b-c]``.
    """
    _binary_elementwise(eqn, deps)
    _propagate_const(eqn, const_vals)
    _propagate_bounds_sub(eqn, const_vals, value_bounds)


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

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.integer_pow.html
    """
    in_indices = index_sets(deps, eqn.invars[0])
    if eqn.params.get("y", 1) == 0:
        deps[eqn.outvars[0]] = empty_index_sets(len(in_indices))
    else:
        deps[eqn.outvars[0]] = copy_index_sets(in_indices)


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
    deps[eqn.outvars[0]] = copy_index_sets(index_sets(deps, eqn.invars[0]))


def prop_convert_element_type(
    eqn: JaxprEqn,
    deps: Deps,
    const_vals: ConstVals,
    value_bounds: ValueBounds | None = None,
) -> None:
    """Type conversion (e.g., float32 → float64) changes dtype without changing values.

    Dependencies pass through unchanged.
    The Jacobian is the identity matrix.

    Also propagates const values and value bounds with the new dtype
    so downstream gather/scatter can resolve static indices.
    JAX inserts ``convert_element_type`` for index dtype changes
    (e.g. int64 → int32) before gather/scatter;
    without const propagation here the chain breaks
    and gathers fall back to conservative.

    Example: y = x.astype(float64) where x = [a, b, c]
        Input deps:  [{0}, {1}, {2}]
        Output deps: [{0}, {1}, {2}]

    Jaxpr:
        invars[0]: input array
        new_dtype: target dtype

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.convert_element_type.html
    """
    deps[eqn.outvars[0]] = copy_index_sets(index_sets(deps, eqn.invars[0]))

    in_val = atom_const_val(eqn.invars[0], const_vals)
    if in_val is not None:
        new_dtype = eqn.params.get("new_dtype")
        if new_dtype is not None:
            const_vals[eqn.outvars[0]] = in_val.astype(new_dtype)
        else:
            # stop_gradient, bitcast_convert_type, etc. — pass through as-is.
            const_vals[eqn.outvars[0]] = in_val

    # Propagate value bounds with dtype cast.
    if value_bounds is not None:
        bounds = atom_value_bounds(eqn.invars[0], const_vals, value_bounds)
        if bounds is not None:
            lo, hi = bounds
            new_dtype = eqn.params.get("new_dtype")
            if new_dtype is not None:
                value_bounds[eqn.outvars[0]] = (
                    lo.astype(new_dtype),
                    hi.astype(new_dtype),
                )
            else:
                value_bounds[eqn.outvars[0]] = (lo, hi)

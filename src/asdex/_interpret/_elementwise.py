"""Propagation rules for element-wise operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import ConstVals, Deps, atom_const_val, atom_numel, index_sets


def propagate_const_binary(
    eqn: JaxprEqn, const_vals: ConstVals, ufuncs: dict[str, np.ufunc]
) -> None:
    """Propagate constant values through a binary op.

    If both inputs are statically known and a matching ufunc exists,
    the output value is computed and stored.
    Used for tracking static indices through arithmetic to gather/scatter.

    Example: z = x + y where x = [1, 2], y = [3, 4]
        const_vals before: {x: [1, 2], y: [3, 4]}
        const_vals after:  {x: [1, 2], y: [3, 4], z: [4, 6]}
    """
    in1 = atom_const_val(eqn.invars[0], const_vals)
    in2 = atom_const_val(eqn.invars[1], const_vals)
    if in1 is not None and in2 is not None:
        ufunc = ufuncs.get(eqn.primitive.name)
        if ufunc is not None:
            const_vals[eqn.outvars[0]] = ufunc(in1, in2)


def prop_zero_derivative(eqn: JaxprEqn, deps: Deps) -> None:
    """Propagate dependencies through zero-derivative primitives.

    Operations like floor, ceil, round, sign, and is_finite have zero derivative
    almost everywhere. Their outputs are piecewise constant, so infinitesimal
    input changes don't affect outputs.

    Mathematically, for f in {floor, ceil, sign, ...}:
        ∂f/∂x = 0  (almost everywhere)
    Therefore, output elements have no dependencies on input elements.

    Example: y = floor(x) where x = [1.7, 2.3, 3.9]
        Input deps:  [{0}, {1}, {2}]
        Output deps: [{}, {}, {}]  (empty sets, no dependence)
    """
    for outvar in eqn.outvars:
        deps[outvar] = [set() for _ in range(atom_numel(outvar))]


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
    # Broadcasting via modular indexing.
    # JAX element-wise ops only broadcast scalars (len 1) against arrays.
    # i % len gives 0 for scalars (i % 1 == 0 for all i),
    # and i for same-sized arrays,
    # so it naturally selects the right element without branching.
    # Size-0 arrays propagate to size-0 output (no elements to combine).
    out_size = 0 if len(in1) == 0 or len(in2) == 0 else max(len(in1), len(in2))
    deps[eqn.outvars[0]] = [
        in1[i % len(in1)] | in2[i % len(in2)] for i in range(out_size)
    ]


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

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.convert_element_type.html
    """
    deps[eqn.outvars[0]] = [s.copy() for s in index_sets(deps, eqn.invars[0])]

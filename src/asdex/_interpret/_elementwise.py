"""Propagation rules for element-wise operations."""

import numpy as np
from jax._src.core import JaxprEqn, Literal, Var

from ._commons import ConstVals, Deps, IndexSets, atom_numel, index_sets, union_all

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


def prop_comparison(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Propagate dependencies through comparison operators (lt, le, gt, ge, eq, ne).

    Comparisons have zero derivative almost everywhere (output is boolean).
    Also tracks const values: if both inputs are consts, the output bool array
    is stored for use in the select_n → gather/scatter chain.

    Example: y = (x < 3) where x = [1, 4, 2]
        Input deps:  [{0}, {1}, {2}]
        Output deps: [{}, {}, {}]  (empty sets, no dependence)
    """
    for outvar in eqn.outvars:
        deps[outvar] = [set() for _ in range(atom_numel(outvar))]

    # Track const values for static index tracking
    def get_val(atom: Var | Literal) -> np.ndarray | None:
        if isinstance(atom, Literal):
            return np.asarray(atom.val)
        if isinstance(atom, Var) and atom in const_vals:
            return const_vals[atom]
        return None

    in1_val = get_val(eqn.invars[0])
    in2_val = get_val(eqn.invars[1])

    if in1_val is not None and in2_val is not None:
        ufunc = _COMPARISON_UFUNCS.get(eqn.primitive.name)
        if ufunc is not None:
            const_vals[eqn.outvars[0]] = ufunc(in1_val, in2_val)


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


def prop_binary_elementwise(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Binary element-wise ops (add, mul, etc.) combine two arrays element-wise.
    Each output element depends on the corresponding elements from both inputs.
    Broadcasting is handled: scalars contribute to all output elements.

    Also tracks const values: if both inputs are tracked consts, the output
    value is computed and stored for use in gather/scatter handlers.

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
    in1_atom = eqn.invars[0]
    in2_atom = eqn.invars[1]
    in1 = index_sets(deps, in1_atom)
    in2 = index_sets(deps, in2_atom)
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

    # Track const values for static index tracking
    def get_val(atom):
        if isinstance(atom, Literal):
            return np.asarray(atom.val)
        if isinstance(atom, Var) and atom in const_vals:
            return const_vals[atom]
        return None

    in1_val = get_val(in1_atom)
    in2_val = get_val(in2_atom)

    if in1_val is not None and in2_val is not None:
        ufunc = _ARITHMETIC_UFUNCS.get(eqn.primitive.name)
        if ufunc is not None:
            const_vals[eqn.outvars[0]] = ufunc(in1_val, in2_val)


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
    """
    deps[eqn.outvars[0]] = [s.copy() for s in index_sets(deps, eqn.invars[0])]

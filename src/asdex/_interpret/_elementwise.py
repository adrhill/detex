"""Propagation rules for element-wise operations."""

from typing import Any

import numpy as np
from jax._src.core import JaxprEqn, Literal, Var

from ._commons import ConstVals, Deps, IndexSets, atom_numel, index_sets, union_all


def _binary_op(name: str, a: Any, b: Any) -> np.ndarray:
    """Apply a binary operation to two arrays, returning an ndarray."""
    if name == "add" or name == "add_any":
        return np.asarray(a + b)
    elif name == "sub":
        return np.asarray(a - b)
    elif name == "mul":
        return np.asarray(a * b)
    elif name == "div":
        return np.asarray(a / b)
    elif name == "pow":
        return np.asarray(a**b)
    elif name == "max":
        return np.maximum(a, b)
    elif name == "min":
        return np.minimum(a, b)
    else:
        msg = f"Unknown binary op: {name}"
        raise ValueError(msg)


def _comparison_op(name: str, a: Any, b: Any) -> np.ndarray:
    """Apply a comparison operation to two arrays, returning an ndarray."""
    if name == "lt":
        return np.asarray(a < b)
    elif name == "le":
        return np.asarray(a <= b)
    elif name == "gt":
        return np.asarray(a > b)
    elif name == "ge":
        return np.asarray(a >= b)
    elif name == "eq":
        return np.asarray(a == b)
    elif name == "ne":
        return np.asarray(a != b)
    else:
        msg = f"Unknown comparison op: {name}"
        raise ValueError(msg)


def prop_zero_derivative(eqn: JaxprEqn, deps: Deps, const_vals: ConstVals) -> None:
    """Propagate dependencies through zero-derivative primitives.

    Operations like floor, ceil, comparisons (eq, lt, gt, ...), and sign
    have zero derivative almost everywhere.
    Their outputs are piecewise constant,
    so infinitesimal input changes don't affect outputs.

    Also tracks const values for comparison ops: if both inputs are consts,
    the output bool value is tracked for use in select_n → gather/scatter.

    Mathematically, for f in {floor, ceil, sign, eq, ...}:
        ∂f/∂x = 0  (almost everywhere)
    Therefore, output elements have no dependencies on input elements.

    Example: y = floor(x) where x = [1.7, 2.3, 3.9]
        Input deps:  [{0}, {1}, {2}]
        Output deps: [{}, {}, {}]  (empty sets, no dependence)
    """
    for outvar in eqn.outvars:
        deps[outvar] = [set() for _ in range(atom_numel(outvar))]

    # Track const values for comparison ops (used in select_n chain)
    prim_name = eqn.primitive.name
    if prim_name in ("lt", "le", "gt", "ge", "eq", "ne") and len(eqn.invars) >= 2:

        def get_val(atom: Any) -> Any:
            if isinstance(atom, Literal):
                return np.asarray(atom.val)
            if isinstance(atom, Var) and atom in const_vals:
                return const_vals[atom]
            return None

        in1_val = get_val(eqn.invars[0])
        in2_val = get_val(eqn.invars[1])

        if in1_val is not None and in2_val is not None:
            try:
                out_var = eqn.outvars[0]
                const_vals[out_var] = _comparison_op(prim_name, in1_val, in2_val)
            except ValueError:
                pass


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
        prim_name = eqn.primitive.name
        try:
            out_var = eqn.outvars[0]
            const_vals[out_var] = _binary_op(prim_name, in1_val, in2_val)
        except ValueError:
            pass  # Unknown op, don't track


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

"""
Global Jacobian sparsity detection via jaxpr graph analysis.

This is TRUE global sparsity detection - it analyzes the computation graph
structure without evaluating derivatives, so results are valid for ALL inputs.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.core import Literal, Var
from scipy.sparse import coo_matrix

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

    # env maps variable -> list of index sets (one per element)
    # This is the key data structure for element-wise tracking
    env: dict[Var, list[set[int]]] = {}

    def get_var_size(v) -> int:
        """Get the total number of elements in a variable."""
        if isinstance(v, Literal):
            val = v.val
            if hasattr(val, "shape"):
                return int(np.prod(val.shape)) if val.shape else 1
            return 1
        aval = v.aval
        if hasattr(aval, "shape"):
            shape = aval.shape
            return int(np.prod(shape)) if shape else 1
        return 1

    def read(v) -> list[set[int]]:
        """Get list of index sets for variable v (one per element)."""
        if isinstance(v, Literal):
            size = get_var_size(v)
            return [set() for _ in range(size)]
        return env.get(v, [set()])

    def write(v, indices: list[set[int]]):
        """Set index sets for variable v."""
        env[v] = indices

    # Initialize: input element i depends on input index i
    input_var = jaxpr.invars[0]
    write(input_var, [{i} for i in range(n)])

    # Process each equation in the jaxpr
    for eqn in jaxpr.eqns:
        prim = eqn.primitive.name
        invars = eqn.invars
        outvars = eqn.outvars

        if prim in ZERO_DERIVATIVE_PRIMITIVES:
            # Zero-derivative ops: output has no dependence on inputs
            for outvar in outvars:
                size = get_var_size(outvar)
                write(outvar, [set() for _ in range(size)])

        elif prim == "slice":
            # Slice extracts elements [start:limit] - preserve element structure
            in_indices = read(invars[0])
            start = eqn.params["start_indices"]
            limit = eqn.params["limit_indices"]
            if len(start) == 1:
                # 1D slice: extract the specific range
                out_indices = in_indices[start[0] : limit[0]]
            else:
                # Multi-dimensional: conservative fallback
                all_deps = set().union(*in_indices)
                out_size = get_var_size(outvars[0])
                out_indices = [all_deps.copy() for _ in range(out_size)]
            write(outvars[0], out_indices)

        elif prim == "squeeze":
            # Squeeze removes size-1 dims, preserves element dependencies
            write(outvars[0], read(invars[0]))

        elif prim == "broadcast_in_dim":
            # Broadcast: replicate dependencies to match output shape
            in_indices = read(invars[0])
            out_shape = eqn.params["shape"]
            out_size = int(np.prod(out_shape))
            if len(in_indices) == 1:
                # Scalar broadcast: all outputs get same deps
                write(outvars[0], [in_indices[0].copy() for _ in range(out_size)])
            else:
                # Array broadcast: conservative (could be smarter)
                all_deps = set().union(*in_indices)
                write(outvars[0], [all_deps.copy() for _ in range(out_size)])

        elif prim == "concatenate":
            # Concatenate: join element lists in order
            out_indices = []
            for invar in invars:
                out_indices.extend(read(invar))
            write(outvars[0], out_indices)

        elif prim == "reshape":
            # Reshape preserves total elements and their dependencies
            in_indices = read(invars[0])
            out_size = get_var_size(outvars[0])
            if len(in_indices) == out_size:
                write(outvars[0], in_indices)
            else:
                # Size mismatch: conservative
                all_deps = set().union(*in_indices)
                write(outvars[0], [all_deps.copy() for _ in range(out_size)])

        elif prim == "integer_pow":
            # x^n: element-wise, preserves structure (unless n=0)
            power = eqn.params.get("y", 1)
            in_indices = read(invars[0])
            if power == 0:
                write(outvars[0], [set() for _ in range(len(in_indices))])
            else:
                write(outvars[0], [s.copy() for s in in_indices])

        elif prim in ("add", "sub", "mul", "div", "pow", "max", "min"):
            # Binary element-wise: merge corresponding elements
            in1 = read(invars[0])
            in2 = read(invars[1])
            out_size = max(len(in1), len(in2))
            out_indices = []
            for i in range(out_size):
                deps = set()
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
            write(outvars[0], out_indices)

        elif prim in (
            "neg",
            "exp",
            "log",
            "sin",
            "cos",
            "tan",
            "sqrt",
            "abs",
            "sinh",
            "cosh",
            "tanh",
            "log1p",
            "expm1",
        ):
            # Unary element-wise: preserve element structure
            in_indices = read(invars[0])
            write(outvars[0], [s.copy() for s in in_indices])

        elif prim == "reduce_sum":
            # Reduction: output depends on all input elements
            in_indices = read(invars[0])
            all_deps = set().union(*in_indices)
            write(outvars[0], [all_deps])

        elif prim == "convert_element_type":
            # Type conversion: preserve dependencies
            in_indices = read(invars[0])
            write(outvars[0], [s.copy() for s in in_indices])

        else:
            # Default fallback: union all input deps for all outputs
            all_deps = set()
            for invar in invars:
                for s in read(invar):
                    all_deps |= s
            for outvar in outvars:
                out_size = get_var_size(outvar)
                write(outvar, [all_deps.copy() for _ in range(out_size)])

    # Extract output dependencies and build sparse matrix
    output_var = jaxpr.outvars[0]
    out_indices = read(output_var)

    rows = []
    cols = []
    for i, deps in enumerate(out_indices):
        for j in deps:
            rows.append(i)
            cols.append(j)

    return coo_matrix(([True] * len(rows), (rows, cols)), shape=(m, n), dtype=bool)


# ============================================================
# Tests
# ============================================================

if __name__ == "__main__":

    def test_sparsity(name, f, n, expected):
        """Test that detected sparsity matches expected pattern."""
        result = jacobian_sparsity(f, n).toarray().astype(int)
        expected = np.array(expected)
        passed = np.array_equal(result, expected)

        print("=" * 50)
        print(f"{name}")
        print("=" * 50)
        print(f"Detected:\n{result}")
        print(f"Expected:\n{expected}")
        print(f"PASSED: {passed}\n")
        return passed

    all_passed = True

    # Test 1: Simple dependencies
    def f1(x):
        return jnp.array(
            [
                x[0] + x[1],  # depends on x[0], x[1]
                x[1] * x[2],  # depends on x[1], x[2]
                x[2],  # depends on x[2]
            ]
        )

    all_passed &= test_sparsity(
        "Test 1: f(x) = [x0+x1, x1*x2, x2]",
        f1,
        n=3,
        expected=[[1, 1, 0], [0, 1, 1], [0, 0, 1]],
    )

    # Test 2: More complex
    def f2(x):
        a = x[0] * x[1]
        b = jnp.sin(x[2])
        c = a + b
        return jnp.array([c, x[3], a * x[3]])

    all_passed &= test_sparsity(
        "Test 2: f(x) = [x0*x1 + sin(x2), x3, x0*x1*x3]",
        f2,
        n=4,
        expected=[[1, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 1]],
    )

    # Test 3: Diagonal Jacobian (element-wise)
    def f3(x):
        return x**2

    all_passed &= test_sparsity(
        "Test 3: f(x) = x^2 (element-wise)",
        f3,
        n=4,
        expected=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    )

    # Test 4: Dense Jacobian
    def f4(x):
        return jnp.array([jnp.sum(x), jnp.prod(x)])

    all_passed &= test_sparsity(
        "Test 4: f(x) = [sum(x), prod(x)]", f4, n=3, expected=[[1, 1, 1], [1, 1, 1]]
    )

    # Test 5: SCT README example
    def f5(x):
        return jnp.array([x[0] ** 2, 2 * x[0] * x[1] ** 2, jnp.sin(x[2])])

    all_passed &= test_sparsity(
        "Test 5: SCT README - f(x) = [x1^2, 2*x1*x2^2, sin(x3)]",
        f5,
        n=3,
        expected=[[1, 0, 0], [1, 1, 0], [0, 0, 1]],
    )

    print("=" * 50)
    print(f"ALL TESTS PASSED: {all_passed}")
    print("=" * 50)

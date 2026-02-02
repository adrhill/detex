# Jacobian Sparsity Detection in JAX

This note explains how to detect Jacobian sparsity patterns without computing actual derivatives, and provides a working implementation in JAX.

## 1. Problem Statement

### What is Jacobian Sparsity?

For a function $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is an $m \times n$ matrix where entry $(i, j)$ is $\partial f_i / \partial x_j$. A Jacobian is **sparse** when most entries are structurally zero—meaning they're zero for *all* input values, not just some.

```
Example: f([x₁, x₂, x₃]) = [x₁ + x₂, x₂ * x₃, x₃]

Jacobian:
┌                 ┐
│ 1   1   0       │   <- f₁ depends on x₁, x₂ only
│ 0   x₃  x₂      │   <- f₂ depends on x₂, x₃ only
│ 0   0   1       │   <- f₃ depends on x₃ only
└                 ┘

Sparsity pattern (which entries are non-zero):
┌           ┐
│ 1   1   0 │
│ 0   1   1 │
│ 0   0   1 │
└           ┘
```

### Why Sparsity Detection Matters

**1. Sparse Automatic Differentiation**

Computing a full $m \times n$ Jacobian with forward-mode AD requires $n$ passes; with reverse-mode, $m$ passes. But if the Jacobian has structure, we can do better:
- Column $j$ only depends on outputs that use input $j$
- Row $i$ only depends on inputs that output $i$ uses
- With coloring algorithms, structurally orthogonal columns/rows can share a single AD pass

**2. Newton Methods and Optimization**

Sparse linear algebra is dramatically faster than dense. Knowing the sparsity pattern enables:
- Efficient factorizations (sparse LU, Cholesky)
- Better preconditioners
- Reduced memory footprint

**3. The Cost Problem**

Computing a full Jacobian just to find its sparsity pattern is wasteful:
- For $f: \mathbb{R}^{1000} \to \mathbb{R}^{1000}$, computing the Jacobian costs 1000 AD passes
- The sparsity pattern can be detected in a single forward pass

## 2. The Sparsity Detection Approach

### Key Insight: Track Dependencies, Not Values

Instead of computing $\partial f_i / \partial x_j$, we ask: "does $f_i$ depend on $x_j$ at all?"

This is purely structural—we propagate **index sets** through the computation:
- Each input $x_j$ starts with the index set $\{j\}$
- Operations combine index sets: if $z = x + y$, then $z$'s index set is the union of $x$'s and $y$'s
- The output's index set tells us which inputs affect it

### Global vs Local Sparsity

**Global sparsity** considers all possible execution paths. For:
```python
z = x if condition else y
```
Global analysis says $z$ depends on *both* $x$ and $y$—conservative but safe.

**Local sparsity** depends on actual input values. If `condition` is true, $z$ depends only on $x$. This gives tighter patterns but requires evaluating the function with real inputs.

This note focuses on **global sparsity**.

### Comparison to Full Jacobian

| Approach | Cost | What you get |
|----------|------|--------------|
| Forward-mode AD | $O(n)$ AD passes | Full Jacobian values |
| Reverse-mode AD | $O(m)$ AD passes | Full Jacobian values |
| Sparsity detection | $O(1)$ forward pass | Boolean sparsity pattern |

The sparsity detection pass is typically much cheaper because it doesn't compute derivatives—just propagates sets.

## 3. Implementation in JAX

### The Strategy

JAX provides `make_jaxpr`, which captures a function's computation as a graph of primitive operations. We can:
1. Get the computation graph (jaxpr)
2. Propagate index sets through the graph
3. Read off the sparsity pattern from output index sets

### Understanding Jaxpr

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.array([x[0] + x[1], x[1] * x[2]])

jaxpr = jax.make_jaxpr(f)(jnp.zeros(3))
print(jaxpr)
```

Output:
```
{ lambda ; a:f32[3]. let
    b:f32[] = slice[limit_indices=(1,) start_indices=(0,) strides=None] a
    c:f32[] = squeeze[dimensions=(0,)] b
    d:f32[] = slice[limit_indices=(2,) start_indices=(1,) strides=None] a
    e:f32[] = squeeze[dimensions=(0,)] d
    f:f32[] = add c e
    g:f32[] = slice[limit_indices=(3,) start_indices=(2,) strides=None] a
    h:f32[] = squeeze[dimensions=(0,)] g
    i:f32[] = mul e h
    j:f32[2] = broadcast_in_dim[...] f
    k:f32[2] = broadcast_in_dim[...] i
    l:f32[2] = concatenate[dimension=0] j k
  in (l,) }
```

Each line is an equation: `output = primitive[params] inputs`. We trace through these, propagating index sets.

### Operator Classification

For sparsity detection, we classify operators by whether they have non-zero derivatives:

| Category | Description | Examples |
|----------|-------------|----------|
| **Zero-derivative** | Output doesn't depend on input for derivative purposes | `floor`, `ceil`, comparisons |
| **Linear** | Output depends on inputs | `add`, `sub`, `reshape`, `slice` |
| **Nonlinear** | Output depends on inputs (non-zero 2nd derivative, relevant for Hessians) | `mul`, `sin`, `exp` |

For Jacobian sparsity, both linear and nonlinear ops propagate dependencies the same way: union of input index sets.

## 4. Complete Code Example

```python
"""
Global Jacobian sparsity detection via jaxpr graph analysis.

This is TRUE global sparsity detection - it analyzes the computation graph
structure without evaluating derivatives, so results are valid for ALL inputs.
"""

import jax
import jax.numpy as jnp
from jax._src.core import Var, Literal
from scipy.sparse import coo_matrix
from typing import Dict, Set, List
import numpy as np


# Primitives with zero derivatives (output doesn't depend on input)
ZERO_DERIVATIVE_PRIMITIVES = frozenset([
    'floor', 'ceil', 'round', 'sign',
    'eq', 'ne', 'lt', 'le', 'gt', 'ge',
    'is_finite',
])


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
    env: Dict[Var, List[Set[int]]] = {}

    def get_var_size(v) -> int:
        """Get the total number of elements in a variable."""
        if isinstance(v, Literal):
            val = v.val
            if hasattr(val, 'shape'):
                return int(np.prod(val.shape)) if val.shape else 1
            return 1
        aval = v.aval
        if hasattr(aval, 'shape'):
            shape = aval.shape
            return int(np.prod(shape)) if shape else 1
        return 1

    def read(v) -> List[Set[int]]:
        """Get list of index sets for variable v (one per element)."""
        if isinstance(v, Literal):
            size = get_var_size(v)
            return [set() for _ in range(size)]
        return env.get(v, [set()])

    def write(v, indices: List[Set[int]]):
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

        elif prim == 'slice':
            # Slice extracts elements [start:limit] - preserve element structure
            in_indices = read(invars[0])
            start = eqn.params['start_indices']
            limit = eqn.params['limit_indices']
            if len(start) == 1:
                # 1D slice: extract the specific range
                out_indices = in_indices[start[0]:limit[0]]
            else:
                # Multi-dimensional: conservative fallback
                all_deps = set().union(*in_indices)
                out_size = get_var_size(outvars[0])
                out_indices = [all_deps.copy() for _ in range(out_size)]
            write(outvars[0], out_indices)

        elif prim == 'squeeze':
            # Squeeze removes size-1 dims, preserves element dependencies
            write(outvars[0], read(invars[0]))

        elif prim == 'broadcast_in_dim':
            # Broadcast: replicate dependencies to match output shape
            in_indices = read(invars[0])
            out_shape = eqn.params['shape']
            out_size = int(np.prod(out_shape))
            if len(in_indices) == 1:
                # Scalar broadcast: all outputs get same deps
                write(outvars[0], [in_indices[0].copy() for _ in range(out_size)])
            else:
                # Array broadcast: conservative (could be smarter)
                all_deps = set().union(*in_indices)
                write(outvars[0], [all_deps.copy() for _ in range(out_size)])

        elif prim == 'concatenate':
            # Concatenate: join element lists in order
            out_indices = []
            for invar in invars:
                out_indices.extend(read(invar))
            write(outvars[0], out_indices)

        elif prim == 'reshape':
            # Reshape preserves total elements and their dependencies
            in_indices = read(invars[0])
            out_size = get_var_size(outvars[0])
            if len(in_indices) == out_size:
                write(outvars[0], in_indices)
            else:
                # Size mismatch: conservative
                all_deps = set().union(*in_indices)
                write(outvars[0], [all_deps.copy() for _ in range(out_size)])

        elif prim == 'integer_pow':
            # x^n: element-wise, preserves structure (unless n=0)
            power = eqn.params.get('y', 1)
            in_indices = read(invars[0])
            if power == 0:
                write(outvars[0], [set() for _ in range(len(in_indices))])
            else:
                write(outvars[0], [s.copy() for s in in_indices])

        elif prim in ('add', 'sub', 'mul', 'div', 'pow', 'max', 'min'):
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

        elif prim in ('neg', 'exp', 'log', 'sin', 'cos', 'tan', 'sqrt', 'abs',
                      'sinh', 'cosh', 'tanh', 'log1p', 'expm1'):
            # Unary element-wise: preserve element structure
            in_indices = read(invars[0])
            write(outvars[0], [s.copy() for s in in_indices])

        elif prim == 'reduce_sum':
            # Reduction: output depends on all input elements
            in_indices = read(invars[0])
            all_deps = set().union(*in_indices)
            write(outvars[0], [all_deps])

        elif prim == 'convert_element_type':
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

    return coo_matrix(
        ([True] * len(rows), (rows, cols)),
        shape=(m, n),
        dtype=bool
    )


# ============================================================
# Example usage and tests
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
        return jnp.array([
            x[0] + x[1],      # depends on x[0], x[1]
            x[1] * x[2],      # depends on x[1], x[2]
            x[2]              # depends on x[2]
        ])

    all_passed &= test_sparsity(
        "Test 1: f(x) = [x₀+x₁, x₁*x₂, x₂]",
        f1, n=3,
        expected=[[1, 1, 0],
                  [0, 1, 1],
                  [0, 0, 1]]
    )

    # Test 2: More complex
    def f2(x):
        a = x[0] * x[1]
        b = jnp.sin(x[2])
        c = a + b
        return jnp.array([c, x[3], a * x[3]])

    all_passed &= test_sparsity(
        "Test 2: f(x) = [x₀*x₁ + sin(x₂), x₃, x₀*x₁*x₃]",
        f2, n=4,
        expected=[[1, 1, 1, 0],
                  [0, 0, 0, 1],
                  [1, 1, 0, 1]]
    )

    # Test 3: Diagonal Jacobian (element-wise)
    def f3(x):
        return x ** 2

    all_passed &= test_sparsity(
        "Test 3: f(x) = x² (element-wise)",
        f3, n=4,
        expected=[[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
    )

    # Test 4: Dense Jacobian
    def f4(x):
        return jnp.array([jnp.sum(x), jnp.prod(x)])

    all_passed &= test_sparsity(
        "Test 4: f(x) = [sum(x), prod(x)]",
        f4, n=3,
        expected=[[1, 1, 1],
                  [1, 1, 1]]
    )

    # Test 5: SCT README example
    def f5(x):
        return jnp.array([
            x[0]**2,
            2 * x[0] * x[1]**2,
            jnp.sin(x[2])
        ])

    all_passed &= test_sparsity(
        "Test 5: SCT README - f(x) = [x₁², 2*x₁*x₂², sin(x₃)]",
        f5, n=3,
        expected=[[1, 0, 0],
                  [1, 1, 0],
                  [0, 0, 1]]
    )

    print("=" * 50)
    print(f"ALL TESTS PASSED: {all_passed}")
    print("=" * 50)
```

## 5. Limitations and Extensions

### Limitations of This Approach

**1. Conservative for control flow**

Global sparsity analysis unions all branches, which can over-report dependencies:
```python
def f(x):
    if x[0] > 0:
        return x[1]
    else:
        return x[2]
```
Reports dependence on both `x[1]` and `x[2]`, though only one is used for any given input.

**2. Array indexing challenges**

When indices are data-dependent, sparsity is harder to determine:
```python
def f(x, i):
    return x[i]  # Which input does this depend on?
```

**3. Element-wise tracking complexity**

To get per-element sparsity (not just per-variable), we need to track which elements of intermediate arrays depend on which inputs. The implementation above does this conservatively—it could be made more precise for specific primitives like `slice` and `gather`.

### More Sophisticated Approaches

**1. True Operator Overloading (like SparseConnectivityTracer.jl)**

Define custom tracer types that propagate index sets at the Python level:
- Replace floats with `Tracer(indices: Set[int])`
- Overload all operators (`__add__`, `__mul__`, etc.) to merge index sets
- Works with any Python code, not just JAX-traceable functions
- Single forward pass, no graph extraction needed

**2. More Precise Element Tracking**

The implementation above is conservative for some primitives. More precise tracking could:
- Handle multi-dimensional slicing exactly
- Track broadcast patterns precisely
- Handle gather/scatter with known indices

**3. Symbolic Differentiation**

Use symbolic math (SymPy) to compute symbolic Jacobian entries, check which are structurally zero. More expensive but handles arbitrary expressions.

### Further Reading

- **SparseConnectivityTracer.jl** - Julia package implementing operator-overloading sparsity detection
- **SparseDiffTools.jl** - Sparse Jacobian computation using detected sparsity + coloring
- **ColPack** - Graph coloring algorithms for efficient sparse Jacobian computation
- **"What Color Is Your Jacobian?"** - Paper on graph coloring for sparse AD

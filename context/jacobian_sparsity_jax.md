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


## 4. Limitations and Extensions

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

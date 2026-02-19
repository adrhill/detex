# Sparsity Detection

Sparsity detection determines which entries of a Jacobian can be nonzero.
Given \(f: \mathbb{R}^n \to \mathbb{R}^m\),
it computes a binary [global sparsity pattern](global-sparsity.md) —
a conservative superset of the true nonzero structure of the Jacobian \(J_{ij} = \partial f_i / \partial x_j\).
This pattern is the input to [graph coloring](coloring.md),
which exploits the sparsity to reduce the number of AD passes.

## Why Abstract Interpretation

Since we need [global sparsity patterns](global-sparsity.md) —
patterns that are valid for all inputs, not just a specific one —
numerical approaches like finite-difference probing or computing the Jacobian via AD
are not suitable.
They evaluate the function at a concrete input
and can only reveal the _local_ pattern at that point,
missing nonzeros that happen to be zero at the probe.

`asdex` uses **abstract interpretation** instead:
it analyzes the structure of the computation
without evaluating it on real numbers.
This gives global patterns directly —
no numerical evaluation, no dependence on a particular input,
and no risk of missing nonzeros due to cancellation.

## Jaxpr: JAX's Intermediate Representation

JAX represents computations as **jaxprs** (JAX expressions) —
a flat sequence of primitive operations with explicit data flow.
When you call `jax.make_jaxpr(f)(x)`,
JAX traces `f` symbolically (without evaluating it)
and returns a jaxpr that records every operation.

A jaxpr consists of:

- **Input variables** — the function's arguments.
- **Equations** — one per primitive operation (e.g. `sin`, `add`, `gather`),
  each with input atoms, output variables, and parameters.
- **Output variables** — the function's return values.

For example, `f(x) = sin(x) + 1` produces the following jaxpr:

```jaxpr
{ lambda ; a:f32[3]. let
    b:f32[3] = sin a
    c:f32[3] = add b 1.0
  in (c,) }
```

`asdex` hooks into this representation:
it calls `jax.make_jaxpr` to obtain the computation graph,
then walks the equations one by one,
propagating index sets instead of numerical values.

## Index Set Propagation

The core idea is to track, for each element of each intermediate array,
_which input elements it depends on_.
This dependency information is stored as index sets —
a list of `set[int]`, one set per array element.

The algorithm proceeds in three steps:

**1. Initialization.**
Each input element \(x_j\) starts with the singleton set \(\{j\}\),
meaning it depends only on itself.
For a 3-element input, the initial index sets are `[{0}, {1}, {2}]`.

**2. Propagation.**
Walk through each equation in the jaxpr.
For each primitive, a **handler** maps input index sets to output index sets
according to how that operation routes dependencies.

**3. Extraction.**
After processing all equations,
read the index sets of the output variables.
Output \(i\) depends on input \(j\) iff \(j \in S_i\),
which directly gives the sparsity pattern.

## Example

Consider \(f(x) = \bigl[\sin(x_1 \cdot x_2),\; x_2 + x_3\bigr]\)
with input \(x \in \mathbb{R}^3\).

**Initialization:**

| Variable | Index sets |
|:---------|:-----------|
| \(x\) | `[{0}, {1}, {2}]` |

**Equation 1:** `mul x[0] x[1] → t1` — elementwise multiply.
Each output depends on both operands:

| Variable | Index sets |
|:---------|:-----------|
| \(t_1\) | `[{0, 1}]` |

**Equation 2:** `sin t1 → t2` — elementwise unary op.
Dependencies pass through unchanged:

| Variable | Index sets |
|:---------|:-----------|
| \(t_2\) | `[{0, 1}]` |

**Equation 3:** `add x[1] x[2] → t3` — elementwise add.
Union of both operands:

| Variable | Index sets |
|:---------|:-----------|
| \(t_3\) | `[{1, 2}]` |

**Extraction:**
The output is \([t_2, t_3]\), so the index sets are `[{0, 1}, {1, 2}]`.
This encodes the sparsity pattern:

\[
J = \begin{pmatrix} \times & \times & \\ & \times & \times \end{pmatrix}
\]

The true Jacobian is \(\bigl[\begin{smallmatrix} x_2 \cos(x_1 x_2) & x_1 \cos(x_1 x_2) & 0 \\ 0 & 1 & 1 \end{smallmatrix}\bigr]\),
which confirms the detected pattern.

## Primitive Handlers

Each JAX primitive has a handler that defines how it propagates index sets.
The handlers fall into a few families.

### Elementwise Operations

Unary operations like `sin`, `exp`, `neg`
pass each element's index set through unchanged —
\(y_i = g(x_i)\) means \(S_{y_i} = S_{x_i}\).

Binary operations like `add`, `mul`, `sub`
take the union of their operands' index sets —
\(y_i = x_i \oplus z_i\) means \(S_{y_i} = S_{x_i} \cup S_{z_i}\).
When shapes differ, broadcasting rules determine which elements pair up.

### Reductions

A reduction like `sum` over an axis unions all index sets along that axis.
If \(y_i = \sum_k x_{ik}\), then \(S_{y_i} = \bigcup_k S_{x_{ik}}\).
A full reduction (no remaining axes) unions everything into a single set.

### Permutations and Reshapes

Operations like `transpose`, `reshape`, `slice`, `reverse`, and `broadcast`
rearrange elements without combining them.
Each output element maps to exactly one input element,
so the handler copies the corresponding index set.
`asdex` implements this via a position map:
apply the operation to an array where each element holds its own flat index,
then read off the mapping.

### Indexing (Gather and Scatter)

`gather` and `scatter` are the most complex primitives.
When the indices are **statically known** (constants in the jaxpr),
the handler resolves exactly which input position each output reads from
and copies index sets accordingly — just like a permutation.

When the indices are **dynamic** (computed from inputs),
the handler cannot know which elements will be accessed at runtime.
It falls back to the [conservative strategy](#fallback-handlers) described below.

This is why `asdex` tracks constant values through the computation graph:
if an index array is built from literals and arithmetic on constants,
the handler can still resolve it precisely
even though it is not a direct literal in the jaxpr.

## Fallback Handlers

Not every JAX primitive has a precise handler.
When `asdex` encounters an unhandled primitive,
it uses a conservative fallback:
every output element is assumed to depend on every input element.
This is always correct — it is a superset of the true pattern —
but it may be much less sparse than necessary.

A small number of primitives use this fallback intentionally
(e.g. callbacks into opaque Python code where dependencies cannot be analyzed).
For all other cases, `asdex` raises an error on unknown primitives
rather than silently falling back,
since silent over-approximation could mask bugs in the handler coverage.

!!! tip

    More precise handlers can be added for fallback primitives
    to reduce conservatism and produce sparser patterns.
    Please open an issue if you encounter overly conservative patterns.

## Sources of Conservatism

Even with precise handlers,
three mechanisms make global patterns conservative relative to local ones:

1. **Branching** (`cond`, `select_n`):
   the detector takes the **union** over all branches,
   since it cannot know which branch will execute at runtime.
   This is the primary difference from local detection.
2. **Multiplication**:
   \(f(x) = x_1 \cdot x_2\) always reports both dependencies globally,
   even though one factor might be zero at a particular input.
3. **Dynamic indexing**:
   when gather/scatter indices depend on the input,
   the handler must assume any element could be accessed.

## Hessian Detection

Hessian sparsity is detected by applying Jacobian detection to the gradient:

\[
\operatorname{hessian\_sparsity}(f) = \operatorname{jacobian\_sparsity}(\nabla f)
\]

This composes naturally with JAX's autodiff:
`jax.grad` produces a jaxpr,
which `asdex` analyzes the same way.

# Sparsity Detection

Sparsity detection determines which entries of a Jacobian can be nonzero.
Given \(f: \mathbb{R}^n \to \mathbb{R}^m\),
it computes a binary sparsity pattern —
a superset of the true nonzero structure of the Jacobian \(J_{ij} = \partial f_i / \partial x_j\).
This pattern is the input to [graph coloring](coloring.md),
which exploits the sparsity to reduce the number of AD passes.

## Global vs. Local Sparsity Patterns

Consider \(f(x) = x_1 \cdot x_2\), whose Jacobian is \([x_2,\; x_1]\).

**A local pattern** is the sparsity at a specific input point \(x\),
so it depends on the numerical values.
At \(x = (1, 0)\), the local pattern is `[0, 1]`,
but at \(x = (0, 1)\) it is `[1, 0]`.

**A global pattern** is the union of local patterns over the entire input domain,
making it input-independent.
For the same example, the global pattern is always `[1, 1]` (no sparsity).
Global patterns are supersets of local patterns —
less sparse, but valid everywhere.

**`asdex` exclusively computes global patterns**,
so the result depends only on the function's structure,
not on any particular input.

## Why Conservative Patterns Are Required

A global sparsity pattern used for coloring must be either accurate or conservative.
If the pattern misses a nonzero entry,
the coloring may merge rows or columns that actually conflict,
silently producing **wrong results**.
Overestimating extra nonzeros is safe —
it just uses more colors than strictly necessary.
This is why `asdex` errs on the side of conservatism:
correctness comes first.

## How asdex Detects Sparsity

`asdex` uses a form of [abstract interpretation](https://en.wikipedia.org/wiki/Abstract_interpretation):
instead of evaluating the function on real numbers,
it traces the function into JAX's intermediate representation (jaxpr)
and propagates **index sets** forward through the computation graph.
Each input element \(x_j\) starts with the singleton set \(\{j\}\),
and each primitive operation propagates these sets
according to its mathematical structure:

- **Elementwise ops** (sin, exp, add): preserve per-element sets.
- **Reductions** (sum): union all input sets.
- **Indexing** (gather/scatter): route sets based on index structure.

The output index sets directly encode the sparsity pattern:
output \(i\) depends on input \(j\) iff \(j \in S_i\).
No derivatives are evaluated — the analysis is purely structural.

## Sources of Conservatism

Three mechanisms make global patterns conservative:

1. **Branching** (`cond`, `select_n`):
   the detector takes the **union** over all branches,
   since it cannot know which branch will execute at runtime.
   This is the primary difference from local detection.
2. **Multiplication**:
   \(f(x) = x_1 \cdot x_2\) always reports both dependencies globally,
   even though one factor might be zero at a particular input.
3. **Fallback handlers**:
   primitives without a precise handler conservatively assume
   every output depends on every input.

!!! tip

    More precise handlers can be added for fallback primitives
    to reduce conservatism and produce sparser patterns.
    Please open an issue if you encounter overly conservative patterns. 

## Hessian Detection

Hessian sparsity is detected by applying Jacobian detection to the gradient:

\[
\operatorname{hessian\_sparsity}(f) = \operatorname{jacobian\_sparsity}(\nabla f)
\]

This composes naturally with JAX's autodiff:
`jax.grad` produces a jaxpr,
which `asdex` analyzes the same way.

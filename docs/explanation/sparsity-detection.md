# Sparsity Detection

Sparsity detection determines which entries of a Jacobian can be nonzero.
Given \(f: \mathbb{R}^n \to \mathbb{R}^m\),
it computes a binary [global sparsity pattern](global-sparsity.md) —
a conservative superset of the true nonzero structure of the Jacobian \(J_{ij} = \partial f_i / \partial x_j\).
This pattern is the input to [graph coloring](coloring.md),
which exploits the sparsity to reduce the number of AD passes.

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

# Plan

Delete items when completed.

Effort and priority in brackets after each heading:
- Effort: `S` (< 1 session), `M` (1-2 sessions), `L` (3+ sessions)
- Priority: `!` high, `!!` urgent (e.g. `[M!]`, `[S!!]`)

---

## Now

### `scan` per-timestep dependency tracking [L]

The scan handler unions xs dependencies across all iterations,
so `ys[t]` appears to depend on all xs slices even when only `xs[t]` matters.

| Test | Issue |
|------|-------|
| `test_scan_cumulative_sum` | Lower-triangular, not all-xs |
| `test_scan_2d_carry_and_xs` | Progressive block-diagonal |
| `test_scan_reverse` | Upper-triangular |
| `test_scan_noncontiguous_input` | Progressive deps, not all-xs |
| `test_scan_pytree_ys` | Lower-triangular sums |
| `test_scan_length_one` | ys[0] = carry_init = zeros, should have no x deps |
| `test_scan_scalar_carry_scalar_xs` | ys[t] should depend on xs[0..t-1], not all xs |
| `test_scan_ys_independent_of_carry` | ys[t] depends only on xs[t], not all xs |
| `test_scan_with_cond_inside` | Lower-triangular, not all-xs |

All nine tests carry `@pytest.mark.fallback` and `TODO(scan)` comments.

### `diag` via `dynamic_update_slice` is conservative [M]

`jnp.diag(x)` lowers to `dynamic_update_slice` with loop indices.
The true pattern is sparse (out[i*n+i] depends on x[i] only),
but the handler reports `tile(eye(n), (n, 1))`.

Tracked in `test_diag_1d` in `test_platform_index.py`.

### Scatter Pattern 4 (partial-row scatter) [S]

`mat.at[0, :2].set(updates)` still falls back to conservative.
Already tracked with `@pytest.mark.fallback` in `test_scatter_2d`.

## Ideas

- Second-order analysis for `reduce_sum` Hessian precision
  (tracking which *pairs* of inputs interact — beyond the current first-order index-set framework)
- Symbolic index tracking for gather/scatter composition
  (knowing that `gather` at `[2,3]` then `scatter-add` at `[5]`
  only creates `output[5] ← input[2,3]`, not `output[*] ← input[*]`)
- Algebraic cancellation detection
  (e.g. `f(x) - f(x) = 0` — requires symbolic simplification beyond index-set propagation)

# CUTEst

Conservative patterns found via CUTEst integration tests (`tests/test_cutest.py`).
354 passed. Of those, ~60 emit conservativeness warnings.

## Conservative Hessians (22 problems)

Hessian sparsity is detected as `jacobian_sparsity(grad(f))`.

### Severely conservative (detected 100% dense, true density <15%)

| Problem | Extra nnz | True density | Gradient primitives |
|---------|-----------|--------------|---------------------|
| ARGLINA | 39,800 | 0.5% | `reduce_sum`, `split`, `concatenate` |
| VANDANMSLS | 468 | 3.3% | `ge`, `le`, `or`, `select_n`, nested `jit` |

ARGLINA uses `reduce_sum` and `split`/`concatenate`.
The VJP of `reduce_sum` broadcasts cotangents back,
and `split`/`concatenate` introduce coupling across all elements.
This is a structural limitation: the Hessian of `sum(f(Ax))` where `A` is dense
genuinely couples all inputs through `A^T @ ... @ A`,
but when `A` is structured (e.g., an appended identity row),
the true Hessian is sparser than what the graph structure implies.

VANDANMSLS uses `ge`, `le`, `or`, `select_n` to build conditional masks.
The nested `jit` is inlined by `prop_nested_jaxpr`.
The comparison+select chains propagate consts correctly,
but the conservativeness comes from `reduce_sum` coupling
in the gradient jaxpr.

### Moderately conservative (detected 100% dense, true density 15–80%)

These are mostly small problems (n ≤ 8) where the Hessian is nearly dense anyway.
The extra nonzeros come from the same VJP index-expression issue as above,
or from `reduce_sum` in least-squares objectives
(`sum(residuals²)`) coupling residual terms.

| Problem | Extra nnz | True density |
|---------|-----------|--------------|
| ALLINITU | 10 | 37.5% |
| COOLHANSLS | 42 | 48.1% |
| BROWNBS | 2 | 50.0% |
| CLUSTERLS | 2 | 50.0% |
| HEART8LS | 32 | 50.0% |
| GAUSSIAN | 4 | 55.6% |
| HEART6LS | 16 | 55.6% |
| VESUVIOULS | 22 | 65.6% |
| VESUVIALS | 18 | 71.9% |
| VESUVIOLS | 18 | 71.9% |
| BEALE | 1 | 75.0% |
| EXPFIT | 1 | 75.0% |
| MGH17SLS | 6 | 76.0% |
| HELIX | 2 | 77.8% |

For least-squares problems (BROWNBS, HEART6LS, HEART8LS, VESUVI*, etc.),
`f(x) = sum(r_i(x)^2)` produces a Hessian `2 * (J^T J + sum r_i * H_i)`.
The `J^T J` term couples variables that share a residual,
but `sum` in the jaxpr unions all residual dependencies,
making every variable appear coupled to every other.
The true Hessian is sparser because not all residual pairs share variables.

This is a fundamental limitation of the index-set lattice.
The union over `reduce_sum` is exact for the sum itself,
but in the Hessian context, `J^T J` only has nonzeros where `J` columns overlap.
Resolving this would require second-order analysis
(tracking which *pairs* of inputs interact),
which is beyond the current first-order index-set framework.

### Near-exact (few extra nonzeros)

| Problem | Extra nnz | True density |
|---------|-----------|--------------|
| HIMMELBH | 1 | 25.0% |
| LUKSAN13LS | 2 | 5.0% |
| OSBORNEB | 4 | 96.7% |
| QING | 1 | 1.0% |
| SPIN2LS | 15 | 99.9% |

These are nearly exact with only 1–15 extra nonzeros.
The extras are likely structural zeros:
positions where `∂²f/∂x_i∂x_j` is structurally present in the jaxpr
but evaluates to zero for all inputs (e.g., `x_i * 0 * x_j`).

## Conservative Jacobians — equality constraints (13 problems)

### Severely conservative (detected 100% dense, true density <20%)

| Problem | Extra nnz | True density | Key primitives |
|---------|-----------|--------------|----------------|
| HADAMARD | 76,000 | 5.0% | `scan`, `gather`, `select_n`, `cumsum` |
| TRO11X3 | 8,554 | 5.0% | `gather`, `scatter`, `scatter-add` |
| TRO5X5 | 4,022 | 6.9% | `gather`, `scatter`, `scatter-add` |
| TRO4X4 | 1,334 | 11.8% | `gather`, `scatter`, `scatter-add` |
| TRO6X2 | 786 | 12.7% | `gather`, `scatter`, `scatter-add` |
| TRO3X3 | 298 | 17.2% | `gather`, `scatter`, `scatter-add` |

The TRO problems are structural truss optimizations.
Their constraints use `gather` with static indices
to select member properties from a design vector,
and `scatter`/`scatter-add` to assemble stiffness matrices.
The `scatter` here uses Pattern 1 (batched along dim 0) but with indices
that select the same operand row multiple times (repeated scatter targets).
When multiple updates target the same row, the handler correctly unions deps,
but the constraint Jacobian is sparse because each constraint
only involves a small subset of design variables.
The conservativeness comes from gather/scatter coupling:
the gather selects a subset of variables,
then scatter-add writes them into a shared accumulator,
making downstream reads appear to depend on all gathered variables.

This is a precision-of-composition issue.
Each gather and scatter is individually correct,
but composing them through a shared accumulator
(gather from design → compute → scatter-add into global)
unions deps from different members.
Fixing this would require **symbolic index tracking**:
knowing that `gather` at index `[2,3]` followed by `scatter-add` at index `[5]`
only creates a dependency `output[5] ← input[2,3]`, not `output[*] ← input[*]`.
This is architecturally difficult because the current framework
evaluates each primitive independently.

### Fully spurious (ground truth has 0 nnz)

| Problem | Extra nnz |
|---------|-----------|
| TENBARS1 | 62 |
| TENBARS2 | 62 |
| TENBARS3 | 62 |
| TENBARS4 | 62 |
| FLETCHER | 4 |
| S316-322 | 2 |

These constraints are constant w.r.t. the decision variables,
but asdex detects structural dependencies.
This happens when a constraint like `c(x) = sum(x) - constant`
is implemented as `concatenate([computed_terms, constant_terms])`
and the primal jaxpr contains `slice` or `gather` operations
that structurally reference the input even for constant output positions.

For TENBARS1–4, the equality constraints encode structural compatibility
using `sqrt`, `div`, and `mul` with constants —
the jaxpr contains `slice` operations on the design vector
that appear in both variable and constant terms,
so every constraint output structurally depends on the sliced variables
even when the dependence cancels algebraically.

Zero-skipping for `div(0, x)` and `integer_pow(0, n)` has been implemented,
but does not help these problems.
The spurious nonzeros come from **algebraic cancellation** —
detecting `f(x) - f(x) = 0` requires symbolic simplification
that's beyond index-set propagation.

### Small extras

| Problem | Extra nnz |
|---------|-----------|
| BATCH | 6 |
| BT12 | 1 |
| BT8 | 2 |
| HS61 | 2 |

Small extras (1–6 nnz) are structural zeros:
the jaxpr has a path from input to output,
but the derivative is algebraically zero for all inputs
(e.g., `x * y` where `y` is always zero along certain constraint rows).

## Conservative Jacobians — inequality constraints (24 problems)

### Severely conservative

| Problem | Extra nnz | Detected density | True density | Key primitives |
|---------|-----------|-----------------|--------------|----------------|
| HAIFAM | 12,039 | 85.9% | 4.8% | `gather`, `scatter`, nested `jit` |
| OET7 | 3,006 | 100% | 57.1% | `iota`, nested `jit`, `reshape` |
| OET6 | 2,004 | 100% | 60.0% | `iota`, nested `jit`, `reshape` |
| OET4 | 1,004 | 100% | 75.0% | `iota`, nested `jit`, `reshape` |
| OET2 | 1,002 | 100% | 66.7% | `iota`, nested `jit`, `reshape` |

Note: nested `jit` jaxprs *are* inlined by `prop_nested_jaxpr`,
so the conservativeness comes from primitives *inside* those jaxprs,
not from the `jit` boundary itself.

HAIFAM is similar to the TRO problems:
a network/assignment problem using `gather`/`scatter`
with indices that select subsets of design variables.
The gather/scatter composition through shared accumulators
causes the same precision-of-composition issue as the TRO problems.

OET2/4/6/7 are semi-infinite programming problems
whose constraints are parameterized by a discretization grid.
The asdex detection is correct: each constraint structurally depends on all 3 variables
(through `t * exp(v * b)` terms).
The CUTEst fixture shows fewer nonzeros because
the original SIF formulation has a different structure than the sif2jax translation.
At the starting point `y0 = [0, 0, 0]`, `t = 0` makes the `v`-partial numerically zero,
but the structural dependency is real.
This is a **fixture mismatch**, not an asdex conservativeness issue.

### Moderately conservative

| Problem | Extra nnz | True density |
|---------|-----------|--------------|
| HS108 | 20 | 16.2% |
| HS33 | 4 | 33.3% |
| S365 | 6 | 45.7% |
| S365MOD | 6 | 45.7% |
| HAIFAS | 13 | 15.4% |
| HS43 | 3 | 75.0% |

HS108 uses `integer_pow` and `mul` with `neg` —
the extra nonzeros come from products of variables
where one factor is structurally present but algebraically zero.

HS33 uses `integer_pow` with `slice` —
the `x^2` terms couple `x[i]` to itself,
but `concatenate` of independent constraint terms
makes each output appear to depend on the full sliced range.

S365/S365MOD, HAIFAS: similar structural-zero issues
from polynomial and product constraint expressions.

These are mostly structural zeros
that would require algebraic simplification to eliminate.
Zero-skipping in `integer_pow` (when the base is a tracked zero)
has been implemented but doesn't help here —
the bases are not known constants at trace time.

### Small extras (1–4 nnz)

Structural zeros in the Jacobian:
the jaxpr has a path from input to output,
but the derivative evaluates to zero for all inputs.

| Problem | Extra nnz | Likely cause |
|---------|-----------|--------------|
| CB2 | 2 | `exp` product terms |
| CB3 | 2 | `exp` product terms |
| CHACONN2 | 2 | `integer_pow` product terms |
| DIPIGRI | 2 | polynomial cross-terms |
| GIGOMEZ2 | 2 | `integer_pow` product terms |
| GIGOMEZ3 | 2 | `integer_pow` product terms |
| HS100 | 2 | polynomial cross-terms |
| HS113 | 3 | polynomial cross-terms |
| HS65 | 1 | polynomial cross-terms |
| MAKELA2 | 1 | `exp` terms |
| OET1 | 2 | `iota`-derived indices inside nested `jit` |
| OET3 | 4 | `iota`-derived indices inside nested `jit` |
| PENTAGON | 3 | trigonometric cross-terms |
| SIPOW1 | 4 | `iota`-based indexing |
| SIPOW2 | 4 | `iota`-based indexing |

Most of these (CB2, CB3, CHACONN2, GIGOMEZ*, DIPIGRI, HS*, PENTAGON, MAKELA2)
have constraints of the form `g(x_i, x_j)` where the jaxpr's `mul` or `exp`
structurally involves `x_k` but the partial `∂g/∂x_k` is always zero.
These are inherent to index-set propagation and can't be improved
without value-dependent analysis.

OET1/OET3, SIPOW1/SIPOW2 have small extras (2–4 nnz)
from structural zeros in `iota`-based indexing inside nested `jit`.

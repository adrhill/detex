# TODO

Remaining handler issues found during the post-hardening audit (PR #51)
and the conservative-pattern audit.

## 1. `scan` merges deps across all time steps

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

All nine tests now carry `@pytest.mark.fallback` and `TODO(scan)` comments.

## 2. `diag` via `dynamic_update_slice` is conservative

`jnp.diag(x)` lowers to `dynamic_update_slice` with loop indices.
The true pattern is sparse (out[i*n+i] depends on x[i] only),
but the handler reports `tile(eye(n), (n, 1))`.

Tracked in `test_diag_1d` in `test_platform_index.py`.

## 3. Scatter Pattern 4 (partial-row scatter)

`mat.at[0, :2].set(updates)` still falls back to conservative.
Already tracked with `@pytest.mark.fallback` in `test_scatter_2d`.

# CUTEst

Conservative patterns found via CUTEst integration tests (`tests/test_cutest.py`).
354 passed. Of those, ~60 emit conservativeness warnings.

## Conservative Hessians (27 problems)

Hessian sparsity is detected as `jacobian_sparsity(grad(f))`.
The gradient jaxpr contains `gather`, `scatter-add`, and `pad` primitives
from VJP rules that don't appear in the primal.
The VJP scatter-add uses a `ScatterDimensionNumbers` configuration
that the scatter handler doesn't recognize,
causing it to fall back to conservative
even when the indices are const.

### Root cause: unhandled `scatter-add` dimension configuration

The primal jaxpr of problems like GENROSE uses simple `slice` operations
(e.g., `x[1:]`, `x[:-1]`).
But `grad(f)` replaces these with scatter-add to accumulate cotangents:
`iota` generates position arrays, `lt`/`select_n` build masks,
and `scatter-add` or `gather` uses these as indices.

The const propagation chain works correctly:
`iota → add → lt → select_n → broadcast_in_dim` all propagate through `const_vals`,
and `gather` receives const indices and produces precise patterns.
However, `scatter-add` falls back to conservative because its
`ScatterDimensionNumbers(update_window_dims=(1,), inserted_window_dims=(), ...)`
configuration is not recognized by the scatter handler,
even though the indices are available as consts.

**Handler improvement**: Extend the scatter handler to recognize
the VJP scatter-add configuration
(`update_window_dims=(1,)` with a 2D update and 2D index array).
This would make the VJP scatter-add as precise as the primal `slice` operations.

### Severely conservative (detected 100% dense, true density <15%)

| Problem | Extra nnz | True density | Gradient primitives |
|---------|-----------|--------------|---------------------|
| GENROSE | 248,502 | 0.6% | `gather`, `scatter-add`, `iota`, `lt`, `select_n`, `pad` |
| ARGLINA | 39,800 | 0.5% | `reduce_sum`, `split`, `concatenate` |
| COATING | 15,656 | 12.8% | `gather`, `scatter-add`, `iota`, `lt`, `select_n`, `split` |
| LUKSAN17LS | 9,408 | 5.9% | `gather`, `scatter-add`, `iota`, `lt`, `select_n`, `pad` |
| LUKSAN21LS | 9,506 | 4.9% | `gather`, `scatter-add`, `iota`, `lt`, `select_n`, `pad` |
| ERRINROS | 2,352 | 5.9% | `gather`, `scatter-add`, `iota`, `lt`, `select_n`, `pad` |
| VANDANMSLS | 468 | 3.3% | `ge`, `le`, `or`, `select_n`, nested `jit` |

GENROSE, COATING, LUKSAN17LS, LUKSAN21LS, ERRINROS all share the same pattern:
the primal uses `slice` for shifting operations (`x[1:] - x[:-1]`),
whose VJP introduces `gather`/`scatter-add` with `iota`-derived indices.
The `gather` half is already precise (const propagation through the
`iota → lt → select_n` chain works).
The conservativeness comes entirely from the `scatter-add` half.

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
but the downstream scatter/gather uses an unhandled dimension configuration.

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

**Handler improvement**: This is a fundamental limitation of the index-set lattice.
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

## Conservative Jacobians — equality constraints (14 problems)

### Severely conservative (detected 100% dense, true density <20%)

| Problem | Extra nnz | True density | Key primitives |
|---------|-----------|--------------|----------------|
| HADAMARD | 76,000 | 5.0% | `scan`, `gather`, `select_n`, `cumsum` |
| TRO11X3 | 8,554 | 5.0% | `gather`, `scatter`, `scatter-add` |
| MSS1 | 6,192 | 5.8% | `gather`, `iota`, `lt`, `select_n` |
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

**Handler improvement**: This is a precision-of-composition issue.
Each gather and scatter is individually correct,
but composing them through a shared accumulator
(gather from design → compute → scatter-add into global)
unions deps from different members.
Fixing this would require **symbolic index tracking**:
knowing that `gather` at index `[2,3]` followed by `scatter-add` at index `[5]`
only creates a dependency `output[5] ← input[2,3]`, not `output[*] ← input[*]`.
This is architecturally difficult because the current framework
evaluates each primitive independently.

MSS1 is a network flow problem with similar gather-based indexing.
The `iota + lt + select_n` chain propagates consts correctly,
but the downstream scatter uses an unhandled dimension configuration.

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
Each constraint row depends on a small subset of the 3 decision variables.
The nested `jit` bodies contain `iota`-based index arithmetic
and comparison+select chains that build index arrays.
Const propagation through these chains works,
but the downstream gather/scatter uses unhandled dimension configurations.

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

**Handler improvement**: These are mostly structural zeros
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

OET1/OET3, SIPOW1/SIPOW2 would be resolved by
handling the VJP scatter-add dimension configuration (see Summary §1).

## Summary of handler improvements

Ordered by estimated impact (number of CUTEst problems resolved):

### 1. Handle VJP `scatter-add` dimension configuration

VJP rules for `slice` produce `scatter-add` with
`ScatterDimensionNumbers(update_window_dims=(1,), inserted_window_dims=(), ...)`,
a 2D index array `(n, 1)`, and 2D updates `(n, 1)`.
The indices are const (propagated through the `iota → lt → select_n` chain),
but the scatter handler doesn't recognize this configuration.

**Handler improvement**: Extend the scatter handler to support
this dimension configuration with const indices.
Each row of the index array selects one output position;
this is essentially a batched single-element scatter.

**Impact**: Resolves GENROSE, COATING, ERRINROS, LUKSAN17LS, LUKSAN21LS (Hessian),
OET2/4/6/7 (inequality Jacobian), and partially MSS1, VANDANMSLS, HAIFAM.
~12–15 problems affected across Hessian and Jacobian tests.
This is the single highest-impact improvement.

### 2. ~~Zero-skipping in `div` and `integer_pow`~~ ✓ Done

Implemented in `_div.py` and `_elementwise.py`:
- `div(0, x)` now clears deps (like `mul(0, x)` already did).
- `integer_pow(0, n)` for `n > 1` now clears deps.
- All three (`mul`, `div`, `integer_pow`) now propagate value bounds
  via interval arithmetic for downstream gather/scatter precision.

**CUTEst impact**: None of the flagged problems (TENBARS, FLETCHER, S316-322)
actually benefit — their spurious nonzeros come from algebraic cancellations
(e.g. `f(x) - f(x) = 0`), not from `div(0, x)` or `integer_pow(0, n)` patterns.
The bounds propagation will help when `mul`/`div`/`integer_pow`
appear in index computation chains feeding into gather/scatter.

### 3. Merge value bounds in `select_n` for dynamic predicates

Currently `select_n` only propagates bounds when the predicate is a known constant.
When the predicate is dynamic, bounds from both branches could be merged
as `(min(lo_a, lo_b), max(hi_a, hi_b))`.

**Impact**: Would allow Python `//` (floor division) to propagate bounds end-to-end.
`//` lowers to a nested jaxpr containing `div`, `rem`, `sign`, and `select_n` —
the `select_n` with a dynamic predicate currently blocks bounds flow.

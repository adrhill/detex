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

## Later

### Symbolic index tracking for gather/scatter composition [L]

Composing gather → compute → scatter-add through a shared accumulator
unions deps from different members,
making `output[*]` appear to depend on `input[*]`.
Fixing this requires tracking that `gather` at `[2,3]` then `scatter-add` at `[5]`
only creates `output[5] ← input[2,3]`.

Architecturally difficult: the current framework evaluates each primitive independently.

**CUTEst impact** (see `CUTEST_ANALYSIS.md`):
TRO3X3/4X4/5X5/6X2/11X3 (298–8,554 extra nnz),
HAIFAM (12,039 extra nnz), HAIFAS (13 extra nnz).

### Second-order analysis for `reduce_sum` Hessian precision [L]

The VJP of `reduce_sum` broadcasts cotangents back,
so `J^T J` appears fully dense even when `J` columns don't all overlap.
For least-squares `f(x) = sum(r_i(x)^2)`,
the Hessian has nonzeros only where `J` columns overlap,
but `sum` unions all residual dependencies.
Fixing this requires tracking which *pairs* of inputs interact,
beyond the current first-order index-set framework.

**CUTEst impact** (see `CUTEST_ANALYSIS.md`):
ARGLINA (39,800 extra nnz), VANDANMSLS (468),
plus 14 moderately conservative and 5 near-exact Hessian problems.

## Ideas

- Algebraic cancellation detection
  (e.g. `f(x) - f(x) = 0` — requires symbolic simplification beyond index-set propagation).
  Would fix TENBARS1–4 (62 extra nnz each), FLETCHER (4), S316-322 (2).

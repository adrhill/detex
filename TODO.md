# TODO

Remaining handler issues found during the post-hardening audit (PR #51)
and the conservative-pattern audit.

## 1. `const_vals` not propagated through shape-transforming ops

`_reshape`, `_slice`, `_transpose`, and `_tile` don't propagate `const_vals`.
Index arrays passing through these ops before gather/scatter lose static tracking
and cause conservative fallback.

Already confirmed: `test_gather_indices_through_reshape` is xfailed for this reason.

## 2. Dynamic-index fallbacks are overly conservative

When indices depend on input, handlers fall back to all-ones.
In many cases the index range is bounded and could be narrowed.

| Test | Handler | Issue |
|------|---------|-------|
| `test_dynamic_slice_dynamic_start` | `dynamic_slice` | Slice window is bounded; `argmax(x[:2])` → {0,1}, so x[4] is never touched |
| `test_dynamic_update_slice_dynamic_start` | `dynamic_update_slice` | Same bounded-index issue |
| `test_gather_dynamic_indices_fallback` | `gather` | `argmax(x[:2])` → {0,1}, so indices are `[0,1]` or `[1,2]` — x[3] is unreachable |
| `test_scatter_dynamic_indices` | `scatter` | `argmax(x[3:])` on 2 elements → {0,1}, so out[2] always equals x[2] |

All four tests now carry `@pytest.mark.fallback` and `TODO` comments with the precise pattern.

## 3. `dot_general` ignores constant operand values

The `dot_general` handler unions all input indices for each output,
ignoring value-level zeros in constant matrices.

| Test | Issue |
|------|-------|
| `test_matmul_with_constant` | `W @ x` with sparse `W = [[1,0,0],[0,1,0]]` gives all-ones instead of W's nonzero structure |
| `test_iota_eye` | `jnp.eye(3) @ x` gives all-ones instead of identity |

Both tests now carry `@pytest.mark.fallback` and `TODO(dot_general)` comments.

## 4. `elementwise` mul doesn't special-case multiplication by zero

`0 * x` reports `[[1]]` even though the zero is a known constant via `const_vals`.
The mul handler could propagate an empty index set when one operand is zero.

Tracked in `test_multiply_by_zero` in `test_detection.py`
and `test_multiply_by_zero_hessian` in `test_scalar.py`.

## 5. `scan` merges deps across all time steps

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

## 6. `diag` via `dynamic_update_slice` is conservative

`jnp.diag(x)` lowers to `dynamic_update_slice` with loop indices.
The true pattern is sparse (out[i*n+i] depends on x[i] only),
but the handler reports `tile(eye(n), (n, 1))`.

Tracked in `test_diag_1d` in `test_platform_index.py`.

## 7. Scatter Pattern 4 (partial-row scatter)

`mat.at[0, :2].set(updates)` still falls back to conservative.
Already tracked with `@pytest.mark.fallback` in `test_scatter_2d`.

# TODO

Remaining handler issues found during the post-hardening audit (PR #51)
and the conservative-pattern audit.

## 1. Dynamic-index fallbacks are overly conservative

When indices depend on input, handlers fall back to all-ones.
In many cases the index range is bounded and could be narrowed.

| Test | Handler | Issue |
|------|---------|-------|
| `test_dynamic_slice_dynamic_start` | `dynamic_slice` | Slice window is bounded; `argmax(x[:2])` → {0,1}, so x[4] is never touched |
| `test_dynamic_update_slice_dynamic_start` | `dynamic_update_slice` | Same bounded-index issue |
| `test_gather_dynamic_indices_fallback` | `gather` | `argmax(x[:2])` → {0,1}, so indices are `[0,1]` or `[1,2]` — x[3] is unreachable |
| `test_scatter_dynamic_indices` | `scatter` | `argmax(x[3:])` on 2 elements → {0,1}, so out[2] always equals x[2] |

All four tests now carry `@pytest.mark.fallback` and `TODO` comments with the precise pattern.

## 2. `scan` merges deps across all time steps

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

## 3. `diag` via `dynamic_update_slice` is conservative

`jnp.diag(x)` lowers to `dynamic_update_slice` with loop indices.
The true pattern is sparse (out[i*n+i] depends on x[i] only),
but the handler reports `tile(eye(n), (n, 1))`.

Tracked in `test_diag_1d` in `test_platform_index.py`.

## 4. Scatter Pattern 4 (partial-row scatter)

`mat.at[0, :2].set(updates)` still falls back to conservative.
Already tracked with `@pytest.mark.fallback` in `test_scatter_2d`.

## 5. `conv_general_dilated` with `batch_group_count > 1`

The handler falls back to conservative when `batch_group_count > 1`.
The true pattern is block-diagonal (64/576 nnz):
each batch group's outputs depend only on that group's inputs.

Tracked in `test_conv_batch_group_count` in `test_conv.py`.

## 6. `gather` unrecognized dimension_numbers patterns

The gather handler only recognizes specific patterns of `GatherDimensionNumbers`.
Unrecognized configurations fall back to conservative.

| Test | Issue |
|------|-------|
| `test_gather_single_dim_start_map_mismatch` | `start_index_map` doesn't match collapsed dim; true pattern is a permutation (2/24 nnz) |
| `test_gather_single_dim_partial_slice` | Non-collapsed slice doesn't span full operand; true pattern is a permutation (4/48 nnz) |
| `test_gather_multi_dim_start_map_mismatch` | Reversed `start_index_map`; true nnz is 5/600 |
| `test_gather_multi_dim_partial_non_collapsed` | Partial non-collapsed slice; true pattern is two slices (6/360 nnz) |
| `test_gather_batching_dims` | `operand_batching_dims` not yet supported; true pattern is a permutation (2/12 nnz) |

All five tests carry `@pytest.mark.fallback` and `TODO(gather)` comments.

# Plan

Delete items when completed.

Effort and priority in brackets after each heading:
- Effort: `S` (< 1 session), `M` (1-2 sessions), `L` (3+ sessions)
- Priority: `!` high, `!!` urgent (e.g. `[M!]`, `[S!!]`)

---

## Now

### `scan` per-timestep dependency tracking [M]

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

#### Key insight: forward simulation, not fixed-point

Scan is a forward computation, not a circular one.
Unlike `while_loop` (unknown iteration count, same inputs each iteration),
scan has a known `length` and different `xs[t]` per timestep.
There is no circularity when xs aren't merged,
so `fixed_point_loop` is unnecessary — plain forward simulation is both simpler and exact.

#### Current handler (3 stages, all lossy)

1. **Merge xs** (lines 72–88): unions `xs[t]` across all `t` into one `x_slice_indices`.
   This is the root cause — timestep identity is destroyed here.
2. **Fixed-point on carry** (lines 90–96): iterates the body with merged xs
   until carry stabilizes. Correct but overapproximate because the input is merged.
3. **Tile ys** (lines 102–114): copies the converged `y_slice` to every timestep.
   All `ys[t]` get identical (overapproximate) deps.

#### Proposed approach: transfer matrices

The body jaxpr is a fixed function `(carry, x_slice) → (new_carry, y_slice)`.
Its dependency structure can be decomposed into four boolean transfer matrices:

- **C→C**`[i,j]`: `carry_out[i]` depends on `carry_in[j]`
- **X→C**`[i,j]`: `carry_out[i]` depends on `x_slice[j]`
- **C→Y**`[i,j]`: `y_slice[i]` depends on `carry_in[j]`
- **X→Y**`[i,j]`: `y_slice[i]` depends on `x_slice[j]`

These can be extracted with a single `prop_jaxpr` call
using indicator index sets (`carry_in[i] = {i}`, `x_slice[j] = {carry_size + j}`),
then reading off which indicators appear in each output.

Forward simulation becomes boolean matrix-vector operations:

```
carry_deps[0] = init_carry_deps
for t in range(length):          # reversed(range(length)) if reverse=True
    xs_t = xs_deps[t]
    ys_deps[t][i] = ∪{carry_deps[j] : C→Y[i,j]} | ∪{xs_t[j] : X→Y[i,j]}
    carry_deps[i] = ∪{carry_deps[j] : C→C[i,j]} | ∪{xs_t[j] : X→C[i,j]}
```

Cost: `O(prop_cost + length × (carry_size + xs_slice_size)²)`.
The `prop_jaxpr` call happens once; the forward loop is cheap set unions.

#### Soundness: transfer matrices must be conservative

Transfer matrices are extracted once and reused for all timesteps,
so they must be conservative for *every* possible carry value, not just `carry_init`.

**Const values for carry and x_slice must NOT be passed** during extraction.
Only closed-over body constants (which don't change across timesteps) may use const info.
Otherwise, const-killing produces matrices that are too tight:
e.g. body `(c, x) → (c + x, c * x)` with `carry_init = 0`
would kill X→Y (since `0 * x = 0`),
but at `t=1` carry is non-zero and `y = carry * x` genuinely depends on `x`.

Without const info for carry/x_slice, the matrices overapproximate —
every structural dependency is preserved regardless of runtime values.
This means the forward simulation is conservative at every timestep:
carry_deps grows monotonically (union of conservative deps),
and ys_deps inherits that conservativeness.

**Trade-off:** extracting without carry const info
loses precision at `t=0` when `carry_init` is a known constant.
For example, `test_scan_length_one` expects `ys[0]` to have no x deps
because `carry_init = zeros` and `y = carry`.
With conservative transfer matrices, C→Y would (correctly) show no x dep here
because `y = carry` means X→Y is structurally zero regardless of const info.
But for bodies like `y = carry * x`, conservative matrices would report
a spurious x dep at `t=0` — which is conservative (safe), just not tight.

#### Alternative: per-timestep `prop_jaxpr`

Instead of transfer matrices, call `prop_jaxpr` once per timestep
with actual carry deps and xs[t] deps, threading carry forward:

```
carry_deps = init_carry_deps
for t in range(length):
    body_out = prop_jaxpr(body, consts + carry_deps + xs_deps[t], state_consts)
    ys_deps[t] = body_out[num_carry:]
    carry_deps = body_out[:num_carry]
```

This is sound because `prop_jaxpr` is always conservative,
and it naturally handles const-killing (pass actual `state_consts` each call).
Cost is `O(length × prop_cost)`.
Simpler to implement but slower for large lengths.

Both approaches are sound.
The transfer matrix approach is faster;
the per-timestep approach is more precise and simpler to implement.
Start with per-timestep, optimize to transfer matrices later if needed.

#### Implementation plan

1. Replace xs merging + fixed-point + tiling with forward simulation.
2. Call `prop_jaxpr` per timestep, threading carry deps forward.
3. Handle `reverse=True` by iterating `t` in reverse order.
4. Write per-timestep ys deps directly (no tiling).
5. Update all 9 tests to assert the precise (not overapproximate) pattern.
6. Optimize: extract transfer matrices (without carry/x_slice const info)
   and replace per-timestep `prop_jaxpr` with boolean matrix-vector operations.

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

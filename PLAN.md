# Plan

Delete items when completed.

Effort and priority in brackets after each heading:
- Effort: `S` (< 1 session), `M` (1-2 sessions), `L` (3+ sessions)
- Priority: `!` high, `!!` urgent (e.g. `[M!]`, `[S!!]`)

---

## Now

### Add inline comments to expected matrices in test files [S]

Annotate expected sparsity matrices with inline comments
that explain what each row computes, directly on the array literal.
This makes test expectations self-documenting
without requiring the reader to cross-reference the docstring.

Move explanations out of docstrings and into the matrix itself.
Keep docstrings short (one-line summary + setup context).

**Style:**

```python
expected = np.array(
    [
        [1, 1, 1],  # carry_out = x[0] + x[1] + x[2]
        [0, 0, 0],  # ys[0] = carry_init = 0
        [1, 0, 0],  # ys[1] = x[0]
        [1, 1, 0],  # ys[2] = x[0] + x[1]
    ],
    dtype=int,
)
```

When consecutive rows share the same pattern (e.g. elementwise over a 2-element array),
annotate only the first row of each group.

Apply to all `tests/_interpret/test_*.py` files, not just `test_scan.py`.

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

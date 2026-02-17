# _interpret — Custom Jaxpr Interpreter for Index Set Propagation

Implements a custom jaxpr interpreter that propagates per-element dependency index sets (`set[int]`)
through primitives to determine Jacobian sparsity patterns.

## Structure

```
__init__.py        # prop_jaxpr, prop_dispatch, fallbacks
_commons.py        # IndexSets, Deps, ConstVals, utilities
_elementwise.py    # unary, binary, zero-derivative, integer_pow, convert_element_type
_slice.py          # slice
_squeeze.py        # squeeze
_reshape.py        # reshape
_broadcast.py      # broadcast_in_dim
_concatenate.py    # concatenate
_transpose.py      # transpose (dimension permutation)
_rev.py            # rev (reverse along dimensions)
_reduce.py         # reduce_sum, reduce_max, reduce_min, reduce_prod
_top_k.py          # top_k (top-k selection along last axis)
_gather.py         # gather (static/dynamic indices)
_scatter.py        # scatter, scatter-add (static/dynamic indices)
_select.py         # select_n
_conv.py           # conv_general_dilated
_dot_general.py    # dot_general (generalized matrix multiply)
_while.py          # while_loop (fixed-point iteration)
_scan.py           # scan (fixed-point iteration on carry)
_cond.py           # cond (union over branches)
_split.py          # split (partition along axis)
_tile.py           # tile (repeat along dimensions)
_dynamic_slice.py  # dynamic_slice, dynamic_update_slice
_platform_index.py # platform_index (platform selection constant)
```

## Key Types

- `IndexSets` = `list[set[int]]` — per-element dependency sets for one array
- `Deps` = `dict[Var, IndexSets]` — maps jaxpr variables to their index sets
- `ConstVals` = `dict[Var, np.ndarray]` — statically-known values for precise gather/scatter

## Common Utilities in `_commons.py`

- **`position_map(shape)`** —
  builds an array where each element holds its own flat position.
  Used to derive permutation maps by applying operations (transpose, slice, etc.)
  to this array.
- **`permute_indices(in_indices, permutation_map)`** —
  builds output index sets by copying from input positions according to a map.
  Used by handlers where each output reads exactly one input element
  (transpose, rev, slice, reshape, broadcast, split, tile, gather, dynamic_slice).
- **`fixed_point_loop(iterate_fn, carry, n_carry)`** —
  runs a body function on carry index sets until they stabilize.
  Used by `while_loop` and `scan`.
- **`conservative_indices(all_indices, out_size)`** —
  conservative fallback where every output element depends on the union of all inputs.

## Naming Conventions

- **"indices" / "index sets"**: refers to `IndexSets` (`list[set[int]]`),
  the per-element dependency sets used for sparsity tracking.
- **"map"**: refers to integer arrays that map output positions to input positions
  (e.g., `permutation_map`, `position_map`).
  These are numpy arrays, not index sets.
- Avoid using "deps" in docstrings; prefer "index sets" or "input index sets".

## Const Value Tracking

Handlers like `broadcast_in_dim`, `select_n`, and binary ops propagate concrete values through `const_vals`.
This lets `gather`/`scatter` resolve static indices precisely instead of falling back to conservative.

## Tests

Each handler module `_foo.py` has a corresponding test file `tests/_interpret/test_foo.py`.

## Adding a New Handler

1. Write `prop_<name>(eqn, deps, ...)` in the appropriate module.
2. Add a `case` branch in `prop_dispatch`.
3. Remove from the fallback `case` group if upgrading from conservative.
4. Add tests in the corresponding `tests/_interpret/test_<module>.py` file.

## Writing Style

Use **semantic line breaks** everywhere: one sentence or clause per line in docstrings, comments, and markdown.
This applies to all prose, not just docstrings.

## Comments

Focus comments on **why**, not what.
Explain why a branch exists, why a particular approach was chosen, or why a fallback is needed.
Don't narrate what the code already says.

## Handler Docstring Style

1. **Semantic summary**: What the operation does and how dependencies flow.
2. **Math**: The Jacobian structure in concise mathematical notation.
3. **Example**: A concrete input/output trace showing dependency sets before and after.
4. **Jaxpr**: The `eqn.invars` and `eqn.params` layout the handler reads.
5. **URL**: Link to the JAX docs for the primitive, as a bare URL on the last line.

## References

- [Understanding jaxprs](https://docs.jax.dev/en/latest/jaxpr.html)
- [Writing custom jaxpr interpreters](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html)

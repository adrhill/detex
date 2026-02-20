# _interpret — Custom Jaxpr Interpreter for Index Set Propagation

Implements a custom jaxpr interpreter that propagates per-element dependency index sets (`set[int]`)
through primitives to determine Jacobian sparsity patterns.

## Structure

- `__init__.py` — `prop_jaxpr`, `prop_dispatch`, fallback handling.
- `_commons.py` — shared types (`IndexSets`, `Deps`, `ConstVals`) and utilities.
- Each JAX primitive has its own module: `_foo.py` contains `prop_foo`.
- Handlers for external packages (Equinox, Flax, etc.) live in their own subfolders
  (e.g., `_equinox/`).

## Key Types

- `IndexSets` = `list[set[int]]` — per-element dependency sets for one array
- `Deps` = `dict[Var, IndexSets]` — maps jaxpr variables to their index sets
- `ConstVals` = `dict[Var, np.ndarray]` — statically-known values for precise gather/scatter

## Naming Conventions

**Terminology** — "indices" and "map" mean different things:
- **"indices" / "index sets"**: `IndexSets` (`list[set[int]]`),
  the per-element dependency sets used for sparsity tracking.
- **"map"**: numpy integer arrays that map output positions to input positions.
  Not index sets.

**Variable names** — use these consistently across handlers:
- `in_indices`: input index sets (from `index_sets(deps, atom)`)
- `in_shape`: input array shape (from `atom_shape(atom)`)
- `in_val`: const value for a unary input (from `atom_const_val(atom, const_vals)`)
- `in1_val` / `in2_val`: const values for binary inputs.
  Use descriptive prefixes when roles differ:
  `lhs_val` / `rhs_val` (dot_general), `pred_val` / `which_val` (select), etc.
- `permutation_map`: the flat integer array passed to `permute_indices()`
- `in_position_map`: when a `position_map()` result is stored
  (e.g., reused across loop iterations in `_split.py`)

**Docstrings** — avoid the term "deps"; prefer "index sets" or "input index sets".

## Common Utilities in `_commons.py`

- **`position_map(shape)`** —
  builds an array where each element holds its own flat position.
  Applying operations (transpose, slice, flip) to this array
  reveals which input position each output position reads from.
- **`permute_indices(in_indices, permutation_map)`** —
  builds output index sets by copying from input positions according to a map.
  Used by handlers where each output reads exactly one input element
  (transpose, rev, slice, reshape, broadcast, split, tile, gather, dynamic_slice).
- **`fixed_point_loop(iterate_fn, carry, n_carry)`** —
  runs a body function on carry index sets until they stabilize.
  Used by `while_loop` and `scan`.
- **`propagate_const_unary(eqn, const_vals, transform)`** —
  propagates a const value through a unary op by applying `transform`.
  Mirrors `propagate_const_binary` for the single-input case.
- **`conservative_indices(all_indices, out_size)`** —
  conservative fallback where every output element depends on the union of all inputs.

## Const Value Tracking

Handlers like `broadcast_in_dim`, `select_n`, and `propagate_const_elementwise`
propagate concrete values through `const_vals`.
This lets downstream handlers resolve static indices precisely.

**Invariant**: if a required const value is missing from `const_vals`,
the handler must assume the worst and return a conservative pattern.
This applies to `gather`, `scatter`, `dynamic_slice`, `dynamic_update_slice`,
`dot_general` (zero-skipping), and `mul` (zero-clearing).

## Adding a New Handler

1. Write `prop_<name>(eqn, deps, ...)` in the appropriate module.
2. Add a `case` branch in `prop_dispatch`.
3. Remove from the fallback `case` group if upgrading from conservative.
4. Add tests in the corresponding `tests/_interpret/test_<module>.py` file.

For primitives from external packages (Equinox, Flax, etc.),
place the handler in a dedicated subfolder (e.g., `_equinox/_select_if_vmap.py`)
with tests in `tests/_interpret/_equinox/`.

## Tests

Each handler module `_foo.py` has a corresponding test file `tests/_interpret/test_foo.py`.

## Writing Style

Use **semantic line breaks** everywhere:
one sentence or clause per line in docstrings, comments, and markdown.
This applies to all prose, not just docstrings.

Focus comments on **why**, not what.
Explain why a branch exists, why a particular approach was chosen, or why a fallback is needed.
Don't narrate what the code already says.

### Handler Docstring Style

1. **Semantic summary**: What the operation does and how dependencies flow.
2. **Math**: The Jacobian structure in concise mathematical notation.
3. **Example**: A concrete input/output trace showing dependency sets before and after.
4. **Jaxpr**: The `eqn.invars` and `eqn.params` layout the handler reads.
5. **URL**: Link to the JAX docs for the primitive, as a bare URL on the last line.

## References

- [Understanding jaxprs](https://docs.jax.dev/en/latest/jaxpr.html)
- [Writing custom jaxpr interpreters](https://docs.jax.dev/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html)

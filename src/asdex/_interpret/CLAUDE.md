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
_reduction.py      # reduce_sum
_reduce_max.py     # reduce_max
_gather.py         # gather (static/dynamic indices)
_scatter.py        # scatter, scatter-add (static/dynamic indices)
_select.py         # select_n
_conv.py           # conv_general_dilated
_dot_general.py    # dot_general (generalized matrix multiply)
_while.py          # while_loop (fixed-point iteration)
_cond.py           # cond (union over branches)
_dynamic_slice.py  # dynamic_slice, dynamic_update_slice
```

## Key Types

- `IndexSets` = `list[set[int]]` — per-element dependency sets for one array
- `Deps` = `dict[Var, IndexSets]` — maps jaxpr variables to their index sets
- `ConstVals` = `dict[Var, np.ndarray]` — statically-known values for precise gather/scatter

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

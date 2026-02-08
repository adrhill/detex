---
name: add-handler
description: Add a precise primitive handler to the jaxpr interpreter, replacing a conservative fallback.
argument-hint: "[primitive-name]"
disable-model-invocation: true
---

# Add precise handler for `$ARGUMENTS`

Add a precise propagation handler for the `$ARGUMENTS` primitive,
replacing the conservative fallback.

## Workflow

### 1. Research

Do all research in this step using extended thinking.
Before writing code:

- Read the JAX docs for the primitive: fetch `https://docs.jax.dev/en/latest/_autosummary/jax.lax.$ARGUMENTS.html`
- Read `src/asdex/_interpret/CLAUDE.md` for conventions (docstring style, semantic line breaks, handler structure)
- Read `src/asdex/_interpret/_commons.py` to understand available utilities
- Read an existing handler with similar structure (e.g. `_pad.py`, `_transpose.py`, `_reduction.py`) as a reference
- Read the existing test in `tests/_interpret/test_internals.py` for the primitive (search for `$ARGUMENTS`)
- Read `src/asdex/_interpret/__init__.py` to see the current dispatch and fallback setup

Understand the primitive's semantics:
how do input and output element indices map to each other?
What is the Jacobian structure (permutation, selection, block-diagonal, etc.)?

### 2. Implement handler

Create `src/asdex/_interpret/_$ARGUMENTS.py` with `prop_$ARGUMENTS(eqn, deps)`.

Follow the handler docstring style from `_interpret/CLAUDE.md`:
1. **Semantic summary**: What the operation does and how dependencies flow.
2. **Math**: The Jacobian structure in concise mathematical notation.
3. **Example**: A concrete input/output trace showing dependency sets before and after.
4. **Jaxpr**: The `eqn.invars` and `eqn.params` layout the handler reads.
5. **URL**: Link to the JAX docs for the primitive, as a bare URL on the last line.

### 3. Wire up dispatch

In `src/asdex/_interpret/__init__.py`:
- Import the new handler
- Add a `case "$ARGUMENTS":` branch in `prop_dispatch` calling the handler
- Remove `"$ARGUMENTS"` from the conservative fallback `case` group

### 4. Update tests

In `tests/_interpret/test_internals.py`:
- Update the existing test: change expected values from dense (`np.ones`) to the precise pattern
- Remove the `@pytest.mark.fallback` marker and `TODO` comments

Create `tests/_interpret/test_$ARGUMENTS.py` with thorough tests:
- Multiple dimensionalities (1D, 2D, 3D, 4D where applicable)
- Edge cases (size-1 dimensions, identity/trivial parameters)
- Real-world usage patterns (e.g. `jnp` functions that lower to this primitive)

### 5. Update docs

- `TODO.md`: check off the primitive and its test items
- `src/asdex/_interpret/CLAUDE.md`: add the new module to the file listing

### 6. Verify

Run in order:
```bash
uv run ruff check src/asdex/_interpret/_$ARGUMENTS.py
uv run pytest tests/_interpret/test_$ARGUMENTS.py -v
uv run pytest tests/_interpret/test_internals.py -v
uv run pytest tests/ -x
```

### 7. Adversarial tests

Reread the JAX docs for the primitive: fetch `https://docs.jax.dev/en/latest/_autosummary/jax.lax.$ARGUMENTS.html`

Try to break the implementation by testing inputs the handler might not expect:

- **Dimensionality**: 1D, 2D, 3D, and higher — if any are missing, add them.
- **Degenerate shapes**: size-0 dimensions, size-1 dimensions, scalar inputs (where the primitive supports them).
- **Boundary parameters**: empty parameter lists, all-dimensions, single-dimension, negative indices (if applicable).
- **Compositions**: the primitive chained with itself (e.g. double-reverse, transpose-of-transpose) or with related ops.
- **Non-contiguous patterns**: inputs where dependencies are not simply `{i}` per element (e.g. from a prior broadcast or reduction) to verify `.copy()` and set merging behave correctly.

For each new test, verify the expected output by hand or against `jax.jacobian`.
Update and re-verify the handler if any test reveals a bug.

### 8. Simplify

Review the implementation with fresh eyes and look for opportunities to reduce complexity:

- **Vectorize loops**: can per-element Python loops be replaced with numpy operations?
  Pattern: build a flat permutation or index array with `np.arange`, `np.flip`, `np.transpose`, `np.indices`, or `np.ravel_multi_index`,
  then index into `in_indices` in a single list comprehension.
  See `_rev.py`, `_reshape.py`, `_concatenate.py`, and `_broadcast.py` for examples.
- **Remove unused imports**: after vectorizing, utilities like `flat_to_coords`, `row_strides`, and `numel` may no longer be needed.
- **Eliminate intermediate variables**: if a value is computed and used only once, inline it.
- **Simplify special cases**: can a special-case branch be absorbed into the general case?

After any change, re-run verification (step 6).

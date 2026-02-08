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

- Reread the JAX docs for the primitive: fetch `https://docs.jax.dev/en/latest/_autosummary/jax.lax.$ARGUMENTS.html`
- Did we test on 1-, 2-, and N-dimensional inputs (in case they are supported)? If not, add such tests.
- Add tests for edge-cases, try to break our implementation and uncover wrong assumptions.
- Update and verify the handler if needed. 

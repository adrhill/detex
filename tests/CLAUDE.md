# Test Suite

## Structure

- Top-level test files (`test_*.py`) cover the public modules in `src/asdex/`.
- `_interpret/` mirrors the handler modules: `_interpret/test_foo.py` tests `src/asdex/_interpret/_foo.py`.
- External-package handler tests live in subfolders (e.g., `_interpret/_equinox/`).

## Running Tests

Always run linting and type checking before tests:

```bash
uv run ruff check --fix .  # lint + auto-fix
uv run ruff format .       # format
uv run ty check            # type check
uv run pytest              # run tests
```

## Markers

Use markers to run subsets of tests:

| Marker | Description |
|--------|-------------|
| `elementwise` | Simple element-wise operations |
| `array_ops` | Array manipulation (slice, concat, reshape) |
| `control_flow` | Conditional operations (where, select) |
| `reduction` | Reduction operations (sum, max, prod) |
| `vmap` | Batched/vmapped operations |
| `coloring` | Row coloring algorithm tests |
| `jacobian` | Sparse Jacobian computation tests |
| `hessian` | Hessian sparsity detection and computation |
| `fallback` | Documents conservative fallback behavior (TODO) |
| `bug` | Documents known bugs |

```bash
uv run pytest -m fallback        # Run only fallback tests
uv run pytest -m "not fallback"  # Skip fallback tests
uv run pytest -m coloring        # Run only coloring tests
uv run pytest -m jacobian        # Run only sparse Jacobian tests
uv run pytest -m hessian         # Run only Hessian tests
```

## Conventions

- Tests documenting **expected future behavior** (TODOs) should use the `fallback` marker and include a `TODO(primitive)` comment explaining the precise expected behavior.
- **Whenever you discover a conservative pattern** (the handler produces a correct but overly dense result), you **must** document it with a `TODO(primitive)` comment showing the true precise pattern.
  Catching these is extremely valuable — each one is a concrete roadmap entry for improving sparsity detection.
- Tests documenting **known bugs** should use the `bug` marker and `pytest.raises` to assert the current (broken) behavior.
- Each test function should have a docstring explaining what it tests.

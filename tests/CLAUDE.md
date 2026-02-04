# Test Suite

## Structure

```
tests/
├── conftest.py                 # Pytest configuration and markers
├── test_jacobian_sparsity.py   # Core API + element-wise operations
├── test_array_ops.py           # Array manipulation (slice, concat, broadcast, etc.)
├── test_control_flow.py        # Conditionals (where, select)
├── test_reductions.py          # Reduction operations (sum, max, argmax)
├── test_vmap.py                # Batched/vmapped operations
├── test_bitset.py              # BitSet unit tests
├── test_benchmarks.py          # Performance benchmarks
├── test_conv.py                # Convolution tests
└── test_sympy.py               # SymPy-based randomized tests
```

## Running Tests

Always run linting and type checking before tests:

```bash
uv run ruff check --fix .
uv run ty check
uv run pytest
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
| `fallback` | Documents conservative fallback behavior (TODO) |
| `bug` | Documents known bugs |

```bash
uv run pytest -m fallback        # Run only fallback tests
uv run pytest -m "not fallback"  # Skip fallback tests
uv run pytest -m vmap            # Run only vmap tests
```

## Conventions

- Tests documenting **expected future behavior** (TODOs) should use the `fallback` marker and include a `TODO(primitive)` comment explaining the precise expected behavior.
- Tests documenting **known bugs** should use the `bug` marker and `pytest.raises` to assert the current (broken) behavior.
- Each test function should have a docstring explaining what it tests.

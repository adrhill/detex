# Test Suite

## Structure

```
tests/
├── conftest.py                 # Pytest configuration and markers
├── test_detection.py           # Sparsity detection tests
├── test_coloring.py            # Row coloring tests
├── test_decompression.py       # Sparse Jacobian computation tests
├── test_vmap.py                # Batched/vmapped operations
├── test_benchmarks.py          # Performance benchmarks
├── test_sympy.py               # SymPy-based randomized tests
├── test_diffrax.py             # Integration tests for diffrax tracing
└── _interpret/                 # Tests for _interpret submodules
    ├── test_slice.py           # Slice operations
    ├── test_squeeze.py         # Squeeze operations
    ├── test_reshape.py         # Reshape operations
    ├── test_broadcast.py       # Broadcast operations
    ├── test_concatenate.py     # Concatenate and stack
    ├── test_gather.py          # Gather (fancy indexing)
    ├── test_scatter.py         # Scatter (at[].set, at[].add, segment_sum)
    ├── test_elementwise.py     # Elementwise operations
    ├── test_dynamic_slice.py   # dynamic_slice, dynamic_update_slice
    ├── test_reduce.py          # reduce_sum, reduce_max, reduce_min, reduce_prod, argmax
    ├── test_top_k.py           # top_k (top-k selection along last axis)
    ├── test_reduce_and.py      # reduce_and, reduce_or, reduce_xor (bitwise reductions)
    ├── test_rev.py             # Rev (reverse/flip) operations
    ├── test_conv.py            # Convolution tests
    ├── test_internals.py       # Internal propagation, fallbacks, custom_call
    ├── test_select.py          # select_n (jnp.where, lax.select)
    ├── test_while.py           # while_loop propagation
    ├── test_cond.py            # cond (conditional branching)
    └── test_nested_jaxpr.py    # const_vals into jit, custom_jvp
```

Each handler module `src/asdex/_interpret/_foo.py` has a corresponding test file `tests/_interpret/test_foo.py`.

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
- Tests documenting **known bugs** should use the `bug` marker and `pytest.raises` to assert the current (broken) behavior.
- Each test function should have a docstring explaining what it tests.

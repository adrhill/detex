# asdex - Automatic Sparse Differentiation in JAX

This package implements [Automatic Sparse Differentiation](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/) (ASD) in JAX.

## Overview

ASD exploits Jacobian sparsity to reduce the cost of computing sparse Jacobians:

1. **Detection**: Analyze the jaxpr computation graph to detect the global sparsity pattern
2. **Coloring**: Assign colors to rows so that rows sharing non-zero columns get different colors
3. **Decompression**: Compute one VJP per color instead of one per row, then extract the sparse Jacobian

## Structure

```
src/asdex/
├── __init__.py         # Public API
├── detection.py        # Sparsity pattern detection via jaxpr analysis
├── coloring.py         # Row-wise graph coloring
├── decompression.py    # Sparse Jacobian computation via VJPs
└── _interpret/         # Primitive handlers for index set propagation
```

The structure of the test folder is described in `tests/CLAUDE.md`.

## Development

```bash
# Lint and format
uv run ruff check --fix .    # lint + auto-fix
uv run ruff format .         # format

# Type check
uv run ty check

# Run tests
uv run pytest
```

## Architecture

```
sparse_jacobian(f, x)
  │
  ├─ 1. DETECTION: jacobian_sparsity(f, n)
  │     ├─ make_jaxpr(f) → computation graph
  │     ├─ Initialize env: input[i] depends on {i}
  │     ├─ prop_jaxpr() → propagate index sets through primitives
  │     └─ Build BCOO sparsity pattern from output dependencies
  │
  ├─ 2. COLORING: color_rows(sparsity)
  │     ├─ Build conflict graph (rows sharing columns)
  │     └─ Greedy coloring → rows with same color are orthogonal
  │
  └─ 3. DECOMPRESSION
        ├─ For each color: VJP with combined seed vector
        └─ Extract J[i,j] = grad[color[i]][j]
```

## Documentation

- Use semantic line breaks in comments and docstrings: start a new line after each sentence or clause boundary, so the source reflects logical structure without affecting rendered output.
- Add brief inline comments for non-obvious parameters and variables.
- Keep descriptions concise without redundant phrases.

## Design philosophy

When writing new code, adhere to these design principles:

- **Minimize complexity**: The primary goal of software design is to minimize complexity—anything that makes a system hard to understand and modify.

- **Information hiding**: Each module should encapsulate design decisions that other modules don't need to know about, preventing information leakage across boundaries.

- **Pull complexity downward**: It's better for a module to be internally complex if it keeps the interface simple for others. Don't expose complexity to callers.

- **Favor exceptions over wrong results**: Raise errors for unknown edge cases rather than guessing. 

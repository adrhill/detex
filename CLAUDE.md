# detex - Automatic Sparse Differentiation in JAX

This package implements [Automatic Sparse Differentiation](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/) (ASD) in JAX, inspired by [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl).

## Overview

ASD exploits Jacobian sparsity to reduce the cost of computing sparse Jacobians:

1. **Detection**: Analyze the jaxpr computation graph to detect the sparsity pattern
2. **Coloring**: Assign colors to rows so that rows sharing non-zero columns get different colors
3. **Decompression**: Compute one VJP per color instead of one per row, then extract the sparse Jacobian

## Structure

```
src/detex/
├── __init__.py         # Public API
├── detection.py        # Sparsity pattern detection via jaxpr analysis
├── coloring.py         # Row-wise graph coloring
├── decompression.py    # Sparse Jacobian computation via VJPs
└── _propagate/         # Primitive handlers for index set propagation

tests/
├── test_detection.py       # Sparsity detection tests
├── test_coloring.py        # Row coloring tests
├── test_decompression.py   # Sparse Jacobian tests
├── test_control_flow.py    # Conditionals (where, select)
├── test_vmap.py            # Batched/vmapped operations
├── test_benchmarks.py      # Performance benchmarks
├── test_sympy.py           # SymPy-based randomized tests
└── _propagate/             # Tests for primitive handlers
```

## Development

```bash
# Install dependencies and pre-commit hooks
uv sync --group dev
uv run pre-commit install

# Run tests
uv run pytest

# Lint and format
uv run ruff check --fix .
uv run ruff format .

# Type check
uv run ty check
```

**Important**: Always run both `ruff` and `ty` after making changes.

## Key Concepts

1. **Global Sparsity**: The sparsity pattern is valid for all inputs. It's detected by analyzing the computation graph structure, not by evaluating derivatives.

2. **Element-wise Tracking**: Dependencies are tracked per-element, not per-variable. This is essential for detecting diagonal patterns like `f(x) = x^2`.

3. **Row Coloring**: Rows that don't share non-zero columns are structurally orthogonal and can be computed together in a single VJP.

## Design Philosophy

- **Minimize complexity**: The primary goal is to minimize complexity, anything that makes a system hard to understand and modify.
- **Information hiding**: Each module should encapsulate design decisions that other modules don't need to know about.
- **Pull complexity downward**: It's better for a module to be internally complex if it keeps the interface simple for others.

# detex - Sparsity Detection Exploration in JAX/Python

This folder contains an exploration of Jacobian sparsity detection implemented in JAX, inspired by [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl).

## Overview

The implementation uses **jaxpr graph analysis** to detect global sparsity patterns. Unlike JVP-based approaches, this analyzes the computation graph structure directly, producing results valid for ALL inputs.

## Structure

```
src/detex/          # Package source
tests/              # pytest tests
jacobian_sparsity_jax.md  # Detailed explanation and theory
```

## Development

```bash
# Install dependencies and pre-commit hooks
uv sync --group dev
uv run pre-commit install

# Run tests
uv run pytest

# Lint and format
uv run ruff check --fix .    # lint + auto-fix
uv run ruff format .         # format

# Type check
uv run ty check
```

**Important**: Always run both `ruff` and `ty` after making changes.

## Key Concepts

1. **Global vs Local Sparsity**: This implements global sparsity (valid for all inputs). Local sparsity would require tracking actual values through control flow.

2. **Element-wise Tracking**: The implementation tracks dependencies per-element, not per-variable. This is essential for detecting diagonal patterns like `f(x) = x^2`.

3. **Primitive Handling**: Each JAX primitive (`slice`, `concatenate`, `add`, etc.) has specific propagation rules for index sets.

## Architecture

```
Input: f(x), n (input dimension)
  |
  v
make_jaxpr(f) -> computation graph
  |
  v
For each primitive equation:
  - Read input dependencies (list[set[int]] per variable)
  - Apply primitive-specific propagation rule
  - Write output dependencies
  |
  v
Extract output dependencies -> sparse COO matrix
```

## Design philosophy

When writing new code, adhere to these design principles:

- **Minimize complexity**: The primary goal of software design is to minimize complexity—anything that makes a system hard to understand and modify.
- **Information hiding**: Each module should encapsulate design decisions that other modules don't need to know about, preventing information leakage across boundaries.
- **Pull complexity downward**: It's better for a module to be internally complex if it keeps the interface simple for others. Don't expose complexity to callers.

## Limitations

- Multi-dimensional slicing is conservative
- Control flow unions all branches (global sparsity)
- Not all JAX primitives have precise handlers (falls back to conservative union)

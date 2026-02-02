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
# Install dependencies
uv sync --group dev

# Run tests
uv run pytest

# Lint and format
uv run ruff check .          # lint
uv run ruff check --fix .    # lint + auto-fix
uv run ruff format .         # format

# Type check
uv run ty check
```

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

## Limitations

- Multi-dimensional slicing is conservative
- Control flow unions all branches (global sparsity)
- Not all JAX primitives have precise handlers (falls back to conservative union)

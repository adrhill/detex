# asdex - Automatic Sparse Differentiation in JAX

This package implements [Automatic Sparse Differentiation](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/) (ASD) in JAX.

## Overview

ASD exploits sparsity to reduce the cost of computing sparse Jacobians and Hessians:

1. **Detection**: Analyze the jaxpr computation graph to detect the global sparsity pattern
2. **Coloring**: Assign colors to rows so that rows sharing non-zero columns get different colors
3. **Decompression**: Compute one VJP/HVP per color instead of one per row, then extract the sparse matrix

## Structure

```
src/asdex/
├── __init__.py         # Public API
├── pattern.py          # SparsityPattern and ColoredPattern data structures
├── detection.py        # Jacobian and Hessian sparsity detection via jaxpr analysis
├── coloring.py         # Graph coloring (row, column, star) and convenience functions
├── decompression.py    # Sparse Jacobian (VJP/JVP) and Hessian (HVP) computation
└── _interpret/         # Custom jaxpr interpreter for index set propagation
```

The interpreter internals are described in `src/asdex/_interpret/CLAUDE.md`.
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
jacobian(f, x)                           hessian(f, x)
  │                                        │
  ├─ 1. DETECTION                          ├─ 1. DETECTION
  │     jacobian_sparsity(f, input_shape)   │     hessian_sparsity(f, input_shape)
  │     ├─ make_jaxpr(f) → jaxpr           │     └─ jacobian_sparsity(grad(f), input_shape)
  │     ├─ prop_jaxpr() → index sets       │
  │     └─ SparsityPattern                 │
  │                                        │
  ├─ 2. COLORING                           ├─ 2. COLORING
  │     color(sparsity) → ColoredPattern   │     star_color(sparsity) → ColoredPattern
  │                                        │
  └─ 3. DECOMPRESSION                      └─ 3. DECOMPRESSION
        One VJP or JVP per color                 One HVP per color (fwd-over-rev)

Convenience: jacobian_coloring(f, shape)   Convenience: hessian_coloring(f, shape)
             = detect + color                            = detect + star_color
```

## Design philosophy

When writing new code, adhere to these design principles:

- **Minimize complexity**: The primary goal of software design is to minimize complexity—anything that makes a system hard to understand and modify.

- **Information hiding**: Each module should encapsulate design decisions that other modules don't need to know about, preventing information leakage across boundaries.

- **Pull complexity downward**: It's better for a module to be internally complex if it keeps the interface simple for others. Don't expose complexity to callers.

- **Favor exceptions over wrong results**: Raise errors for unknown edge cases rather than guessing. 

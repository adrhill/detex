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
├── pattern.py          # SparsityPattern data structure
├── detection.py        # Jacobian and Hessian sparsity detection via jaxpr analysis
├── coloring.py         # Row-wise graph coloring
├── decompression.py    # Sparse Jacobian (VJP) and Hessian (HVP) computation
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
sparse_jacobian(f, x)                    sparse_hessian(f, x)
  │                                        │
  ├─ 1. DETECTION                          ├─ 1. DETECTION
  │     jacobian_sparsity(f, n)            │     hessian_sparsity(f, n)
  │     ├─ make_jaxpr(f) → jaxpr           │     └─ jacobian_sparsity(grad(f), n)
  │     ├─ prop_jaxpr() → index sets       │
  │     └─ Build SparsityPattern           │
  │                                        │
  ├─ 2. COLORING                           ├─ 2. COLORING
  │     color_rows(sparsity)               │     color_rows(sparsity)
  │                                        │
  └─ 3. DECOMPRESSION                      └─ 3. DECOMPRESSION
        One VJP per color                        One HVP per color (fwd-over-rev)
```

## Design philosophy

When writing new code, adhere to these design principles:

- **Minimize complexity**: The primary goal of software design is to minimize complexity—anything that makes a system hard to understand and modify.

- **Information hiding**: Each module should encapsulate design decisions that other modules don't need to know about, preventing information leakage across boundaries.

- **Pull complexity downward**: It's better for a module to be internally complex if it keeps the interface simple for others. Don't expose complexity to callers.

- **Favor exceptions over wrong results**: Raise errors for unknown edge cases rather than guessing. 

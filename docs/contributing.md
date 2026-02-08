# Contributing

## Development Setup

Clone the repository and install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/adrhill/asdex.git
cd asdex
uv sync --group dev
```

## Code Quality

```bash
# Lint and auto-fix
uv run ruff check --fix .

# Format
uv run ruff format .

# Type check
uv run ty check
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test markers
uv run pytest -m jacobian
uv run pytest -m hessian
uv run pytest -m coloring
```

Available markers:
`elementwise`, `array_ops`, `control_flow`, `reduction`,
`vmap`, `coloring`, `jacobian`, `hessian`, `fallback`, `bug`.

## Documentation

```bash
# Install docs dependencies
uv sync --group docs

# Serve locally with live reload
uv run mkdocs serve

# Build (strict mode catches warnings)
uv run mkdocs build --strict
```

## Style Guide

### Code Style

- Follow existing patterns in the codebase
- Google-style docstrings
- Line length: 88 characters (enforced by ruff)

### Semantic Line Breaks

All comments, docstrings, and documentation use **semantic line breaks**:
one sentence or clause per line.

```python
# Good: one sentence per line
# Assigns colors to rows such that no two rows
# sharing a non-zero column have the same color.

# Bad: arbitrary line wrapping
# Assigns colors to rows such that no two
# rows sharing a non-zero column have the
# same color.
```

### Design Philosophy

- **Minimize complexity**: keep the system easy to understand and modify
- **Information hiding**: encapsulate design decisions within modules
- **Pull complexity downward**: keep interfaces simple, even if internals are complex
- **Favor exceptions over wrong results**: raise errors for unknown edge cases

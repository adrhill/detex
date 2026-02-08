# asdex

[![CI](https://github.com/adrhill/asdex/actions/workflows/ci.yml/badge.svg)](https://github.com/adrhill/asdex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/adrhill/asdex/graph/badge.svg)](https://codecov.io/gh/adrhill/asdex)
[![Benchmarks](https://img.shields.io/badge/benchmarks-view-blue)](https://adrianhill.de/asdex/dev/bench/)

**Automatic Sparse Differentiation in JAX.**

`asdex` exploits sparsity structure to efficiently compute sparse Jacobians and Hessians.
It implements a custom [Jaxpr](https://docs.jax.dev/en/latest/jaxpr.html) interpreter
that detects sparsity patterns from the computation graph,
then uses graph coloring to minimize the number of AD passes needed.

!!! warning "Alpha Software"

    `asdex` is in early development.
    The API may change without notice.
    Use at your own risk.

## Installation

```bash
pip install git+https://github.com/adrhill/asdex.git
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/adrhill/asdex.git
```

## Quick Example

```python
import numpy as np
from asdex import jacobian_coloring, jacobian

def f(x):
    return (x[1:] - x[:-1]) ** 2

# Detect sparsity and color in one step:
colored_pattern = jacobian_coloring(f, input_shape=1000)

# Compute sparse Jacobians efficiently:
x = np.random.randn(1000)
J = jacobian(f, x, colored_pattern)
```

Instead of 999 VJPs or 1000 JVPs,
`asdex` computes the full sparse Jacobian with just 2 JVPs.

## Next Steps

- [Getting Started](tutorials/getting-started.md) — step-by-step tutorial
- [How-To Guides](how-to/jacobians.md) — task-oriented recipes
- [Explanation](explanation/coloring.md) — how and why it works
- [API Reference](reference/jacobian.md) — full API documentation

# asdex

[![CI](https://github.com/adrhill/asdex/actions/workflows/ci.yml/badge.svg)](https://github.com/adrhill/asdex/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/adrhill/asdex/graph/badge.svg)](https://codecov.io/gh/adrhill/asdex)
[![PyPI](https://img.shields.io/pypi/v/asdex)](https://pypi.org/project/asdex/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18788242.svg)](https://doi.org/10.5281/zenodo.18788242)

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://adrianhill.de/asdex/)
[![Benchmarks](https://img.shields.io/badge/benchmarks-view-blue)](https://adrianhill.de/asdex/dev/bench/)
[![Changelog](https://img.shields.io/badge/news-changelog-yellow)](https://github.com/adrhill/asdex/blob/main/CHANGELOG.md)


[Automatic Sparse Differentiation](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/) in JAX.

`asdex` (pronounced _Aztecs_) exploits sparsity structure to efficiently compute sparse Jacobians and Hessians.
It implements a custom [Jaxpr](https://docs.jax.dev/en/latest/jaxpr.html) interpreter
that uses [abstract interpretation](https://en.wikipedia.org/wiki/Abstract_interpretation)
to detect sparsity patterns from the computation graph,
then uses graph coloring to minimize the number of AD passes needed.

## Installation

```bash
pip install asdex
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add asdex
```

## Example

```python
import numpy as np
from asdex import jacobian

def f(x):
    return (x[1:] - x[:-1]) ** 2

jac_fn = jacobian(f, input_shape=50)
# ColoredPattern(49×50, nnz=98, sparsity=96.0%, JVP, 2 colors)
#   2 JVPs (instead of 49 VJPs or 50 JVPs)
# ⎡⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎤   ⎡⣿⎤
# ⎢⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥ → ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⎥   ⎢⣿⎥
# ⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⎥   ⎢⣿⎥
# ⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⎦   ⎣⠉⎦

for x in inputs:
    J = jac_fn(x)
```

Instead of 49 VJPs or 50 JVPs,
`asdex` computes the full sparse Jacobian with just 2 JVPs.

## Documentation

- [Getting Started](https://adrianhill.de/asdex/tutorials/getting-started/) — step-by-step tutorial
- [How-To Guides](https://adrianhill.de/asdex/how-to/jacobians/) — task-oriented recipes
- [Explanation](https://adrianhill.de/asdex/explanation/sparsity-detection/) — how and why it works
- [API Reference](https://adrianhill.de/asdex/reference/) — full API documentation

## Acknowledgements

This package is built with Claude Code based on previous work by [Adrian Hill](https://github.com/adrhill), [Guillaume Dalle](https://github.com/gdalle), and [Alexis Montoison](https://github.com/amontoison) in the [Julia programming language](https://julialang.org):

- [_An Illustrated Guide to Automatic Sparse Differentiation_](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/), A. Hill, G. Dalle, A. Montoison (2025)
- [_Sparser, Better, Faster, Stronger: Efficient Automatic Differentiation for Sparse Jacobians and Hessians_](https://openreview.net/forum?id=GtXSN52nIW), A. Hill & G. Dalle (2025)
- [_Revisiting Sparse Matrix Coloring and Bicoloring_](https://arxiv.org/abs/2505.07308), A. Montoison, G. Dalle, A. Gebremedhin (2025)
- [_SparseConnectivityTracer.jl_](https://github.com/adrhill/SparseConnectivityTracer.jl), A. Hill, G. Dalle
- [_SparseMatrixColorings.jl_](https://github.com/gdalle/SparseMatrixColorings.jl), G. Dalle, A. Montoison
- [_sparsediffax_](https://github.com/gdalle/sparsediffax), G. Dalle

which in turn stands on the shoulders of giants — notably Andreas Griewank, Andrea Walther, and Assefaw Gebremedhin.

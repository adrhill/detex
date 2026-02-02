"""Benchmarks for sparsity detection performance."""

import jax.numpy as jnp
import pytest

from detex import jacobian_sparsity

# -----------------------------------------------------------------------------
# Diagonal sparsity (best case: each output depends on one input)
# -----------------------------------------------------------------------------


@pytest.mark.benchmark(group="diagonal")
def test_bench_diagonal_n100(benchmark):
    """Diagonal Jacobian, n=100"""

    def f(x):
        return x**2

    benchmark(jacobian_sparsity, f, 100)


@pytest.mark.benchmark(group="diagonal")
def test_bench_diagonal_n500(benchmark):
    """Diagonal Jacobian, n=500"""

    def f(x):
        return x**2

    benchmark(jacobian_sparsity, f, 500)


@pytest.mark.benchmark(group="diagonal")
def test_bench_diagonal_n1000(benchmark):
    """Diagonal Jacobian, n=1000"""

    def f(x):
        return x**2

    benchmark(jacobian_sparsity, f, 1000)


# -----------------------------------------------------------------------------
# Dense sparsity (worst case: all outputs depend on all inputs)
# -----------------------------------------------------------------------------


@pytest.mark.benchmark(group="dense")
def test_bench_dense_sum_n100(benchmark):
    """Dense via sum, n=100"""

    def f(x):
        return jnp.array([jnp.sum(x)])

    benchmark(jacobian_sparsity, f, 100)


@pytest.mark.benchmark(group="dense")
def test_bench_dense_sum_n500(benchmark):
    """Dense via sum, n=500"""

    def f(x):
        return jnp.array([jnp.sum(x)])

    benchmark(jacobian_sparsity, f, 500)


@pytest.mark.benchmark(group="dense")
def test_bench_dense_matmul_n100(benchmark):
    """Dense via matmul, n=100 -> 50 outputs"""

    def f(x):
        w = jnp.ones((50, 100))
        return w @ x

    benchmark(jacobian_sparsity, f, 100)


# -----------------------------------------------------------------------------
# Realistic workloads
# -----------------------------------------------------------------------------


@pytest.mark.benchmark(group="realistic")
def test_bench_mlp_layer(benchmark):
    """MLP-like: tanh(W @ x), 100 -> 64"""

    def f(x):
        w = jnp.ones((64, 100))
        return jnp.tanh(w @ x)

    benchmark(jacobian_sparsity, f, 100)


@pytest.mark.benchmark(group="realistic")
def test_bench_elementwise_chain(benchmark):
    """Chain of element-wise ops, n=200"""

    def f(x):
        y = x**2
        y = jnp.sin(y)
        y = y + x
        y = jnp.exp(-y)
        return y

    benchmark(jacobian_sparsity, f, 200)


@pytest.mark.benchmark(group="realistic")
def test_bench_mixed_ops(benchmark):
    """Mixed operations: slicing, concat, reduction"""

    def f(x):
        a = x[:50] ** 2
        b = jnp.sin(x[50:])
        c = jnp.concatenate([a, b])
        d = jnp.sum(c[:25])
        return jnp.concatenate([c, jnp.array([d])])

    benchmark(jacobian_sparsity, f, 100)

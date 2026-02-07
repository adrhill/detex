"""Benchmarks for ASD pipeline: detection, coloring, materialization, end-to-end."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import (
    color_rows,
    hessian_sparsity,
    jacobian_sparsity,
    sparse_hessian,
    sparse_jacobian,
)

N = 200  # Problem size for benchmarks

# -----------------------------------------------------------------------------
# Test functions
# -----------------------------------------------------------------------------


# 1. Heat equation RHS (tridiagonal Jacobian, ~3 colors)
def heat_equation_rhs(u):
    """RHS of 1D heat equation with Dirichlet boundaries."""
    left = -2 * u[0:1] + u[1:2]
    interior = u[:-2] - 2 * u[1:-1] + u[2:]
    right = u[-2:-1] - 2 * u[-1:]
    return jnp.concatenate([left, interior, right])


# 2. Pure Conv Network: Conv -> Conv -> Conv with ReLU (sparse Jacobian)
class _ConvNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x[:, None]  # (N,) -> (N, 1)

        x = nn.Conv(features=8, kernel_size=(5,))(x)
        x = nn.relu(x)

        x = nn.Conv(features=4, kernel_size=(3,))(x)
        x = nn.relu(x)

        x = nn.Conv(features=2, kernel_size=(3,))(x)
        x = nn.relu(x)

        return x.flatten()


_convnet_model = _ConvNet()
_convnet_params = _convnet_model.init(jax.random.key(0), jnp.zeros(N))


def convnet(x):
    """Pure ConvNet: 3 conv layers with ReLU (~95% sparse Jacobian)."""
    return _convnet_model.apply(_convnet_params, x)


# 3. Rosenbrock function (sparse Hessian)
def rosenbrock(x):
    """Rosenbrock function for Hessian benchmarks."""
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)


# -----------------------------------------------------------------------------
# Heat Equation benchmarks (Jacobian)
# -----------------------------------------------------------------------------


@pytest.mark.benchmark(group="heat_equation")
def test_heat_detection(benchmark):
    """Heat equation: sparsity detection"""
    benchmark(jacobian_sparsity, heat_equation_rhs, N)


@pytest.mark.benchmark(group="heat_equation")
def test_heat_coloring(benchmark):
    """Heat equation: graph coloring"""
    sparsity = jacobian_sparsity(heat_equation_rhs, N)
    benchmark(color_rows, sparsity)


@pytest.mark.benchmark(group="heat_equation")
def test_heat_materialization(benchmark):
    """Heat equation: VJP computation (with known sparsity/colors)"""
    x = np.ones(N)
    sparsity = jacobian_sparsity(heat_equation_rhs, N)
    colors, _ = color_rows(sparsity)
    benchmark(
        sparse_jacobian,
        heat_equation_rhs,
        x,
        sparsity,
        colors,
        partition="row",
    )


@pytest.mark.benchmark(group="heat_equation")
def test_heat_end_to_end(benchmark):
    """Heat equation: full pipeline"""
    x = np.ones(N)
    benchmark(sparse_jacobian, heat_equation_rhs, x)


# -----------------------------------------------------------------------------
# ConvNet benchmarks (Jacobian)
# -----------------------------------------------------------------------------


@pytest.mark.benchmark(group="convnet")
def test_convnet_detection(benchmark):
    """ConvNet: sparsity detection"""
    benchmark(jacobian_sparsity, convnet, N)


@pytest.mark.benchmark(group="convnet")
def test_convnet_coloring(benchmark):
    """ConvNet: graph coloring"""
    sparsity = jacobian_sparsity(convnet, N)
    benchmark(color_rows, sparsity)


@pytest.mark.benchmark(group="convnet")
def test_convnet_materialization(benchmark):
    """ConvNet: VJP computation (with known sparsity/colors)"""
    x = np.ones(N)
    sparsity = jacobian_sparsity(convnet, N)
    colors, _ = color_rows(sparsity)
    benchmark(sparse_jacobian, convnet, x, sparsity, colors, partition="row")


@pytest.mark.benchmark(group="convnet")
def test_convnet_end_to_end(benchmark):
    """ConvNet: full pipeline"""
    x = np.ones(N)
    benchmark(sparse_jacobian, convnet, x)


# -----------------------------------------------------------------------------
# Rosenbrock benchmarks (Hessian)
# -----------------------------------------------------------------------------


@pytest.mark.benchmark(group="rosenbrock")
def test_rosenbrock_detection(benchmark):
    """Rosenbrock: Hessian sparsity detection"""
    benchmark(hessian_sparsity, rosenbrock, N)


@pytest.mark.benchmark(group="rosenbrock")
def test_rosenbrock_coloring(benchmark):
    """Rosenbrock: graph coloring"""
    sparsity = hessian_sparsity(rosenbrock, N)
    benchmark(color_rows, sparsity)


@pytest.mark.benchmark(group="rosenbrock")
def test_rosenbrock_materialization(benchmark):
    """Rosenbrock: HVP computation (with known sparsity/colors)"""
    x = np.ones(N)
    sparsity = hessian_sparsity(rosenbrock, N)
    colors, _ = color_rows(sparsity)
    benchmark(sparse_hessian, rosenbrock, x, sparsity, colors)


@pytest.mark.benchmark(group="rosenbrock")
def test_rosenbrock_end_to_end(benchmark):
    """Rosenbrock: full pipeline"""
    x = np.ones(N)
    benchmark(sparse_hessian, rosenbrock, x)

"""Benchmarks for ASD pipeline: detection, coloring, materialization, end-to-end."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import (
    hessian,
    hessian_coloring_from_sparsity,
    hessian_from_coloring,
    hessian_sparsity,
    jacobian,
    jacobian_coloring_from_sparsity,
    jacobian_from_coloring,
    jacobian_sparsity,
)
from asdex.coloring import color_rows

N = 200  # Problem size for benchmarks

# Test functions


# 1. Heat equation RHS (tridiagonal Jacobian, ~3 colors)
#    f: R^N -> R^N (Jacobian benchmark)
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
#    f: R^N -> R (Hessian benchmark)
def rosenbrock(x):
    """Rosenbrock function for Hessian benchmarks."""
    return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)


# 4. SchNet-style GNN energy (sparse Hessian)
#    f: R^(P*3) -> R (Hessian benchmark)
#
#    Exercises gather (neighbor lookup), broadcast (cutoff weighting),
#    and scatter-add (gradient of gather) — the patterns that are pathological
#    for sparsity detection in graph neural networks.

_GNN_N_ATOMS = 10
_GNN_MAX_NEIGHBORS = 5
_GNN_HIDDEN = 8


def _build_chain_graph(
    n_atoms: int, cutoff: float, max_neighbors: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a padded neighborlist for a 1D chain of atoms.

    Returns (centers, others, pair_mask, atom_mask) as numpy arrays.
    """
    positions = np.arange(n_atoms, dtype=np.float64)
    centers = []
    others = []
    pair_mask = []
    for i in range(n_atoms):
        count = 0
        for j in range(n_atoms):
            if i != j and abs(positions[i] - positions[j]) <= cutoff:
                centers.append(i)
                others.append(j)
                pair_mask.append(True)
                count += 1
        for _ in range(max_neighbors - count):
            centers.append(i)
            others.append(0)
            pair_mask.append(False)
    return (
        np.array(centers),
        np.array(others),
        np.array(pair_mask),
        np.ones(n_atoms, dtype=bool),
    )


_gnn_centers, _gnn_others, _gnn_pair_mask, _gnn_atom_mask = _build_chain_graph(
    _GNN_N_ATOMS, cutoff=2.5, max_neighbors=_GNN_MAX_NEIGHBORS
)
_gnn_n_pairs = len(_gnn_centers)
_gnn_input_shape = (_gnn_n_pairs, 3)

# Closure-captured constants (become jaxpr consts for precise gather/scatter).
_gnn_others_jax = jnp.array(_gnn_others)
_gnn_pair_mask_jax = jnp.array(_gnn_pair_mask, dtype=jnp.float32)
_gnn_atom_mask_jax = jnp.array(_gnn_atom_mask, dtype=jnp.float32)

# Fixed weights.
_gnn_W1 = jax.random.normal(jax.random.key(42), (_GNN_HIDDEN,))
_gnn_W2 = jax.random.normal(jax.random.key(43), (4, _GNN_HIDDEN))
_gnn_W3 = jax.random.normal(jax.random.key(44), (_GNN_HIDDEN, 1))


def gnn_energy(R_ij):
    """SchNet-style GNN energy: radial filter, neighbor gather, aggregate."""
    r = jnp.sqrt(jnp.sum(R_ij**2, axis=-1) + 1e-20)  # [P]
    cutoffs = jnp.exp(-(r**2)) * _gnn_pair_mask_jax  # [P]

    # Radial filter: distance → per-pair features.
    radial = jax.nn.silu(jnp.outer(r, _gnn_W1))  # [P, H]

    # Atom embedding from pair displacements (concat R_ij + r, project).
    geo = jnp.concatenate([R_ij, r[:, None]], axis=-1)  # [P, 4]
    atom_features = jax.nn.silu(geo @ _gnn_W2)  # [P, H]

    # Neighbor gather (closure-captured static indices).
    gathered = atom_features[_gnn_others_jax]  # [P, H]

    # Continuous filter: element-wise product with cutoff weighting (broadcast).
    filtered = radial * gathered * cutoffs[:, None]  # [P, H]

    # Per-atom aggregation (reshape + reduce).
    per_atom = filtered.reshape(_GNN_N_ATOMS, _GNN_MAX_NEIGHBORS, _GNN_HIDDEN)
    atom_out = jnp.sum(per_atom, axis=1)  # [N, H]

    # Scalar energy readout.
    energies = (atom_out @ _gnn_W3).squeeze(-1) * _gnn_atom_mask_jax  # [N]
    return jnp.sum(energies)


# Heat Equation benchmarks (Jacobian)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="heat_equation")
def test_heat_detection(benchmark):
    """Heat equation: sparsity detection."""
    benchmark(jacobian_sparsity, heat_equation_rhs, N)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="heat_equation")
def test_heat_coloring(benchmark):
    """Heat equation: graph coloring."""
    sparsity = jacobian_sparsity(heat_equation_rhs, N)
    benchmark(color_rows, sparsity)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="heat_equation")
def test_heat_materialization(benchmark):
    """Heat equation: VJP computation (with known sparsity/colors)."""
    x = np.ones(N)
    coloring = jacobian_coloring_from_sparsity(
        jacobian_sparsity(heat_equation_rhs, N), mode="rev"
    )
    jac_fn = jacobian_from_coloring(heat_equation_rhs, coloring)
    benchmark(jac_fn, x)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="heat_equation")
def test_heat_end_to_end(benchmark):
    """Heat equation: full pipeline."""
    x = np.ones(N)
    jac_fn = jacobian(heat_equation_rhs, N)
    benchmark(jac_fn, x)


# ConvNet benchmarks (Jacobian)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="convnet")
def test_convnet_detection(benchmark):
    """ConvNet: sparsity detection."""
    benchmark(jacobian_sparsity, convnet, N)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="convnet")
def test_convnet_coloring(benchmark):
    """ConvNet: graph coloring."""
    sparsity = jacobian_sparsity(convnet, N)
    benchmark(color_rows, sparsity)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="convnet")
def test_convnet_materialization(benchmark):
    """ConvNet: VJP computation (with known sparsity/colors)."""
    x = np.ones(N)
    coloring = jacobian_coloring_from_sparsity(
        jacobian_sparsity(convnet, N), mode="rev"
    )
    jac_fn = jacobian_from_coloring(convnet, coloring)
    benchmark(jac_fn, x)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="convnet")
def test_convnet_end_to_end(benchmark):
    """ConvNet: full pipeline."""
    x = np.ones(N)
    jac_fn = jacobian(convnet, N)
    benchmark(jac_fn, x)


# Rosenbrock benchmarks (Hessian)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="rosenbrock")
def test_rosenbrock_detection(benchmark):
    """Rosenbrock: Hessian sparsity detection."""
    benchmark(hessian_sparsity, rosenbrock, N)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="rosenbrock")
def test_rosenbrock_coloring(benchmark):
    """Rosenbrock: graph coloring."""
    sparsity = hessian_sparsity(rosenbrock, N)
    benchmark(color_rows, sparsity)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="rosenbrock")
def test_rosenbrock_materialization(benchmark):
    """Rosenbrock: HVP computation (with known sparsity/colors)."""
    x = np.ones(N)
    sparsity = hessian_sparsity(rosenbrock, N)
    coloring = hessian_coloring_from_sparsity(sparsity)
    hess_fn = hessian_from_coloring(rosenbrock, coloring)
    benchmark(hess_fn, x)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="rosenbrock")
def test_rosenbrock_end_to_end(benchmark):
    """Rosenbrock: full pipeline."""
    x = np.ones(N)
    hess_fn = hessian(rosenbrock, N)
    benchmark(hess_fn, x)


# GNN benchmarks (Hessian)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="gnn")
def test_gnn_detection(benchmark):
    """GNN: Hessian sparsity detection."""
    benchmark(hessian_sparsity, gnn_energy, _gnn_input_shape)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="gnn")
def test_gnn_coloring(benchmark):
    """GNN: graph coloring."""
    sparsity = hessian_sparsity(gnn_energy, _gnn_input_shape)
    benchmark(color_rows, sparsity)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="gnn")
def test_gnn_materialization(benchmark):
    """GNN: HVP computation (with known sparsity/colors)."""
    x = np.ones(_gnn_input_shape)
    sparsity = hessian_sparsity(gnn_energy, _gnn_input_shape)
    coloring = hessian_coloring_from_sparsity(sparsity)
    hess_fn = hessian_from_coloring(gnn_energy, coloring)
    benchmark(hess_fn, x)


@pytest.mark.dashboard
@pytest.mark.benchmark(group="gnn")
def test_gnn_end_to_end(benchmark):
    """GNN: full pipeline."""
    x = np.ones(_gnn_input_shape)
    hess_fn = hessian(gnn_energy, _gnn_input_shape)
    benchmark(hess_fn, x)

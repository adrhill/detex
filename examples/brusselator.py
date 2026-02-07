"""Brusselator reaction-diffusion: sparsity demo with asdex + diffrax.

Detects the Jacobian sparsity pattern of the brusselator RHS,
then attempts to trace through a diffrax Euler step.
The RHS detection should succeed;
the diffrax step will likely fail on unsupported primitives.
"""

import traceback

import diffrax
import jax.numpy as jnp

import asdex

# --- Brusselator parameters ---
N = 8  # grid size: 2 species on N×N grid → 2*N*N state variables
A = 1.0
B = 1.0
ALPHA = 1.0  # diffusion coefficient
DX = 1.0


def brusselator_rhs(y: jnp.ndarray) -> jnp.ndarray:
    """Brusselator RHS with periodic BCs via 5-point Laplacian.

    State vector ``y`` has size ``2*N*N``:
    first ``N*N`` entries are species u,
    last ``N*N`` entries are species v.
    """
    n2 = N * N
    u = y[:n2].reshape(N, N)
    v = y[n2:].reshape(N, N)

    # 5-point finite difference Laplacian with periodic BCs
    lap_u = (
        jnp.roll(u, 1, axis=0)
        + jnp.roll(u, -1, axis=0)
        + jnp.roll(u, 1, axis=1)
        + jnp.roll(u, -1, axis=1)
        - 4 * u
    ) / (DX * DX)

    lap_v = (
        jnp.roll(v, 1, axis=0)
        + jnp.roll(v, -1, axis=0)
        + jnp.roll(v, 1, axis=1)
        + jnp.roll(v, -1, axis=1)
        - 4 * v
    ) / (DX * DX)

    # Reaction-diffusion
    du = ALPHA * lap_u + B + u**2 * v - (A + 1) * u
    dv = ALPHA * lap_v + A * u - u**2 * v

    return jnp.concatenate([du.ravel(), dv.ravel()])


# --- Diffrax Euler step wrapper ---
solver = diffrax.Euler()
term = diffrax.ODETerm(lambda t, y, args: brusselator_rhs(y))
stepsize = diffrax.ConstantStepSize()


def euler_step(y0: jnp.ndarray) -> jnp.ndarray:
    """One Euler step from t=0 to t=0.1."""
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=0.1,
        dt0=0.1,
        y0=y0,
        stepsize_controller=stepsize,
        max_steps=1,
    )
    return sol.ys[0]


# --- Main ---
if __name__ == "__main__":
    n = 2 * N * N  # total state dimension

    # 1. Brusselator RHS sparsity
    print(f"Brusselator RHS: {n} → {n}")
    pattern = asdex.jacobian_sparsity(brusselator_rhs, n)
    print(pattern)
    print()

    colored = asdex.color_jacobian_pattern(pattern)
    print(colored)
    print()

    # 2. Diffrax Euler step sparsity
    print(f"Diffrax Euler step: {n} → {n}")
    try:
        step_pattern = asdex.jacobian_sparsity(euler_step, n)
        print(step_pattern)
    except Exception:
        print("Failed to trace diffrax Euler step:")
        traceback.print_exc()

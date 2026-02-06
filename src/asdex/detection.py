"""Jacobian and Hessian sparsity detection via jaxpr graph analysis."""

import jax
import jax.numpy as jnp

from asdex._interpret import prop_jaxpr
from asdex.pattern import SparsityPattern


def jacobian_sparsity(f, n: int) -> SparsityPattern:
    """
    Detect global Jacobian sparsity pattern for f: R^n -> R^m.

    This analyzes the computation graph structure directly, without evaluating any derivatives.
    The result is valid for all inputs.

    The approach:
    1. Get the jaxpr (computation graph) for the function
    2. Propagate per-element index sets through each primitive
    3. Build the sparsity pattern from output dependencies

    Args:
        f: Function taking a 1D array of length n
        n: Input dimension

    Returns:
        SparsityPattern of shape (m, n) where entry (i,j) is present
        if output i depends on input j
    """
    dummy_input = jnp.zeros(n)
    closed_jaxpr = jax.make_jaxpr(f)(dummy_input)
    jaxpr = closed_jaxpr.jaxpr
    m = int(jax.eval_shape(f, dummy_input).size)

    # Initialize: input element i depends on input index i
    input_indices = [[{i} for i in range(n)]]

    # Propagate through the jaxpr
    output_indices_list = prop_jaxpr(jaxpr, input_indices)

    # Extract output dependencies (first output variable)
    out_indices = output_indices_list[0] if output_indices_list else []

    # Build sparsity pattern
    rows = []
    cols = []
    for i, deps in enumerate(out_indices):
        for j in deps:
            rows.append(i)
            cols.append(j)

    return SparsityPattern.from_coordinates(rows, cols, (m, n))


def hessian_sparsity(f, n: int) -> SparsityPattern:
    """
    Detect global Hessian sparsity pattern for f: R^n -> R.

    Computed by analyzing the Jacobian sparsity of the gradient function,
    demonstrating how our sparsity interpreter composes with JAX's autodiff.

    Args:
        f: Scalar-valued function taking a 1D array of length n
        n: Input dimension

    Returns:
        SparsityPattern of shape (n, n) where entry (i,j) is present
        if H[i,j] may be nonzero
    """
    return jacobian_sparsity(jax.grad(f), n)

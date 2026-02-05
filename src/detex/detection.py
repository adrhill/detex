"""Jacobian sparsity detection via jaxpr graph analysis."""

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from detex._propagate import prop_jaxpr


def jacobian_sparsity(f, n: int) -> BCOO:
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
        BCOO sparse boolean matrix of shape (m, n) where entry (i,j) is True
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

    # Build sparse matrix
    rows = []
    cols = []
    for i, deps in enumerate(out_indices):
        for j in deps:
            rows.append(i)
            cols.append(j)

    indices = jnp.array(
        [[r, c] for r, c in zip(rows, cols, strict=True)], dtype=jnp.int32
    )
    if len(rows) == 0:
        indices = jnp.zeros((0, 2), dtype=jnp.int32)
    data = jnp.ones(len(rows), dtype=jnp.int8)

    return BCOO((data, indices), shape=(m, n))

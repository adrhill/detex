"""Jacobian and Hessian sparsity detection via jaxpr graph analysis."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from asdex._interpret import prop_jaxpr
from asdex.pattern import SparsityPattern


def jacobian_sparsity(
    f: Callable, input_shape: int | tuple[int, ...]
) -> SparsityPattern:
    """Detect global Jacobian sparsity pattern for f: R^n -> R^m.

    Analyzes the computation graph structure directly,
    without evaluating any derivatives.
    The result is valid for all inputs.

    Args:
        f: Function taking an array and returning an array.
        input_shape: Shape of the input array.
            An integer is treated as a 1D length.

    Returns:
        SparsityPattern of shape ``(m, n)``
            where ``n = prod(input_shape)`` and ``m = prod(output_shape)``.
            Entry ``(i, j)`` is present if output ``i`` depends on input ``j``.
    """
    dummy_input = jnp.zeros(input_shape)
    closed_jaxpr = jax.make_jaxpr(f)(dummy_input)
    jaxpr = closed_jaxpr.jaxpr
    m = int(jax.eval_shape(f, dummy_input).size)
    n = input_shape if isinstance(input_shape, int) else math.prod(input_shape)

    # Initialize: input element i depends on input index i
    input_indices = [[{i} for i in range(n)]]

    # Build const_vals from closed jaxpr consts for static index tracking
    const_vals = {
        var: np.asarray(val)
        for var, val in zip(jaxpr.constvars, closed_jaxpr.consts, strict=False)
    }

    # Propagate through the jaxpr
    output_indices_list = prop_jaxpr(jaxpr, input_indices, const_vals)

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


def hessian_sparsity(
    f: Callable, input_shape: int | tuple[int, ...]
) -> SparsityPattern:
    """Detect global Hessian sparsity pattern for f: R^n -> R.

    Analyzes the Jacobian sparsity of the gradient function,
    without evaluating any derivatives.
    The result is valid for all inputs.

    Args:
        f: Scalar-valued function taking an array.
        input_shape: Shape of the input array.
            An integer is treated as a 1D length.

    Returns:
        SparsityPattern of shape ``(n, n)``
            where ``n = prod(input_shape)``.
            Entry ``(i, j)`` is present if ``H[i, j]`` may be nonzero.
    """
    return jacobian_sparsity(jax.grad(f), input_shape)

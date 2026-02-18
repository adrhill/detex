"""Verification utilities for checking asdex results against JAX references."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from asdex.coloring import hessian_coloring, jacobian_coloring
from asdex.decompression import hessian, jacobian
from asdex.pattern import ColoredPattern


class VerificationError(AssertionError):
    """Raised when asdex's sparse result does not match JAX's dense reference.

    This indicates that the detected sparsity pattern is missing nonzeros,
    which is a bug — asdex's patterns should always be conservative
    (i.e., contain at least all true nonzeros).
    If you encounter this error,
    please help out asdex's development by reporting this at
    https://github.com/adrhill/asdex/issues.
    """


def check_jacobian_correctness(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    *,
    colored_pattern: ColoredPattern | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-7,
) -> None:
    """Verify asdex's sparse Jacobian against ``jax.jacobian`` at a given input.

    Computes the sparse Jacobian using asdex and the dense Jacobian using JAX,
    then checks that all entries match within the given tolerances.

    Args:
        f: Function taking an array and returning an array.
        x: Input at which to evaluate the Jacobian.
        colored_pattern: Optional pre-computed colored pattern.
            If None, sparsity is detected and colored automatically.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Raises:
        VerificationError: If the sparse and dense Jacobians disagree.
    """
    x = jnp.asarray(x)

    if colored_pattern is None:
        colored_pattern = jacobian_coloring(f, input_shape=x.shape)

    J_sparse = jacobian(f, colored_pattern)(x).todense()
    J_dense = jax.jacobian(f)(x)

    _check_allclose(J_sparse, J_dense, "Jacobian", rtol=rtol, atol=atol)


def check_hessian_correctness(
    f: Callable[[ArrayLike], ArrayLike],
    x: ArrayLike,
    *,
    colored_pattern: ColoredPattern | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-7,
) -> None:
    """Verify asdex's sparse Hessian against ``jax.hessian`` at a given input.

    Computes the sparse Hessian using asdex and the dense Hessian using JAX,
    then checks that all entries match within the given tolerances.

    Args:
        f: Scalar-valued function taking an array.
        x: Input at which to evaluate the Hessian.
        colored_pattern: Optional pre-computed colored pattern.
            If None, sparsity is detected and colored automatically.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Raises:
        VerificationError: If the sparse and dense Hessians disagree.
    """
    x = jnp.asarray(x)

    if colored_pattern is None:
        colored_pattern = hessian_coloring(f, input_shape=x.shape)

    H_sparse = hessian(f, colored_pattern)(x).todense()
    H_dense = jax.hessian(f)(x)

    _check_allclose(H_sparse, H_dense, "Hessian", rtol=rtol, atol=atol)


def _check_allclose(
    sparse: jax.Array,
    dense: jax.Array,
    name: str,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Compare sparse and dense results, raising VerificationError on mismatch."""
    sparse_np = np.asarray(sparse)
    dense_np = np.asarray(dense)

    if sparse_np.shape != dense_np.shape:
        raise VerificationError(
            f"asdex's sparse {name} has shape {sparse_np.shape} "
            f"but JAX's dense reference has shape {dense_np.shape}. "
            "This likely means the detected sparsity pattern is missing nonzeros. "
            "Please help out asdex's development by reporting this at "
            "https://github.com/adrhill/asdex/issues"
        )

    try:
        np.testing.assert_allclose(sparse_np, dense_np, rtol=rtol, atol=atol)
    except AssertionError:
        raise VerificationError(
            f"asdex's sparse {name} does not match JAX's dense reference. "
            "This likely means the detected sparsity pattern is missing nonzeros. "
            "Please help out asdex's development by reporting this at "
            "https://github.com/adrhill/asdex/issues"
        ) from None

"""asdex - Global Jacobian and Hessian sparsity detection via jaxpr graph analysis.

This is global sparsity detection - asdex analyzes the computation graph
structure without evaluating derivatives, so results are valid for all inputs.
"""

from asdex.coloring import (
    color_cols,
    color_hessian_pattern,
    color_jacobian_pattern,
    color_rows,
    color_symmetric,
    hessian_coloring,
    jacobian_coloring,
)
from asdex.decompression import hessian, jacobian
from asdex.detection import hessian_sparsity, jacobian_sparsity
from asdex.pattern import ColoredPattern, SparsityPattern
from asdex.verify import (
    VerificationError,
    check_hessian_correctness,
    check_jacobian_correctness,
)

__all__ = [
    "ColoredPattern",
    "SparsityPattern",
    "VerificationError",
    "check_hessian_correctness",
    "check_jacobian_correctness",
    "color_cols",
    "color_hessian_pattern",
    "color_jacobian_pattern",
    "color_rows",
    "color_symmetric",
    "hessian",
    "hessian_coloring",
    "hessian_sparsity",
    "jacobian",
    "jacobian_coloring",
    "jacobian_sparsity",
]

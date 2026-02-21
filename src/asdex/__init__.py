"""asdex - Global Jacobian and Hessian sparsity detection via jaxpr graph analysis.

This is global sparsity detection - asdex analyzes the computation graph
structure without evaluating derivatives, so results are valid for all inputs.
"""

from asdex.coloring import (
    DenseColoringWarning,
    color_hessian_pattern,
    color_jacobian_pattern,
    hessian_coloring,
    jacobian_coloring,
)
from asdex.decompression import (
    hessian,
    hessian_from_coloring,
    jacobian,
    jacobian_from_coloring,
)
from asdex.detection import hessian_sparsity, jacobian_sparsity
from asdex.modes import HessianMode, JacobianMode
from asdex.pattern import ColoredPattern, SparsityPattern
from asdex.verify import (
    VerificationError,
    check_hessian_correctness,
    check_jacobian_correctness,
)

__all__ = [
    "ColoredPattern",
    "DenseColoringWarning",
    "HessianMode",
    "JacobianMode",
    "SparsityPattern",
    "VerificationError",
    "check_hessian_correctness",
    "check_jacobian_correctness",
    "color_hessian_pattern",
    "color_jacobian_pattern",
    "hessian",
    "hessian_coloring",
    "hessian_from_coloring",
    "hessian_sparsity",
    "jacobian",
    "jacobian_coloring",
    "jacobian_from_coloring",
    "jacobian_sparsity",
]

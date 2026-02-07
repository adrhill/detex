"""
asdex - Global Jacobian and Hessian sparsity detection via jaxpr graph analysis.

This is global sparsity detection - asdex analyzes the computation graph
structure without evaluating derivatives, so results are valid for all inputs.
"""

from asdex.coloring import (
    color,
    color_cols,
    color_rows,
    hessian_coloring,
    jacobian_coloring,
    star_color,
)
from asdex.decompression import hessian, jacobian
from asdex.detection import hessian_sparsity, jacobian_sparsity
from asdex.pattern import ColoredPattern, SparsityPattern

__all__ = [
    "jacobian_sparsity",
    "hessian_sparsity",
    "color",
    "color_rows",
    "color_cols",
    "star_color",
    "jacobian_coloring",
    "hessian_coloring",
    "ColoredPattern",
    "jacobian",
    "hessian",
    "SparsityPattern",
]

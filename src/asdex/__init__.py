"""
asdex - Global Jacobian and Hessian sparsity detection via jaxpr graph analysis.

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

__all__ = [
    # End-to-end: detect + color + decompress
    "jacobian",
    "hessian",
    # Convenience: detect + color
    "jacobian_coloring",
    "hessian_coloring",
    # Detection
    "jacobian_sparsity",
    "hessian_sparsity",
    # Coloring
    "color_jacobian_pattern",
    "color_hessian_pattern",
    "color_rows",
    "color_cols",
    "color_symmetric",
    # Data structures
    "SparsityPattern",
    "ColoredPattern",
]

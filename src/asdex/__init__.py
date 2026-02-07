"""
asdex - Global Jacobian and Hessian sparsity detection via jaxpr graph analysis.

This is global sparsity detection - asdex analyzes the computation graph
structure without evaluating derivatives, so results are valid for all inputs.
"""

from asdex.coloring import color_cols, color_rows, star_color
from asdex.decompression import sparse_hessian, sparse_jacobian
from asdex.detection import hessian_sparsity, jacobian_sparsity
from asdex.pattern import SparsityPattern

__all__ = [
    "jacobian_sparsity",
    "hessian_sparsity",
    "color_rows",
    "color_cols",
    "star_color",
    "sparse_jacobian",
    "sparse_hessian",
    "SparsityPattern",
]

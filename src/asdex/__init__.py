"""
asdex - Global Jacobian sparsity detection via jaxpr graph analysis.

This is global sparsity detection - asdex analyzes the computation graph
structure without evaluating derivatives, so results are valid for all inputs.
"""

from asdex.coloring import color_rows
from asdex.decompression import sparse_jacobian
from asdex.detection import jacobian_sparsity

__all__ = ["jacobian_sparsity", "color_rows", "sparse_jacobian"]

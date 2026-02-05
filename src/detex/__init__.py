"""
detex - Global Jacobian sparsity detection via jaxpr graph analysis.

This is global sparsity detection - detex analyzes the computation graph
structure without evaluating derivatives, so results are valid for all inputs.
"""

from detex.coloring import color_rows
from detex.decompression import sparse_jacobian
from detex.detection import jacobian_sparsity

__all__ = ["jacobian_sparsity", "color_rows", "sparse_jacobian"]

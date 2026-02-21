"""Public type aliases for AD mode selection."""

from typing import Literal

ColoringMode = Literal["row", "column", "symmetric", "auto"]
"""Coloring mode for sparse differentiation.

``"row"`` for row coloring (compatible with VJPs or HVPs),
``"column"`` for column coloring (compatible with JVPs or HVPs),
``"symmetric"`` for symmetric (star) coloring (compatible with JVPs, VJPs, or HVPs),
``"auto"`` selects automatically.
"""

JacobianMode = Literal["fwd", "rev", "auto"]
"""AD mode for Jacobian computation.

``"fwd"`` uses JVPs (forward-mode AD),
``"rev"`` uses VJPs (reverse-mode AD),
``"auto"`` selects automatically based on the coloring mode.
"""

HessianMode = Literal["fwd_over_rev", "rev_over_fwd", "rev_over_rev", "auto"]
"""AD composition strategy for Hessian-vector products.

``"fwd_over_rev"`` uses forward-over-reverse,
``"rev_over_fwd"`` uses reverse-over-forward,
``"rev_over_rev"`` uses reverse-over-reverse,
``"auto"`` defaults to ``"fwd_over_rev"``.
"""


def resolve_ad_mode(coloring_mode: ColoringMode, ad_mode: JacobianMode) -> JacobianMode:
    """Resolve the AD mode given a coloring mode.

    Validates compatibility and resolves ``"auto"``.

    Args:
        coloring_mode: The resolved coloring mode (must not be ``"auto"``).
        ad_mode: The requested AD mode.

    Returns:
        A concrete AD mode (``"fwd"`` or ``"rev"``).

    Raises:
        ValueError: If the combination is incompatible
            (e.g. ``"row"`` + ``"fwd"``).
    """
    if ad_mode == "auto":
        if coloring_mode == "row":
            return "rev"
        if coloring_mode == "column":
            return "fwd"
        # symmetric: default to fwd
        return "fwd"

    if coloring_mode == "row" and ad_mode != "rev":
        raise ValueError(
            f"Row coloring is only compatible with ad_mode='rev', got {ad_mode!r}."
        )
    if coloring_mode == "column" and ad_mode != "fwd":
        raise ValueError(
            f"Column coloring is only compatible with ad_mode='fwd', got {ad_mode!r}."
        )
    # symmetric is compatible with both fwd and rev
    return ad_mode


def resolve_hessian_mode(ad_mode: HessianMode) -> HessianMode:
    """Resolve ``"auto"`` Hessian mode to ``"fwd_over_rev"``.

    Args:
        ad_mode: The requested Hessian AD mode.

    Returns:
        A concrete Hessian AD mode.
    """
    if ad_mode == "auto":
        return "fwd_over_rev"
    return ad_mode

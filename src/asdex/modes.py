"""Public type aliases and validation for AD mode selection."""

from typing import Literal, get_args

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


def assert_coloring_mode(coloring_mode: str) -> None:
    """Raise ``ValueError`` if *coloring_mode* is not a valid ``ColoringMode``."""
    if coloring_mode not in get_args(ColoringMode):
        raise ValueError(
            f"Unknown coloring_mode {coloring_mode!r}. "
            "Expected 'row', 'column', 'symmetric', or 'auto'."
        )


def assert_jacobian_mode(ad_mode: str) -> None:
    """Raise ``ValueError`` if *ad_mode* is not a valid ``JacobianMode``."""
    if ad_mode not in get_args(JacobianMode):
        raise ValueError(
            f"Unknown ad_mode {ad_mode!r}. Expected 'fwd', 'rev', or 'auto'."
        )


def assert_hessian_mode(ad_mode: str) -> None:
    """Raise ``ValueError`` if *ad_mode* is not a valid ``HessianMode``."""
    if ad_mode not in get_args(HessianMode):
        raise ValueError(
            f"Unknown ad_mode {ad_mode!r}. "
            "Expected 'fwd_over_rev', 'rev_over_fwd', 'rev_over_rev', or 'auto'."
        )


def resolve_coloring_mode(
    coloring_mode: ColoringMode, ad_mode: JacobianMode
) -> ColoringMode:
    """Resolve ``"auto"`` coloring mode from an AD mode hint.

    When ``coloring_mode`` is ``"auto"`` and ``ad_mode`` is specific,
    resolves to the natural coloring:
    ``"fwd"`` implies ``"column"``, ``"rev"`` implies ``"row"``.

    When both are ``"auto"`` or ``coloring_mode`` is already specific,
    returns ``coloring_mode`` unchanged.

    Args:
        coloring_mode: The requested coloring mode.
        ad_mode: AD mode hint.

    Returns:
        A coloring mode (possibly still ``"auto"``).

    Raises:
        ValueError: If either input is unknown.
    """
    assert_coloring_mode(coloring_mode)
    assert_jacobian_mode(ad_mode)
    if coloring_mode == "auto" and ad_mode != "auto":
        return "row" if ad_mode == "rev" else "column"
    return coloring_mode


def resolve_ad_mode(coloring_mode: ColoringMode, ad_mode: JacobianMode) -> JacobianMode:
    """Resolve the AD mode given a coloring mode.

    Validates inputs, checks compatibility, and resolves ``"auto"``.

    Args:
        coloring_mode: The resolved coloring mode (must not be ``"auto"``).
        ad_mode: The requested AD mode.

    Returns:
        A concrete AD mode (``"fwd"`` or ``"rev"``).

    Raises:
        ValueError: If ``coloring_mode`` is ``"auto"``,
            if either input is unknown,
            or if the combination is incompatible
            (e.g. ``"row"`` + ``"fwd"``).
    """
    assert_coloring_mode(coloring_mode)
    assert_jacobian_mode(ad_mode)

    if coloring_mode == "auto":
        raise ValueError(
            "coloring_mode must be resolved before calling resolve_ad_mode, "
            f"got {coloring_mode!r}."
        )

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

    Raises:
        ValueError: If ``ad_mode`` is unknown.
    """
    assert_hessian_mode(ad_mode)
    if ad_mode == "auto":
        return "fwd_over_rev"
    return ad_mode

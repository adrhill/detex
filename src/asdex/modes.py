"""Type aliases and validation for AD mode selection."""

import warnings
from typing import Literal, assert_never, get_args

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


def _assert_coloring_mode(coloring_mode: str) -> None:
    """Raise ``ValueError`` if *coloring_mode* is not a valid ``ColoringMode``."""
    if coloring_mode not in get_args(ColoringMode):
        raise ValueError(
            f"Unknown coloring_mode {coloring_mode!r}. "
            "Expected 'row', 'column', 'symmetric', or 'auto'."
        )


def _assert_jacobian_mode(ad_mode: str) -> None:
    """Raise ``ValueError`` if *ad_mode* is not a valid ``JacobianMode``."""
    if ad_mode not in get_args(JacobianMode):
        raise ValueError(
            f"Unknown ad_mode {ad_mode!r}. Expected 'fwd', 'rev', or 'auto'."
        )


def _assert_hessian_mode(ad_mode: str) -> None:
    """Raise ``ValueError`` if *ad_mode* is not a valid ``HessianMode``."""
    if ad_mode not in get_args(HessianMode):
        raise ValueError(
            f"Unknown ad_mode {ad_mode!r}. "
            "Expected 'fwd_over_rev', 'rev_over_fwd', 'rev_over_rev', or 'auto'."
        )


def _assert_jacobian_args(
    colored_pattern: object | None,
    coloring_mode: ColoringMode,
    ad_mode: JacobianMode,
) -> None:
    """Validate and warn for Jacobian input arguments.

    Asserts that ``coloring_mode`` and ``ad_mode`` are valid,
    and warns if ``coloring_mode`` is set but will be ignored
    because a pre-computed pattern was provided.
    """
    _assert_coloring_mode(coloring_mode)
    _assert_jacobian_mode(ad_mode)
    if colored_pattern is not None and coloring_mode != "auto":
        warnings.warn(
            "coloring_mode is ignored when colored_pattern is provided.",
            stacklevel=3,
        )


def _assert_hessian_args(
    colored_pattern: object | None,
    coloring_mode: ColoringMode,
    ad_mode: HessianMode,
) -> None:
    """Validate and warn for Hessian input arguments.

    Asserts that ``coloring_mode`` and ``ad_mode`` are valid,
    and warns if ``coloring_mode`` is set but will be ignored
    because a pre-computed pattern was provided.
    """
    _assert_coloring_mode(coloring_mode)
    _assert_hessian_mode(ad_mode)
    if colored_pattern is not None and coloring_mode != "auto":
        warnings.warn(
            "coloring_mode is ignored when colored_pattern is provided.",
            stacklevel=3,
        )


def _resolve_coloring_mode(
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
    _assert_coloring_mode(coloring_mode)
    _assert_jacobian_mode(ad_mode)
    if coloring_mode != "auto":
        return coloring_mode
    match ad_mode:
        case "rev":
            return "row"
        case "fwd":
            return "column"
        case "auto":
            return coloring_mode
        case _ as unreachable:
            assert_never(unreachable)


def _resolve_ad_mode(
    coloring_mode: ColoringMode, ad_mode: JacobianMode
) -> JacobianMode:
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
    _assert_coloring_mode(coloring_mode)
    _assert_jacobian_mode(ad_mode)

    if coloring_mode == "auto":
        raise ValueError(
            "coloring_mode must be resolved before calling _resolve_ad_mode, "
            f"got {coloring_mode!r}."
        )

    match ad_mode:
        case "auto":
            match coloring_mode:
                case "row":
                    return "rev"
                case "column":
                    return "fwd"
                case "symmetric":
                    return "fwd"
                case _ as unreachable:
                    assert_never(unreachable)

        case "fwd":
            if coloring_mode == "row":
                raise ValueError(
                    f"Row coloring is only compatible with ad_mode='rev', got {ad_mode!r}."
                )
            return ad_mode

        case "rev":
            if coloring_mode == "column":
                raise ValueError(
                    f"Column coloring is only compatible with ad_mode='fwd', got {ad_mode!r}."
                )
            return ad_mode

        case _ as unreachable:
            assert_never(unreachable)


def _resolve_hessian_mode(ad_mode: HessianMode) -> HessianMode:
    """Resolve ``"auto"`` Hessian mode to ``"fwd_over_rev"``.

    Args:
        ad_mode: The requested Hessian AD mode.

    Returns:
        A concrete Hessian AD mode.

    Raises:
        ValueError: If ``ad_mode`` is unknown.
    """
    _assert_hessian_mode(ad_mode)
    match ad_mode:
        case "auto":
            return "fwd_over_rev"
        case "fwd_over_rev" | "rev_over_fwd" | "rev_over_rev":
            return ad_mode
        case _ as unreachable:
            assert_never(unreachable)

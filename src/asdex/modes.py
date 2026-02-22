"""Type aliases and validation for AD mode selection."""

from typing import Literal, assert_never, get_args

JacobianMode = Literal["fwd", "rev", "auto"]
"""AD mode for Jacobian computation.

``"fwd"`` uses JVPs (forward-mode AD),
``"rev"`` uses VJPs (reverse-mode AD),
``"auto"`` selects automatically.
"""

HessianMode = Literal["fwd_over_rev", "rev_over_fwd", "rev_over_rev", "auto"]
"""AD composition strategy for Hessian-vector products.

``"fwd_over_rev"`` uses forward-over-reverse,
``"rev_over_fwd"`` uses reverse-over-forward,
``"rev_over_rev"`` uses reverse-over-reverse,
``"auto"`` defaults to ``"fwd_over_rev"``.
"""


def _assert_jacobian_mode(mode: str) -> None:
    """Raise ``ValueError`` if *mode* is not a valid ``JacobianMode``."""
    if mode not in get_args(JacobianMode):
        raise ValueError(f"Unknown mode {mode!r}. Expected 'fwd', 'rev', or 'auto'.")


def _assert_hessian_mode(mode: str) -> None:
    """Raise ``ValueError`` if *mode* is not a valid ``HessianMode``."""
    if mode not in get_args(HessianMode):
        raise ValueError(
            f"Unknown mode {mode!r}. "
            "Expected 'fwd_over_rev', 'rev_over_fwd', 'rev_over_rev', or 'auto'."
        )


def _resolve_jacobian_mode(mode: JacobianMode, *, symmetric: bool) -> JacobianMode:
    """Resolve ``"auto"`` Jacobian mode.

    When ``symmetric`` is True, defaults to ``"fwd"``.
    When ``symmetric`` is False, returns ``mode`` unchanged
    (caller handles the auto-pick-best logic).

    Args:
        mode: The requested AD mode.
        symmetric: Whether the coloring is symmetric.

    Returns:
        A Jacobian AD mode (possibly still ``"auto"``).

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    _assert_jacobian_mode(mode)
    if mode != "auto":
        return mode
    if symmetric:
        return "fwd"
    return mode


def _resolve_hessian_mode(mode: HessianMode) -> HessianMode:
    """Resolve ``"auto"`` Hessian mode to ``"fwd_over_rev"``.

    Args:
        mode: The requested Hessian AD mode.

    Returns:
        A concrete Hessian AD mode.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    _assert_hessian_mode(mode)
    match mode:
        case "auto":
            return "fwd_over_rev"
        case "fwd_over_rev" | "rev_over_fwd" | "rev_over_rev":
            return mode
        case _ as unreachable:
            assert_never(unreachable)

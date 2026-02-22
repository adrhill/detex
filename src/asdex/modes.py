"""Type aliases and resolution for AD mode selection."""

from typing import Literal, get_args

JacobianMode = Literal["fwd", "rev"]
"""AD mode for Jacobian computation.

``"fwd"`` uses JVPs (forward-mode AD),
``"rev"`` uses VJPs (reverse-mode AD).
"""

HessianMode = Literal["fwd_over_rev", "rev_over_fwd", "rev_over_rev"]
"""AD composition strategy for Hessian-vector products.

``"fwd_over_rev"`` uses forward-over-reverse,
``"rev_over_fwd"`` uses reverse-over-forward,
``"rev_over_rev"`` uses reverse-over-reverse.
"""

ColoringMode = Literal["fwd", "rev", "fwd_over_rev", "rev_over_fwd", "rev_over_rev"]
"""AD mode that a coloring was computed for."""


def _assert_jacobian_mode(mode: str) -> None:
    """Raise ``ValueError`` if *mode* is not a valid ``JacobianMode``."""
    if mode not in get_args(JacobianMode):
        raise ValueError(f"Unknown mode {mode!r}. Expected 'fwd' or 'rev'.")


def _assert_hessian_mode(mode: str) -> None:
    """Raise ``ValueError`` if *mode* is not a valid ``HessianMode``."""
    if mode not in get_args(HessianMode):
        raise ValueError(
            f"Unknown mode {mode!r}. "
            "Expected 'fwd_over_rev', 'rev_over_fwd', or 'rev_over_rev'."
        )

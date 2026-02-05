"""Types, constants, and utilities for dependency tracking."""

import math
from collections.abc import Sequence

from jax._src.core import Literal, Var

# =============================================================================
# Type aliases
# =============================================================================

IndexSets = list[set[int]]
"""Per-element dependency index sets for an array."""

Deps = dict[Var, IndexSets]
"""Maps each variable to its per-element dependency index sets."""

Atom = Var | Literal
"""Atomic elements in jaxpressions: named intermediates (Var) or constants (Literal)."""


# =============================================================================
# Constants
# =============================================================================

# Primitives with zero derivatives (output doesn't depend on input)
ZERO_DERIVATIVE_PRIMITIVES = frozenset(
    [
        "floor",
        "ceil",
        "round",
        "sign",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "is_finite",
    ]
)

# Primitives that contain nested jaxprs we should trace into
NESTED_JAXPR_PRIMITIVES = frozenset(["jit", "pjit", "xla_call", "named_call"])


# =============================================================================
# Utility functions
# =============================================================================


def union_all(sets: Sequence[set[int]]) -> set[int]:
    """Union all sets together, returning a new set."""
    if not sets:
        return set()
    result: set[int] = set()
    for s in sets:
        result |= s
    return result


def numel(shape: Sequence[int]) -> int:
    """Compute the total number of elements from a shape tuple."""
    return math.prod(shape) if shape else 1


def atom_numel(atom: Atom) -> int:
    """Get the total number of elements in a variable or literal."""
    if isinstance(atom, Literal):
        shape = getattr(atom.val, "shape", ())
        return numel(tuple(shape)) if shape else 1
    shape = getattr(atom.aval, "shape", ())
    return numel(tuple(shape)) if shape else 1


def index_sets(deps: Deps, atom: Atom) -> IndexSets:
    """Get the index sets for a variable or literal."""
    if isinstance(atom, Literal):
        return [set() for _ in range(atom_numel(atom))]
    return deps.get(atom, [set()])


def row_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """Compute row-major strides for multi-dimensional index tracking.

    Used to convert between flat indices and coordinates when propagating
    dependencies through slice and broadcast_in_dim. Each stride tells how
    many flat elements to skip when incrementing one coordinate position.

    For shape (2, 3, 4): row_strides = (12, 4, 1) since moving one step in dim 0
    skips 3*4=12 elements, dim 1 skips 4 elements, and dim 2 skips 1 element.
    """
    result: list[int] = []
    stride = 1
    for dim in reversed(shape):
        result.append(stride)
        stride *= dim
    return tuple(reversed(result))

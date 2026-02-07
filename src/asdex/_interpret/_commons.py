"""Types, constants, and utilities for dependency tracking."""

import math
from collections.abc import Sequence

import numpy as np
from jax._src.core import Literal, Var

# =============================================================================
# Type aliases
# =============================================================================

IndexSets = list[set[int]]
"""Per-element dependency index sets for an array."""

Deps = dict[Var, IndexSets]
"""Maps each variable to its per-element dependency index sets."""

ConstVals = dict[Var, np.ndarray]
"""Maps variables to their concrete numpy array values (for static index tracking)."""

Atom = Var | Literal
"""Atomic elements in jaxpressions: named intermediates (Var) or constants (Literal)."""


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


def atom_const_val(atom: Atom, const_vals: ConstVals) -> np.ndarray | None:
    """Get the concrete value of an atom, if statically known.

    The value is known in two cases:
    - **Literals**: constants embedded directly in the jaxpr.
    - **Tracked vars**: variables in ``const_vals``, whose values were
      computed from constants through earlier operations.

    Returns ``None`` when the value depends on runtime inputs.
    """
    if isinstance(atom, Literal):
        return np.asarray(atom.val)
    if isinstance(atom, Var) and atom in const_vals:
        return const_vals[atom]
    return None


def index_sets(deps: Deps, atom: Atom) -> IndexSets:
    """Get the index sets for a variable or literal."""
    if isinstance(atom, Literal):
        return [set() for _ in range(atom_numel(atom))]
    return deps.get(atom, [set()])


def seed_const_vals(const_vals: ConstVals, constvars, consts) -> None:
    """Populate const_vals for the captured constants of a ClosedJaxpr.

    Without this, gather/scatter inside nested jaxprs (cond branches,
    while bodies, jit-wrapped calls) cannot resolve closure-captured
    index arrays and fall back to conservative.
    """
    for var, val in zip(constvars, consts, strict=True):
        const_vals[var] = np.asarray(val)


def forward_const_vals(
    const_vals: ConstVals, outer_atoms: Sequence[Atom], inner_vars
) -> None:
    """Transfer known const_vals from outer-scope atoms to inner jaxpr variables.

    When entering a nested jaxpr (cond branch, while body, jit call),
    the outer equation's invars and the inner jaxpr's invars are different
    ``Var`` objects representing the same values.
    This copies any concrete values from the outer atoms
    to the corresponding inner vars so that downstream handlers
    (gather, scatter, dynamic_slice) can resolve indices precisely.
    """
    for outer, inner in zip(outer_atoms, inner_vars, strict=False):
        val = atom_const_val(outer, const_vals)
        if val is not None:
            const_vals[inner] = val


def atom_shape(atom: Atom) -> tuple[int, ...]:
    """Get the shape of a variable or literal."""
    if isinstance(atom, Literal):
        return tuple(getattr(atom.val, "shape", ()))
    return tuple(getattr(atom.aval, "shape", ()))


def flat_to_coords(flat: int, strides: tuple[int, ...]) -> list[int]:
    """Convert a flat index to multi-dimensional coordinates using row-major strides."""
    coord = []
    remaining = flat
    for s in strides:
        coord.append(remaining // s)
        remaining %= s
    return coord


def conservative_deps(all_indices: IndexSets, out_size: int) -> IndexSets:
    """Build conservative output deps where every element depends on the union of all inputs."""
    all_deps = union_all(all_indices)
    return [all_deps.copy() for _ in range(out_size)]


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

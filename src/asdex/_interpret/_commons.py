"""Types, constants, and utilities for dependency tracking."""

import math
from collections.abc import Callable, Sequence

import numpy as np
from jax._src.core import Jaxpr, JaxprEqn, Literal, Var

IndexSet = set[int]
"""A single per-element dependency set.

Currently backed by Python's built-in set.
Designed for a future swap to pyroaring.BitMap.
"""


def empty_index_set() -> IndexSet:
    """Create an empty dependency set."""
    return set()


def singleton_index_set(i: int) -> IndexSet:
    """Create a dependency set containing a single index."""
    return {i}


def empty_index_sets(n: int) -> list[IndexSet]:
    """Create n empty dependency sets."""
    return [empty_index_set() for _ in range(n)]


def identity_index_sets(n: int) -> list[IndexSet]:
    """Create identity sets where element i depends on index i."""
    return [singleton_index_set(i) for i in range(n)]


Deps = dict[Var, list[IndexSet]]
"""Maps each variable to its per-element dependency index sets."""

ConstVals = dict[Var, np.ndarray]
"""Maps variables to their concrete numpy array values (for static index tracking)."""

Atom = Var | Literal
"""Atomic elements in jaxpressions: named intermediates (Var) or constants (Literal)."""

PropJaxprFn = Callable[
    [Jaxpr, list[list[IndexSet]], ConstVals | None], list[list[IndexSet]]
]
"""Signature of ``prop_jaxpr``, passed as callback to break circular imports."""

_MAX_FIXED_POINT_ITERS = 500
"""Safety bound for fixed-point iteration in while_loop and scan."""


# Shape and size


def numel(shape: Sequence[int]) -> int:
    """Compute the total number of elements from a shape tuple."""
    return math.prod(shape) if shape else 1


def atom_shape(atom: Atom) -> tuple[int, ...]:
    """Get the shape of a variable or literal."""
    if isinstance(atom, Literal):
        return tuple(getattr(atom.val, "shape", ()))
    return tuple(getattr(atom.aval, "shape", ()))


def atom_numel(atom: Atom) -> int:
    """Get the total number of elements in a variable or literal."""
    if isinstance(atom, Literal):
        shape = getattr(atom.val, "shape", ())
        return numel(tuple(shape)) if shape else 1
    shape = getattr(atom.aval, "shape", ())
    return numel(tuple(shape)) if shape else 1


# Atom value access


def index_sets(deps: Deps, atom: Atom) -> list[IndexSet]:
    """Get the index sets for a variable or literal."""
    if isinstance(atom, Literal):
        return empty_index_sets(atom_numel(atom))
    return deps.get(atom, [empty_index_set()])


def copy_index_sets(src: list[IndexSet]) -> list[IndexSet]:
    """Deep-copy a list of index sets."""
    return [s.copy() for s in src]


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


def propagate_const_unary(
    eqn: JaxprEqn,
    const_vals: ConstVals,
    transform: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Propagate a const value through a unary op.

    If the input is statically known,
    apply ``transform`` and store the result.
    Without this, downstream handlers (e.g. ``gather``, ``scatter``) cannot resolve
    static index arrays and fall back to conservative.
    """
    in_val = atom_const_val(eqn.invars[0], const_vals)
    if in_val is not None:
        const_vals[eqn.outvars[0]] = transform(in_val)


def propagate_const_binary(
    eqn: JaxprEqn,
    const_vals: ConstVals,
    transform: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    """Propagate a const value through a binary op.

    If both inputs are statically known,
    apply ``transform`` and store the result.
    Without this, downstream handlers (e.g. ``gather``, ``scatter``) cannot resolve
    static index arrays and fall back to conservative.
    """
    in1 = atom_const_val(eqn.invars[0], const_vals)
    in2 = atom_const_val(eqn.invars[1], const_vals)
    if in1 is not None and in2 is not None:
        const_vals[eqn.outvars[0]] = transform(in1, in2)


# Index set operations


def union_all(sets: Sequence[IndexSet]) -> IndexSet:
    """Union all sets together, returning a new set."""
    if not sets:
        return empty_index_set()
    result: IndexSet = empty_index_set()
    for s in sets:
        result |= s
    return result


def check_no_index_sets(deps: Deps, atom: Atom, primitive_name: str) -> None:
    """Verify that an atom carries no input dependencies.

    Some handlers assume that auxiliary inputs
    (index arrays, kernel weights, selectors)
    are constants with empty dependency sets.
    This function validates that assumption
    and raises an informative error when it is violated.
    """
    if any(index_sets(deps, atom)):
        msg = (
            f"'{primitive_name}' handler assumes an auxiliary input "
            "has no dependency on the function's inputs, "
            "but found non-empty index sets. "
            "Please help out asdex's development by reporting this at https://github.com/adrhill/asdex/issues"
        )
        raise ValueError(msg)


def conservative_indices(all_indices: list[IndexSet], out_size: int) -> list[IndexSet]:
    """Build conservative output index sets where every element depends on the union of all inputs."""
    combined = union_all(all_indices)
    return [combined.copy() for _ in range(out_size)]


# Position maps


def position_map(shape: Sequence[int]) -> np.ndarray:
    """Build an array where each element holds its own flat position.

    For shape ``(2, 3)``, returns ``[[0, 1, 2], [3, 4, 5]]``.
    Applying operations (transpose, slice, etc.) to this array
    reveals which input position each output position reads from.
    """
    return np.arange(numel(shape)).reshape(shape)


def permute_indices(
    in_indices: list[IndexSet], permutation_map: Sequence[int] | np.ndarray
) -> list[IndexSet]:
    """Build output index sets by permuting through a flat index array.

    Each output element ``i`` copies its index set from ``in_indices[permutation_map[i]]``.
    This is the common pattern for permutation-like ops
    (transpose, rev, slice, reshape, broadcast, etc.)
    where each output reads exactly one input element.
    """
    return [in_indices[j].copy() for j in permutation_map]


# Coordinate helpers


def row_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """Compute row-major strides for multi-dimensional index tracking.

    Used to convert between flat indices and coordinates when propagating
    dependencies through slice and broadcast_in_dim.
    Each stride tells how many flat elements to skip
    when incrementing one coordinate position.

    For shape (2, 3, 4): row_strides = (12, 4, 1) since moving one step in dim 0
    skips 3*4=12 elements, dim 1 skips 4 elements, and dim 2 skips 1 element.
    """
    result: list[int] = []
    stride = 1
    for dim in reversed(shape):
        result.append(stride)
        stride *= dim
    return tuple(reversed(result))


def flat_to_coords(flat: int, strides: tuple[int, ...]) -> list[int]:
    """Convert a flat index to multi-dimensional coordinates using row-major strides."""
    coord = []
    remaining = flat
    for s in strides:
        coord.append(remaining // s)
        remaining %= s
    return coord


# Const value propagation


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


# Fixed-point iteration


def fixed_point_loop(
    iterate_fn: Callable[[list[list[IndexSet]]], list[list[IndexSet]]],
    carry: list[list[IndexSet]],
    n_carry: int,
) -> list[list[IndexSet]]:
    """Run ``iterate_fn`` on carry index sets until they stabilize.

    Used by ``while_loop`` and ``scan`` to propagate index sets
    through loops via fixed-point iteration.
    Since index sets only grow and are bounded in size
    (i.e., monotone on a finite lattice),
    this always converges.

    Mutates ``carry`` in place and returns the final body output
    (needed by ``scan`` for ``y_slice`` extraction; ignored by ``while_loop``).
    """
    body_output: list[list[IndexSet]] = []
    for _iteration in range(_MAX_FIXED_POINT_ITERS):
        body_output = iterate_fn(carry)

        changed = False
        for i in range(n_carry):
            for j in range(len(carry[i])):
                before = len(carry[i][j])
                carry[i][j] |= body_output[i][j]
                if len(carry[i][j]) > before:
                    changed = True

        if not changed:
            break
    else:
        msg = (
            f"Fixed-point iteration did not converge after "
            f"{_MAX_FIXED_POINT_ITERS} iterations. "
            "Please help out asdex's development by reporting this at "
            "https://github.com/adrhill/asdex/issues"
        )
        raise RuntimeError(msg)  # pragma: no cover

    return body_output

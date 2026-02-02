"""Index set implementations for sparsity tracking.

Provides a unified interface with two backends:
- set[int]: Fast for sparse patterns, O(k) where k = bits set
- bytearray: Fixed memory O(n/8), better for dense patterns at large n

The `create_index_set_ops` factory returns operations tuned for input size.
"""

from collections.abc import Callable, Iterator
from typing import Protocol


class IndexSet(Protocol):
    """Protocol for index set operations."""

    def copy(self) -> "IndexSet": ...
    def __iter__(self) -> Iterator[int]: ...
    def __ior__(self, other: "IndexSet") -> "IndexSet": ...


# =============================================================================
# Set-based implementation (fast for sparse patterns)
# =============================================================================


def _set_empty() -> set[int]:
    return set()


def _set_single(i: int) -> set[int]:
    return {i}


def _set_union_all(sets: list[set[int]]) -> set[int]:
    if not sets:
        return set()
    result = sets[0].copy()
    for s in sets[1:]:
        result |= s
    return result


# =============================================================================
# Bytearray-based implementation (fixed memory for large n)
# =============================================================================


class BitSet:
    """Mutable bitset backed by bytearray."""

    __slots__ = ("_data", "_nbytes")

    def __init__(self, data: bytearray, nbytes: int) -> None:
        self._data = data
        self._nbytes = nbytes

    def copy(self) -> "BitSet":
        return BitSet(bytearray(self._data), self._nbytes)

    def __ior__(self, other: "BitSet") -> "BitSet":
        a = int.from_bytes(self._data, "little")
        b = int.from_bytes(other._data, "little")
        self._data[:] = (a | b).to_bytes(self._nbytes, "little")
        return self

    def __iter__(self) -> Iterator[int]:
        for byte_idx, byte in enumerate(self._data):
            if not byte:
                continue
            base = byte_idx * 8
            while byte:
                if byte & 1:
                    yield base
                byte >>= 1
                base += 1


def _make_bitset_ops(
    n: int,
) -> tuple[
    Callable[[], BitSet],
    Callable[[int], BitSet],
    Callable[[list[BitSet]], BitSet],
]:
    nbytes = (n + 7) // 8

    def empty() -> BitSet:
        return BitSet(bytearray(nbytes), nbytes)

    def single(i: int) -> BitSet:
        data = bytearray(nbytes)
        data[i // 8] = 1 << (i % 8)
        return BitSet(data, nbytes)

    def union_all(bitsets: list[BitSet]) -> BitSet:
        if not bitsets:
            return empty()
        result = bitsets[0].copy()
        for bs in bitsets[1:]:
            result |= bs
        return result

    return empty, single, union_all


# =============================================================================
# Factory function
# =============================================================================

# Threshold for switching to bitsets (based on benchmarks, sets are faster below this)
_BITSET_THRESHOLD = 10000  # Sets win up to at least n=4000 in benchmarks

# Type alias for index sets (either implementation)
type AnyIndexSet = set[int] | BitSet


class IndexSetOps:
    """Operations for creating and manipulating index sets."""

    __slots__ = ("empty", "single", "union_all", "n", "using_bitset")

    empty: Callable[[], AnyIndexSet]
    single: Callable[[int], AnyIndexSet]
    union_all: Callable[[list[AnyIndexSet]], AnyIndexSet]
    n: int
    using_bitset: bool

    def __init__(
        self,
        empty: Callable[[], AnyIndexSet],
        single: Callable[[int], AnyIndexSet],
        union_all: Callable[[list[AnyIndexSet]], AnyIndexSet],
        n: int,
        using_bitset: bool,
    ) -> None:
        self.empty = empty
        self.single = single
        self.union_all = union_all
        self.n = n
        self.using_bitset = using_bitset


def create_index_set_ops(n: int, force_bitset: bool = False) -> IndexSetOps:
    """Create index set operations for the given input dimension.

    Args:
        n: Input dimension (number of bits needed)
        force_bitset: Force bitset implementation regardless of n

    Returns:
        IndexSetOps with empty(), single(i), and union_all(sets) functions
    """
    if force_bitset or n >= _BITSET_THRESHOLD:
        empty, single, union_all = _make_bitset_ops(n)
        return IndexSetOps(empty, single, union_all, n, using_bitset=True)  # type: ignore[arg-type]
    else:
        return IndexSetOps(
            _set_empty,
            _set_single,
            _set_union_all,
            n,
            using_bitset=False,  # type: ignore[arg-type]
        )

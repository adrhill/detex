"""Index set implementations for sparsity tracking.

Provides a unified interface with two backends:
- IdxSet: Fast for sparse patterns, O(k) where k = bits set
- IdxBitset: Fixed memory O(n/8), better for dense patterns at large n
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Self


class AbstractIdx(ABC):
    """Abstract base class for index set operations."""

    __slots__ = ()

    @abstractmethod
    def copy(self) -> Self: ...

    @abstractmethod
    def __iter__(self) -> Iterator[int]: ...

    @abstractmethod
    def __ior__(self, other: Self) -> Self: ...


# =============================================================================
# Set-based implementation (fast for sparse patterns)
# =============================================================================


class IdxSet(AbstractIdx):
    """Index set backed by Python set[int]."""

    __slots__ = ("_data",)

    def __init__(self, data: set[int] | None = None) -> None:
        self._data: set[int] = data if data is not None else set()

    def copy(self) -> "IdxSet":
        return IdxSet(self._data.copy())

    def __iter__(self) -> Iterator[int]:
        return iter(self._data)

    def __ior__(self, other: "IdxSet") -> "IdxSet":  # type: ignore[override]
        self._data |= other._data
        return self

    @staticmethod
    def empty() -> "IdxSet":
        return IdxSet()

    @staticmethod
    def single(i: int) -> "IdxSet":
        return IdxSet({i})

    @staticmethod
    def union_all(sets: list["IdxSet"]) -> "IdxSet":
        if not sets:
            return IdxSet()
        result = sets[0].copy()
        for s in sets[1:]:
            result |= s
        return result


# =============================================================================
# Bytearray-based implementation (fixed memory for large n)
# =============================================================================


class IdxBitset(AbstractIdx):
    """Index set backed by bytearray for fixed memory usage."""

    __slots__ = ("_data", "_nbytes")

    def __init__(self, data: bytearray, nbytes: int) -> None:
        self._data = data
        self._nbytes = nbytes

    def copy(self) -> "IdxBitset":
        return IdxBitset(bytearray(self._data), self._nbytes)

    def __ior__(self, other: "IdxBitset") -> "IdxBitset":  # type: ignore[override]
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

    @staticmethod
    def empty(nbytes: int) -> "IdxBitset":
        return IdxBitset(bytearray(nbytes), nbytes)

    @staticmethod
    def single(i: int, nbytes: int) -> "IdxBitset":
        data = bytearray(nbytes)
        data[i // 8] = 1 << (i % 8)
        return IdxBitset(data, nbytes)

    @staticmethod
    def union_all(bitsets: list["IdxBitset"]) -> "IdxBitset":
        if not bitsets:
            raise ValueError("Cannot union empty list without nbytes")
        result = bitsets[0].copy()
        for bs in bitsets[1:]:
            result |= bs
        return result

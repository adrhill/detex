"""Tests for bitset.py index set implementations."""

from detex._indexset import IdxBitset, IdxSet

# =============================================================================
# IdxSet tests
# =============================================================================


def test_idxset_empty():
    """Empty set returns an empty set."""
    result = IdxSet.empty()
    assert list(result) == []


def test_idxset_single():
    """Single creates a set with one element."""
    assert list(IdxSet.single(0)) == [0]
    assert list(IdxSet.single(5)) == [5]
    assert list(IdxSet.single(100)) == [100]


def test_idxset_union_all_empty_list():
    """Union of empty list returns empty set."""
    result = IdxSet.union_all([])
    assert list(result) == []


def test_idxset_union_all_single():
    """Union of single set returns copy of that set."""
    original = IdxSet.single(1)
    original |= IdxSet.single(2)
    original |= IdxSet.single(3)
    result = IdxSet.union_all([original])
    assert sorted(result) == [1, 2, 3]
    # Should be a copy, not the same object
    assert result is not original


def test_idxset_union_all_multiple():
    """Union of multiple sets combines all elements."""
    s1 = IdxSet.single(0)
    s1 |= IdxSet.single(1)
    s2 = IdxSet.single(2)
    s2 |= IdxSet.single(3)
    s3 = IdxSet.single(1)
    s3 |= IdxSet.single(4)
    result = IdxSet.union_all([s1, s2, s3])
    assert sorted(result) == [0, 1, 2, 3, 4]


def test_idxset_union_all_disjoint():
    """Union of disjoint sets."""
    result = IdxSet.union_all([IdxSet.single(0), IdxSet.single(1), IdxSet.single(2)])
    assert sorted(result) == [0, 1, 2]


def test_idxset_copy():
    """IdxSet.copy() creates an independent copy."""
    s = IdxSet.single(5)
    copy = s.copy()

    assert list(copy) == [5]
    # Modify original, copy should be unchanged
    s |= IdxSet.single(10)
    assert list(copy) == [5]
    assert sorted(s) == [5, 10]


def test_idxset_ior():
    """IdxSet |= performs in-place union."""
    s1 = IdxSet.single(10)
    s2 = IdxSet.single(20)

    result = s1.__ior__(s2)

    assert result is s1  # Returns self
    assert sorted(s1) == [10, 20]


# =============================================================================
# IdxBitset tests
# =============================================================================


def test_idxbitset_empty():
    """empty() creates a bitset with no bits set."""
    bs = IdxBitset.empty(nbytes=1)
    assert list(bs) == []


def test_idxbitset_single():
    """single(i) creates a bitset with only bit i set."""
    bs = IdxBitset.single(0, nbytes=2)
    assert list(bs) == [0]

    bs = IdxBitset.single(15, nbytes=2)
    assert list(bs) == [15]


def test_idxbitset_copy():
    """IdxBitset.copy() creates an independent copy."""
    bs = IdxBitset.single(5, nbytes=2)
    copy = bs.copy()

    assert list(copy) == [5]
    # Modify original, copy should be unchanged
    bs |= IdxBitset.single(10, nbytes=2)
    assert list(copy) == [5]
    assert sorted(bs) == [5, 10]


def test_idxbitset_ior():
    """IdxBitset |= performs in-place union."""
    bs1 = IdxBitset.single(3, nbytes=2)
    bs2 = IdxBitset.single(7, nbytes=2)

    result = bs1.__ior__(bs2)

    assert result is bs1  # Returns self
    assert sorted(bs1) == [3, 7]


def test_idxbitset_iter_empty():
    """Iteration over empty bitset yields nothing."""
    bs = IdxBitset.empty(nbytes=2)
    assert list(bs) == []


def test_idxbitset_iter_single():
    """Iteration over single-element bitset yields that element."""
    bs = IdxBitset.single(5, nbytes=2)
    assert list(bs) == [5]


def test_idxbitset_iter_multiple():
    """Iteration yields all set bits in order."""
    bs = IdxBitset.single(0, nbytes=4)
    bs |= IdxBitset.single(7, nbytes=4)
    bs |= IdxBitset.single(8, nbytes=4)
    bs |= IdxBitset.single(15, nbytes=4)
    bs |= IdxBitset.single(24, nbytes=4)
    assert list(bs) == [0, 7, 8, 15, 24]


def test_idxbitset_iter_byte_boundary():
    """Bits at byte boundaries are handled correctly."""
    bs = IdxBitset.single(7, nbytes=3)
    bs |= IdxBitset.single(8, nbytes=3)
    bs |= IdxBitset.single(15, nbytes=3)
    bs |= IdxBitset.single(16, nbytes=3)
    assert list(bs) == [7, 8, 15, 16]


def test_idxbitset_union_all():
    """union_all combines multiple bitsets."""
    nbytes = 2
    result = IdxBitset.union_all(
        [
            IdxBitset.single(1, nbytes),
            IdxBitset.single(5, nbytes),
            IdxBitset.single(10, nbytes),
        ]
    )
    assert sorted(result) == [1, 5, 10]


def test_idxbitset_large_n():
    """Bitset operations work for larger n values."""
    nbytes = 125  # 1000 bits

    bs = IdxBitset.single(999, nbytes)
    assert list(bs) == [999]

    combined = IdxBitset.single(0, nbytes)
    combined |= IdxBitset.single(500, nbytes)
    combined |= IdxBitset.single(999, nbytes)
    assert sorted(combined) == [0, 500, 999]

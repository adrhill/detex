"""Tests for bitset.py index set implementations."""

from detex.bitset import (
    _BITSET_THRESHOLD,
    BitSet,
    _make_bitset_ops,
    _set_empty,
    _set_single,
    _set_union_all,
    create_index_set_ops,
)

# =============================================================================
# Set-based implementation tests
# =============================================================================


def test_set_empty():
    """Empty set returns an empty set."""
    result = _set_empty()
    assert result == set()
    assert len(result) == 0


def test_set_single():
    """Single creates a set with one element."""
    assert _set_single(0) == {0}
    assert _set_single(5) == {5}
    assert _set_single(100) == {100}


def test_set_union_all_empty_list():
    """Union of empty list returns empty set."""
    result = _set_union_all([])
    assert result == set()


def test_set_union_all_single():
    """Union of single set returns copy of that set."""
    original = {1, 2, 3}
    result = _set_union_all([original])
    assert result == {1, 2, 3}
    # Should be a copy, not the same object
    assert result is not original


def test_set_union_all_multiple():
    """Union of multiple sets combines all elements."""
    result = _set_union_all([{0, 1}, {2, 3}, {1, 4}])
    assert result == {0, 1, 2, 3, 4}


def test_set_union_all_disjoint():
    """Union of disjoint sets."""
    result = _set_union_all([{0}, {1}, {2}])
    assert result == {0, 1, 2}


# =============================================================================
# BitSet class tests
# =============================================================================


def test_bitset_copy():
    """BitSet.copy() creates an independent copy."""
    empty, single, _ = _make_bitset_ops(16)
    bs = single(5)
    copy = bs.copy()

    assert list(copy) == [5]
    # Modify original, copy should be unchanged
    bs |= single(10)
    assert list(copy) == [5]
    assert sorted(bs) == [5, 10]


def test_bitset_ior():
    """BitSet |= performs in-place union."""
    empty, single, _ = _make_bitset_ops(16)
    bs1 = single(3)
    bs2 = single(7)

    result = bs1.__ior__(bs2)

    assert result is bs1  # Returns self
    assert sorted(bs1) == [3, 7]


def test_bitset_iter_empty():
    """Iteration over empty bitset yields nothing."""
    empty, _, _ = _make_bitset_ops(16)
    bs = empty()
    assert list(bs) == []


def test_bitset_iter_single():
    """Iteration over single-element bitset yields that element."""
    _, single, _ = _make_bitset_ops(16)
    bs = single(5)
    assert list(bs) == [5]


def test_bitset_iter_multiple():
    """Iteration yields all set bits in order."""
    empty, single, union_all = _make_bitset_ops(32)
    bs = union_all([single(0), single(7), single(8), single(15), single(24)])
    assert list(bs) == [0, 7, 8, 15, 24]


def test_bitset_iter_byte_boundary():
    """Bits at byte boundaries are handled correctly."""
    _, single, union_all = _make_bitset_ops(24)
    # Bits at positions 7, 8 (byte boundary), 15, 16 (byte boundary)
    bs = union_all([single(7), single(8), single(15), single(16)])
    assert list(bs) == [7, 8, 15, 16]


# =============================================================================
# _make_bitset_ops tests
# =============================================================================


def test_make_bitset_ops_empty():
    """empty() creates a bitset with no bits set."""
    empty, _, _ = _make_bitset_ops(8)
    bs = empty()
    assert list(bs) == []


def test_make_bitset_ops_single():
    """single(i) creates a bitset with only bit i set."""
    _, single, _ = _make_bitset_ops(16)
    bs = single(0)
    assert list(bs) == [0]

    bs = single(15)
    assert list(bs) == [15]


def test_make_bitset_ops_union_all_empty():
    """union_all([]) returns empty bitset."""
    empty, _, union_all = _make_bitset_ops(8)
    result = union_all([])
    assert list(result) == []


def test_make_bitset_ops_union_all():
    """union_all combines multiple bitsets."""
    _, single, union_all = _make_bitset_ops(16)
    result = union_all([single(1), single(5), single(10)])
    assert sorted(result) == [1, 5, 10]


def test_make_bitset_ops_large_n():
    """Bitset operations work for larger n values."""
    empty, single, union_all = _make_bitset_ops(1000)

    bs = single(999)
    assert list(bs) == [999]

    combined = union_all([single(0), single(500), single(999)])
    assert sorted(combined) == [0, 500, 999]


# =============================================================================
# create_index_set_ops tests
# =============================================================================


def test_create_index_set_ops_uses_set_below_threshold():
    """Below threshold, set-based implementation is used."""
    ops = create_index_set_ops(100)
    assert ops.using_bitset is False
    assert ops.n == 100

    # Verify it actually uses sets
    s = ops.single(5)
    assert isinstance(s, set)


def test_create_index_set_ops_uses_bitset_at_threshold():
    """At or above threshold, bitset implementation is used."""
    ops = create_index_set_ops(_BITSET_THRESHOLD)
    assert ops.using_bitset is True
    assert ops.n == _BITSET_THRESHOLD

    # Verify it uses BitSet
    bs = ops.single(5)
    assert isinstance(bs, BitSet)


def test_create_index_set_ops_force_bitset():
    """force_bitset=True uses bitset regardless of n."""
    ops = create_index_set_ops(10, force_bitset=True)
    assert ops.using_bitset is True
    assert ops.n == 10

    bs = ops.single(5)
    assert isinstance(bs, BitSet)


def test_index_set_ops_interface_consistency():
    """Both implementations have consistent behavior."""
    for use_bitset in [False, True]:
        ops = create_index_set_ops(100, force_bitset=use_bitset)

        # empty
        e = ops.empty()
        assert list(e) == []

        # single
        s = ops.single(42)
        assert list(s) == [42]

        # union_all with empty
        u = ops.union_all([])
        assert list(u) == []

        # union_all with multiple
        u = ops.union_all([ops.single(1), ops.single(5), ops.single(3)])
        assert sorted(u) == [1, 3, 5]


def test_set_copy_and_ior():
    """Set implementation supports copy and |= correctly."""
    s1 = _set_single(10)
    s2 = _set_single(20)

    s1_copy = s1.copy()
    s1 |= s2
    assert sorted(s1) == [10, 20]
    assert list(s1_copy) == [10]


def test_bitset_copy_and_ior():
    """BitSet implementation supports copy and |= correctly."""
    _, single, _ = _make_bitset_ops(100)

    s1 = single(10)
    s2 = single(20)

    s1_copy = s1.copy()
    s1 |= s2
    assert sorted(s1) == [10, 20]
    assert list(s1_copy) == [10]

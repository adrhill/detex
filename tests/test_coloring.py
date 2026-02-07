"""Tests for graph coloring algorithms.

Test cases inspired by SparseMatrixColorings.jl (MIT license)
Copyright (c) 2024 Guillaume Dalle, Alexis Montoison, and contributors
https://github.com/gdalle/SparseMatrixColorings.jl
See also: Dalle & Montoison (2025), https://arxiv.org/abs/2505.07308
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from asdex import (
    ColoredPattern,
    SparsityPattern,
    color,
    color_cols,
    color_rows,
    hessian,
    hessian_coloring,
    jacobian_coloring,
    star_color,
)


def _make_pattern(
    rows: list[int], cols: list[int], shape: tuple[int, int]
) -> SparsityPattern:
    """Helper to create SparsityPattern from row/col lists."""
    return SparsityPattern.from_coordinates(rows, cols, shape)


def _from_dense(matrix: list[list[int]]) -> SparsityPattern:
    """Helper to create SparsityPattern from dense 0/1 matrix."""
    return SparsityPattern.from_dense(np.array(matrix))


def _make_banded(n: int, half_bandwidth: int) -> SparsityPattern:
    """Symmetric banded matrix with given half-bandwidth.

    Matches SparseMatrixColorings.jl's ``banded_matrix(n, 2*half_bandwidth)``.
    """
    rows, cols = [], []
    for i in range(n):
        for k in range(-half_bandwidth, half_bandwidth + 1):
            j = i + k
            if 0 <= j < n:
                rows.append(i)
                cols.append(j)
    return _make_pattern(rows, cols, (n, n))


def _make_arrow(n: int) -> SparsityPattern:
    """Arrow matrix: diagonal + dense first row/column."""
    rows, cols = [], []
    for i in range(n):
        rows.append(i)
        cols.append(i)  # diagonal
        if i > 0:
            rows.append(0)
            cols.append(i)  # first row
            rows.append(i)
            cols.append(0)  # first col
    return _make_pattern(rows, cols, (n, n))


def _is_valid_row_coloring(sparsity: SparsityPattern, colors: np.ndarray) -> bool:
    """Check that no column has two rows with the same color."""
    col_to_rows = sparsity.col_to_rows
    for rows_in_col in col_to_rows.values():
        colors_in_col = colors[rows_in_col]
        if len(colors_in_col) != len(set(colors_in_col)):
            return False
    return True


def _is_valid_col_coloring(sparsity: SparsityPattern, colors: np.ndarray) -> bool:
    """Check that no row has two columns with the same color."""
    row_to_cols = sparsity.row_to_cols
    for cols_in_row in row_to_cols.values():
        colors_in_row = colors[cols_in_row]
        if len(colors_in_row) != len(set(colors_in_row)):
            return False
    return True


def _is_valid_star_coloring(sparsity: SparsityPattern, colors: np.ndarray) -> bool:
    """Check distance-1 coloring + no 2-colored 4-vertex path.

    A star coloring satisfies:
    1. Adjacent vertices have different colors (distance-1).
    2. Every path on 4 vertices uses at least 3 distinct colors.
    """
    n = sparsity.n

    # Build adjacency (undirected, exclude diagonal)
    adj: list[set[int]] = [set() for _ in range(n)]
    for i, j in zip(sparsity.rows, sparsity.cols, strict=True):
        i, j = int(i), int(j)
        if i != j:
            adj[i].add(j)
            adj[j].add(i)

    # Check distance-1: adjacent vertices must have different colors
    for v in range(n):
        for w in adj[v]:
            if colors[v] == colors[w]:
                return False

    # Check no 2-colored 4-vertex path:
    # For every path v0-v1-v2-v3, the set {colors[v0],...,colors[v3]} has size >= 3.
    for v1 in range(n):
        for v2 in adj[v1]:
            if v2 <= v1:
                continue  # avoid checking each edge twice
            for v0 in adj[v1]:
                if v0 == v2:
                    continue
                for v3 in adj[v2]:
                    if v3 == v1:
                        continue
                    path_colors = {colors[v0], colors[v1], colors[v2], colors[v3]}
                    if len(path_colors) < 3:
                        return False

    return True


# =============================================================================
# Row coloring tests
# =============================================================================


@pytest.mark.coloring
def test_diagonal_one_color():
    """Diagonal matrix: all rows are independent, should use 1 color."""
    sparsity = _make_pattern([0, 1, 2, 3], [0, 1, 2, 3], (4, 4))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 1
    assert len(colors) == 4
    assert np.all(colors == 0)
    assert _is_valid_row_coloring(sparsity, colors)


@pytest.mark.coloring
def test_dense_m_colors():
    """Dense matrix: every row conflicts with every other, needs m colors."""
    rows, cols = [], []
    for i in range(4):
        for j in range(4):
            rows.append(i)
            cols.append(j)
    sparsity = _make_pattern(rows, cols, (4, 4))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 4
    assert len(colors) == 4
    assert len(set(colors)) == 4  # All different colors
    assert _is_valid_row_coloring(sparsity, colors)


@pytest.mark.coloring
def test_block_diagonal():
    """Block diagonal: non-overlapping blocks can share colors."""
    # Two 2x2 blocks
    rows = [0, 0, 1, 1, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 2, 3, 2, 3]
    sparsity = _make_pattern(rows, cols, (4, 4))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 2
    assert _is_valid_row_coloring(sparsity, colors)
    # Rows 0,1 conflict; rows 2,3 conflict; but 0,2 and 1,3 don't
    assert colors[0] != colors[1]
    assert colors[2] != colors[3]


@pytest.mark.coloring
def test_tridiagonal():
    """Tridiagonal matrix: needs 2-3 colors depending on structure."""
    # 4x4 tridiagonal
    rows = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]
    sparsity = _make_pattern(rows, cols, (4, 4))

    colors, num_colors = color_rows(sparsity)

    # Tridiagonal needs at most 3 colors (greedy may use 2-3)
    assert 2 <= num_colors <= 3
    assert _is_valid_row_coloring(sparsity, colors)


@pytest.mark.coloring
def test_single_row():
    """Single row matrix."""
    sparsity = _make_pattern([0, 0, 0], [0, 1, 2], (1, 3))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 1
    assert len(colors) == 1
    assert colors[0] == 0


@pytest.mark.coloring
def test_single_column():
    """Single column matrix: all rows conflict."""
    sparsity = _make_pattern([0, 1, 2], [0, 0, 0], (3, 1))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 3
    assert len(set(colors)) == 3
    assert _is_valid_row_coloring(sparsity, colors)


@pytest.mark.coloring
def test_empty_matrix():
    """Empty matrix (0 rows)."""
    sparsity = _make_pattern([], [], (0, 3))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 0
    assert len(colors) == 0


@pytest.mark.coloring
def test_zero_matrix():
    """Matrix with no non-zeros: all rows independent."""
    sparsity = _make_pattern([], [], (3, 3))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 1
    assert len(colors) == 3
    assert np.all(colors == 0)


@pytest.mark.coloring
def test_lower_triangular():
    """Lower triangular: increasing conflicts per row."""
    # 4x4 lower triangular
    rows = []
    cols = []
    for i in range(4):
        for j in range(i + 1):
            rows.append(i)
            cols.append(j)
    sparsity = _make_pattern(rows, cols, (4, 4))

    colors, num_colors = color_rows(sparsity)

    assert _is_valid_row_coloring(sparsity, colors)
    # Lower triangular needs 4 colors (row 3 conflicts with all)
    assert num_colors == 4


@pytest.mark.coloring
def test_checkerboard():
    """Checkerboard pattern: alternating rows/cols."""
    # 4x4 checkerboard (even rows: even cols, odd rows: odd cols)
    rows = []
    cols = []
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                rows.append(i)
                cols.append(j)
    sparsity = _make_pattern(rows, cols, (4, 4))

    colors, num_colors = color_rows(sparsity)

    assert _is_valid_row_coloring(sparsity, colors)
    # Even rows share cols 0,2; odd rows share cols 1,3
    # So we need 2 colors
    assert num_colors == 2


@pytest.mark.coloring
def test_largest_first_improves_coloring():
    """LargestFirst achieves optimal coloring on bridged cliques.

    Two 3-cliques (rows {0,1,2} via col 0, rows {3,4,5} via col 1)
    bridged by col 2 (rows 0 and 3).
    Chromatic number is 3.
    LargestFirst colors the high-degree bridge vertices (0, 3) first,
    allowing the cliques to share colors optimally.
    """
    rows = [0, 1, 2, 3, 4, 5, 0, 3]
    cols = [0, 0, 0, 1, 1, 1, 2, 2]
    sparsity = _make_pattern(rows, cols, (6, 3))

    colors, num_colors = color_rows(sparsity)

    assert _is_valid_row_coloring(sparsity, colors)
    assert num_colors == 3


@pytest.mark.coloring
def test_row_anti_diagonal():
    """Anti-diagonal: all rows are independent, 1 color suffices.

    From SMC small.jl.
    """
    sparsity = _from_dense(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
    )

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 1
    assert _is_valid_row_coloring(sparsity, colors)


@pytest.mark.coloring
def test_row_triangle():
    """Triangle pattern: complete bipartite-like, needs 3 colors.

    From SMC small.jl.
    """
    sparsity = _from_dense(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ]
    )

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 3
    assert _is_valid_row_coloring(sparsity, colors)


@pytest.mark.coloring
def test_row_smc_small():
    """SMC small.jl row coloring test matrix: [1 0 1; 0 1 0; 1 1 0].

    SMC gets 2 colors with LargestFirst.
    """
    sparsity = _from_dense(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
        ]
    )

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 2
    assert _is_valid_row_coloring(sparsity, colors)


@pytest.mark.coloring
def test_row_bidiagonal():
    """Upper bidiagonal 6x6: needs 2 colors.

    From SMC structured.jl.
    """
    sparsity = _from_dense(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 2
    assert _is_valid_row_coloring(sparsity, colors)


# =============================================================================
# Column coloring tests
# =============================================================================


@pytest.mark.coloring
def test_col_diagonal_one_color():
    """Diagonal matrix: all columns are independent, should use 1 color."""
    sparsity = _make_pattern([0, 1, 2, 3], [0, 1, 2, 3], (4, 4))

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 1
    assert len(colors) == 4
    assert np.all(colors == 0)
    assert _is_valid_col_coloring(sparsity, colors)


@pytest.mark.coloring
def test_col_dense_n_colors():
    """Dense matrix: every column conflicts with every other, needs n colors."""
    rows, cols = [], []
    for i in range(4):
        for j in range(4):
            rows.append(i)
            cols.append(j)
    sparsity = _make_pattern(rows, cols, (4, 4))

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 4
    assert len(set(colors)) == 4
    assert _is_valid_col_coloring(sparsity, colors)


@pytest.mark.coloring
def test_col_single_row():
    """Single row: all columns conflict."""
    sparsity = _make_pattern([0, 0, 0], [0, 1, 2], (1, 3))

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 3
    assert len(set(colors)) == 3
    assert _is_valid_col_coloring(sparsity, colors)


@pytest.mark.coloring
def test_col_single_column():
    """Single column: only one column, needs 1 color."""
    sparsity = _make_pattern([0, 1, 2], [0, 0, 0], (3, 1))

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 1
    assert len(colors) == 1
    assert colors[0] == 0


@pytest.mark.coloring
def test_col_block_diagonal():
    """Block diagonal: non-overlapping blocks can share colors."""
    rows = [0, 0, 1, 1, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 2, 3, 2, 3]
    sparsity = _make_pattern(rows, cols, (4, 4))

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 2
    assert _is_valid_col_coloring(sparsity, colors)


@pytest.mark.coloring
def test_col_empty():
    """Empty columns."""
    sparsity = _make_pattern([], [], (3, 0))

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 0
    assert len(colors) == 0


@pytest.mark.coloring
def test_col_tridiagonal():
    """Tridiagonal: column coloring also needs 2-3 colors."""
    rows = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]
    sparsity = _make_pattern(rows, cols, (4, 4))

    colors, num_colors = color_cols(sparsity)

    assert 2 <= num_colors <= 3
    assert _is_valid_col_coloring(sparsity, colors)


@pytest.mark.coloring
def test_col_anti_diagonal():
    """Anti-diagonal: all columns are independent, 1 color suffices.

    From SMC small.jl.
    """
    sparsity = _from_dense(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
    )

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 1
    assert _is_valid_col_coloring(sparsity, colors)


@pytest.mark.coloring
def test_col_triangle():
    """Triangle pattern: needs 3 column colors.

    From SMC small.jl.
    """
    sparsity = _from_dense(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ]
    )

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 3
    assert _is_valid_col_coloring(sparsity, colors)


@pytest.mark.coloring
def test_col_smc_small():
    """SMC small.jl column coloring test matrix: [1 0 1; 0 1 1; 1 0 0].

    SMC gets 2 colors with LargestFirst.
    """
    sparsity = _from_dense(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
        ]
    )

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 2
    assert _is_valid_col_coloring(sparsity, colors)


@pytest.mark.coloring
def test_col_bidiagonal():
    """Upper bidiagonal 6x6: needs 2 column colors.

    From SMC structured.jl.
    """
    sparsity = _from_dense(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    colors, num_colors = color_cols(sparsity)

    assert num_colors == 2
    assert _is_valid_col_coloring(sparsity, colors)


# =============================================================================
# Star coloring tests
# =============================================================================


@pytest.mark.coloring
def test_star_diagonal():
    """Diagonal Hessian: no off-diagonal entries, 1 color suffices."""
    sparsity = _make_pattern([0, 1, 2, 3], [0, 1, 2, 3], (4, 4))

    colors, num_colors = star_color(sparsity)

    assert num_colors == 1
    assert _is_valid_star_coloring(sparsity, colors)


@pytest.mark.coloring
def test_star_dense():
    """Dense symmetric pattern: star coloring is valid."""
    rows, cols = [], []
    for i in range(4):
        for j in range(4):
            rows.append(i)
            cols.append(j)
    sparsity = _make_pattern(rows, cols, (4, 4))

    colors, num_colors = star_color(sparsity)

    assert _is_valid_star_coloring(sparsity, colors)
    # Dense 4x4 needs at least 4 colors for distance-1
    assert num_colors >= 4


@pytest.mark.coloring
def test_star_tridiagonal():
    """Tridiagonal Hessian: star chromatic number is 3.

    Verified against SMC with LargestFirst.
    """
    rows = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]
    sparsity = _make_pattern(rows, cols, (4, 4))

    colors, num_colors = star_color(sparsity)

    assert _is_valid_star_coloring(sparsity, colors)
    assert num_colors == 3


@pytest.mark.coloring
def test_star_arrow_matrix():
    """Arrow matrix: star coloring needs only 2 colors.

    Row coloring needs n colors (all rows conflict via col 0),
    but the star graph has star chromatic number 2.
    Verified against SMC: star=2, row=10 for n=10.
    """
    sparsity = _make_arrow(10)

    star_colors, star_num = star_color(sparsity)
    row_colors, row_num = color_rows(sparsity)

    assert _is_valid_star_coloring(sparsity, star_colors)
    assert _is_valid_row_coloring(sparsity, row_colors)
    assert star_num == 2
    assert row_num == 10


@pytest.mark.coloring
def test_star_what_fig_41():
    """Figure 4.1 from Gebremedhin et al. (2005), "What Color Is Your Jacobian?"

    6x6 symmetric matrix.
    SMC gets 4 colors with LargestFirst + direct decompression.
    """
    sparsity = _from_dense(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 1],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
        ]
    )

    colors, num_colors = star_color(sparsity)

    assert _is_valid_star_coloring(sparsity, colors)
    assert num_colors <= 4


@pytest.mark.coloring
def test_star_what_fig_61():
    """Figure 6.1 from Gebremedhin et al. (2005).

    10x10 symmetric matrix.
    SMC gets 4 colors with LargestFirst + direct decompression.
    """
    sparsity = _from_dense(
        [
            [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
        ]
    )

    colors, num_colors = star_color(sparsity)

    assert _is_valid_star_coloring(sparsity, colors)
    assert num_colors <= 4


@pytest.mark.coloring
@pytest.mark.parametrize(
    ("half_bw", "expected_star"),
    [(1, 3), (2, 5), (3, 7), (5, 11)],
    ids=["tridiag", "pentadiag", "bw3", "bw5"],
)
def test_star_banded(half_bw: int, expected_star: int):
    """Banded matrices have star chromatic number 2*half_bw + 1.

    From SMC theory.jl.
    Verified against SMC: the formula is ``2 * floor(rho/2) + 1``
    where ``rho = 2 * half_bw``.
    """
    sparsity = _make_banded(20, half_bw)

    colors, num_colors = star_color(sparsity)

    assert _is_valid_star_coloring(sparsity, colors)
    assert num_colors == expected_star


@pytest.mark.coloring
def test_star_pentadiagonal_8x8():
    """Pentadiagonal 8x8: star coloring needs 5 colors.

    Verified against SMC.
    """
    sparsity = _make_banded(8, 2)

    colors, num_colors = star_color(sparsity)

    assert _is_valid_star_coloring(sparsity, colors)
    assert num_colors == 5


@pytest.mark.coloring
def test_star_not_square_raises():
    """Star coloring requires a square pattern."""
    sparsity = _make_pattern([0, 1], [0, 1], (3, 4))

    with pytest.raises(ValueError, match="square"):
        star_color(sparsity)


@pytest.mark.coloring
def test_star_empty():
    """Empty pattern."""
    sparsity = _make_pattern([], [], (0, 0))

    colors, num_colors = star_color(sparsity)

    assert num_colors == 0
    assert len(colors) == 0


# =============================================================================
# Unified color() tests
# =============================================================================


@pytest.mark.coloring
def test_color_returns_coloring_result():
    """color() returns a ColoredPattern with correct fields."""
    sparsity = _make_pattern([0, 1, 2, 3], [0, 1, 2, 3], (4, 4))

    result = color(sparsity)

    assert isinstance(result, ColoredPattern)
    assert isinstance(result.num_colors, int)
    assert result.partition in ("row", "column")
    assert len(result.colors) in (4, 4)  # m or n (both 4 here)


@pytest.mark.coloring
def test_color_auto_picks_column_for_tall():
    """Auto picks column coloring for tall-skinny patterns.

    With m=6 and n=2, column coloring needs at most 2 colors
    while row coloring may need up to 6.
    """
    # 6 rows, 2 columns — each row has one entry in each column
    rows = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    cols = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    sparsity = _make_pattern(rows, cols, (6, 2))

    result = color(sparsity)

    assert result.partition == "column"
    assert result.num_colors <= 2
    assert len(result.colors) == 2  # n=2


@pytest.mark.coloring
def test_color_auto_picks_row_for_wide():
    """Auto picks row coloring for wide patterns.

    With m=2 and n=6, row coloring needs at most 2 colors
    while column coloring may need up to 6.
    """
    # 2 rows, 6 columns — each column has entries in both rows
    rows = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    cols = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    sparsity = _make_pattern(rows, cols, (2, 6))

    result = color(sparsity)

    assert result.partition == "row"
    assert result.num_colors <= 2
    assert len(result.colors) == 2  # m=2


@pytest.mark.coloring
def test_color_force_row():
    """color(sparsity, "row") forces row partition."""
    sparsity = _make_pattern([0, 1, 2, 3], [0, 1, 2, 3], (4, 4))

    result = color(sparsity, "row")

    assert result.partition == "row"
    assert len(result.colors) == 4  # m=4
    assert _is_valid_row_coloring(sparsity, result.colors)


@pytest.mark.coloring
def test_color_force_column():
    """color(sparsity, "column") forces column partition."""
    sparsity = _make_pattern([0, 1, 2, 3], [0, 1, 2, 3], (4, 4))

    result = color(sparsity, "column")

    assert result.partition == "column"
    assert len(result.colors) == 4  # n=4
    assert _is_valid_col_coloring(sparsity, result.colors)


# =============================================================================
# jacobian_coloring / hessian_coloring tests
# =============================================================================


@pytest.mark.coloring
def test_jacobian_coloring_basic():
    """jacobian_coloring returns a correct ColoredPattern."""

    def f(x):
        return x**2

    result = jacobian_coloring(f, (4,))

    assert isinstance(result, ColoredPattern)
    assert result.sparsity.shape == (4, 4)
    assert result.num_colors == 1  # diagonal → 1 color


@pytest.mark.coloring
def test_jacobian_coloring_partition():
    """jacobian_coloring respects the partition argument."""

    def f(x):
        return x**2

    result_row = jacobian_coloring(f, (3,), partition="row")
    result_col = jacobian_coloring(f, (3,), partition="column")

    assert result_row.partition == "row"
    assert result_col.partition == "column"


@pytest.mark.coloring
def test_hessian_coloring_basic():
    """hessian_coloring returns a ColoredPattern with star coloring."""

    def f(x):
        return jnp.sum(x**2)

    result = hessian_coloring(f, (4,))

    assert isinstance(result, ColoredPattern)
    assert result.partition == "column"
    assert result.sparsity.shape == (4, 4)
    # Diagonal Hessian → 1 color
    assert result.num_colors == 1


@pytest.mark.coloring
def test_hessian_coloring_coupled():
    """hessian_coloring uses star coloring for a coupled function."""

    def f(x):
        return x[0] * x[1] + x[1] * x[2] + jnp.sum(x**2)

    result = hessian_coloring(f, (3,))

    assert isinstance(result, ColoredPattern)
    assert result.partition == "column"
    # Star coloring should use fewer colors than n for sparse Hessians
    assert result.num_colors <= 3


# =============================================================================
# _compressed_pattern tests
# =============================================================================


@pytest.mark.coloring
def test_compressed_pattern_column():
    """Column compressed pattern has shape (m, num_colors)."""
    sparsity = _from_dense(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
        ]
    )
    result = color(sparsity, partition="column")
    compressed = result._compressed_pattern()

    assert compressed.shape == (3, result.num_colors)
    # Every original row with a nonzero should appear in compressed
    dense_orig = sparsity.todense()
    dense_comp = compressed.todense()
    for i in range(3):
        has_orig = np.any(dense_orig[i] != 0)
        has_comp = np.any(dense_comp[i] != 0)
        assert has_orig == has_comp


@pytest.mark.coloring
def test_compressed_pattern_row():
    """Row compressed pattern has shape (num_colors, n)."""
    sparsity = _from_dense(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
        ]
    )
    result = color(sparsity, partition="row")
    compressed = result._compressed_pattern()

    assert compressed.shape == (result.num_colors, 3)
    # Every original column with a nonzero should appear in compressed
    dense_orig = sparsity.todense()
    dense_comp = compressed.todense()
    for j in range(3):
        has_orig = np.any(dense_orig[:, j] != 0)
        has_comp = np.any(dense_comp[:, j] != 0)
        assert has_orig == has_comp


# =============================================================================
# __str__ visualization tests
# =============================================================================


@pytest.mark.coloring
def test_str_column_contains_arrow():
    """Column partition __str__ contains → for side-by-side display."""
    sparsity = _from_dense(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
        ]
    )
    result = color(sparsity, partition="column")
    s = str(result)

    assert "→" in s
    assert "●" in s


@pytest.mark.coloring
def test_str_row_contains_downarrow():
    """Row partition __str__ contains ↓ for stacked display."""
    sparsity = _from_dense(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
        ]
    )
    result = color(sparsity, partition="row")
    s = str(result)

    assert "↓" in s
    assert "●" in s


# =============================================================================
# hessian with colored_pattern tests
# =============================================================================


@pytest.mark.hessian
def test_hessian_with_colored_pattern():
    """hessian works with a pre-computed ColoredPattern."""

    def f(x):
        return jnp.sum(x**2) + x[0] * x[1]

    x = np.array([1.0, 2.0, 3.0])
    cp = hessian_coloring(f, x.shape)
    result = hessian(f, x, colored_pattern=cp).todense()
    expected = jax.hessian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)

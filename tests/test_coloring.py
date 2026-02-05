"""Tests for row-wise graph coloring algorithm."""

from collections import defaultdict

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from detex import color_rows


def _make_bcoo(rows: list[int], cols: list[int], shape: tuple[int, int]) -> BCOO:
    """Helper to create BCOO from row/col lists."""
    if len(rows) == 0:
        indices = jnp.zeros((0, 2), dtype=jnp.int32)
    else:
        indices = jnp.array(
            [[r, c] for r, c in zip(rows, cols, strict=True)], dtype=jnp.int32
        )
    data = jnp.ones(len(rows), dtype=jnp.int8)
    return BCOO((data, indices), shape=shape)


def _is_valid_coloring(sparsity: BCOO, colors: np.ndarray) -> bool:
    """Check that no column has two rows with the same color."""
    # Group rows by column
    col_to_rows: dict[int, list[int]] = defaultdict(list)
    indices = np.asarray(sparsity.indices)
    for row, col in indices:
        col_to_rows[int(col)].append(int(row))

    # Check each column
    for rows_in_col in col_to_rows.values():
        colors_in_col = colors[rows_in_col]
        if len(colors_in_col) != len(set(colors_in_col)):
            return False
    return True


# =============================================================================
# Basic coloring tests
# =============================================================================


@pytest.mark.coloring
def test_diagonal_one_color():
    """Diagonal matrix: all rows are independent, should use 1 color."""
    sparsity = _make_bcoo([0, 1, 2, 3], [0, 1, 2, 3], (4, 4))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 1
    assert len(colors) == 4
    assert np.all(colors == 0)
    assert _is_valid_coloring(sparsity, colors)


@pytest.mark.coloring
def test_dense_m_colors():
    """Dense matrix: every row conflicts with every other, needs m colors."""
    rows, cols = [], []
    for i in range(4):
        for j in range(4):
            rows.append(i)
            cols.append(j)
    sparsity = _make_bcoo(rows, cols, (4, 4))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 4
    assert len(colors) == 4
    assert len(set(colors)) == 4  # All different colors
    assert _is_valid_coloring(sparsity, colors)


@pytest.mark.coloring
def test_block_diagonal():
    """Block diagonal: non-overlapping blocks can share colors."""
    # Two 2x2 blocks
    rows = [0, 0, 1, 1, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 2, 3, 2, 3]
    sparsity = _make_bcoo(rows, cols, (4, 4))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 2
    assert _is_valid_coloring(sparsity, colors)
    # Rows 0,1 conflict; rows 2,3 conflict; but 0,2 and 1,3 don't
    assert colors[0] != colors[1]
    assert colors[2] != colors[3]


@pytest.mark.coloring
def test_tridiagonal():
    """Tridiagonal matrix: needs 2-3 colors depending on structure."""
    # 4x4 tridiagonal
    rows = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]
    sparsity = _make_bcoo(rows, cols, (4, 4))

    colors, num_colors = color_rows(sparsity)

    # Tridiagonal needs at most 3 colors (greedy may use 2-3)
    assert 2 <= num_colors <= 3
    assert _is_valid_coloring(sparsity, colors)


@pytest.mark.coloring
def test_single_row():
    """Single row matrix."""
    sparsity = _make_bcoo([0, 0, 0], [0, 1, 2], (1, 3))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 1
    assert len(colors) == 1
    assert colors[0] == 0


@pytest.mark.coloring
def test_single_column():
    """Single column matrix: all rows conflict."""
    sparsity = _make_bcoo([0, 1, 2], [0, 0, 0], (3, 1))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 3
    assert len(set(colors)) == 3
    assert _is_valid_coloring(sparsity, colors)


@pytest.mark.coloring
def test_empty_matrix():
    """Empty matrix (0 rows)."""
    sparsity = _make_bcoo([], [], (0, 3))

    colors, num_colors = color_rows(sparsity)

    assert num_colors == 0
    assert len(colors) == 0


@pytest.mark.coloring
def test_zero_matrix():
    """Matrix with no non-zeros: all rows independent."""
    sparsity = _make_bcoo([], [], (3, 3))

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
    sparsity = _make_bcoo(rows, cols, (4, 4))

    colors, num_colors = color_rows(sparsity)

    assert _is_valid_coloring(sparsity, colors)
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
    sparsity = _make_bcoo(rows, cols, (4, 4))

    colors, num_colors = color_rows(sparsity)

    assert _is_valid_coloring(sparsity, colors)
    # Even rows share cols 0,2; odd rows share cols 1,3
    # So we need 2 colors
    assert num_colors == 2

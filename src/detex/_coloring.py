"""Row-wise graph coloring for sparse Jacobian computation.

Greedy coloring assigns colors to rows such that rows sharing a non-zero column
get different colors. Same-colored rows are structurally orthogonal and can be
evaluated together in a single VJP.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix


def _build_row_conflict_sets(sparsity: coo_matrix) -> list[set[int]]:
    """Build conflict graph: rows conflict if they share a non-zero column.

    Uses CSC (compressed sparse column) format for efficient column iteration.
    For each column, all rows with non-zeros in that column conflict with each other.

    Args:
        sparsity: Sparse boolean matrix of shape (m, n)

    Returns:
        List of sets where conflicts[i] contains all rows that conflict with row i
    """
    m = sparsity.shape[0]
    csc = sparsity.tocsc()

    conflicts: list[set[int]] = [set() for _ in range(m)]

    # For each column, mark all pairs of rows as conflicting
    for col in range(csc.shape[1]):
        start, end = csc.indptr[col], csc.indptr[col + 1]
        rows_in_col = csc.indices[start:end]

        # All rows in this column conflict with each other
        for i, row_i in enumerate(rows_in_col):
            for row_j in rows_in_col[i + 1 :]:
                conflicts[row_i].add(row_j)
                conflicts[row_j].add(row_i)

    return conflicts


def color_rows(sparsity: coo_matrix) -> tuple[NDArray[np.int32], int]:
    """Greedy row-wise coloring for sparse Jacobian computation.

    Assigns colors to rows such that no two rows sharing a non-zero column
    have the same color. This enables computing multiple Jacobian rows in
    a single VJP by using a combined seed vector.

    The algorithm is greedy: for each row in order, assign the smallest color
    not used by any conflicting row.

    Args:
        sparsity: Sparse boolean matrix of shape (m, n) representing the
            Jacobian sparsity pattern

    Returns:
        Tuple of (colors, num_colors) where:
        - colors: Array of shape (m,) with color assignment for each row
        - num_colors: Total number of colors used
    """
    m = sparsity.shape[0]

    if m == 0:
        return np.array([], dtype=np.int32), 0

    conflicts = _build_row_conflict_sets(sparsity)

    colors = np.full(m, -1, dtype=np.int32)
    num_colors = 0

    for row in range(m):
        # Find colors used by conflicting rows
        used_colors: set[int] = set()
        for neighbor in conflicts[row]:
            if colors[neighbor] >= 0:
                used_colors.add(colors[neighbor])

        # Assign smallest unused color
        color = 0
        while color in used_colors:
            color += 1

        colors[row] = color
        num_colors = max(num_colors, color + 1)

    return colors, num_colors

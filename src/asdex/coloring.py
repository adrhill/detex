"""Graph coloring for sparse Jacobian and Hessian computation.

Greedy coloring assigns colors to vertices such that conflicting vertices
get different colors.
Row coloring enables computing multiple Jacobian rows in a single VJP.
Column coloring enables computing multiple Jacobian columns in a single JVP.
Star coloring exploits Hessian symmetry for fewer colors.

Algorithms adapted from SparseMatrixColorings.jl (MIT license)
Copyright (c) 2024 Guillaume Dalle, Alexis Montoison, and contributors
https://github.com/gdalle/SparseMatrixColorings.jl
See also: Dalle & Montoison (2025), https://arxiv.org/abs/2505.07308
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from asdex.pattern import SparsityPattern


@dataclass(frozen=True)
class ColoringResult:
    """Result of a graph coloring for sparse differentiation.

    Attributes:
        colors: Color assignment array.
            Shape ``(m,)`` for row partition, ``(n,)`` for column partition.
        num_colors: Total number of colors used.
        partition: Whether rows or columns were colored.
    """

    colors: NDArray[np.int32]
    num_colors: int
    partition: Literal["row", "column"]


def _greedy_color(
    num_vertices: int,
    conflicts: list[set[int]],
) -> tuple[NDArray[np.int32], int]:
    """Greedy graph coloring with LargestFirst vertex ordering.

    Vertices are sorted by decreasing degree (number of conflicts)
    before the greedy loop.
    For each vertex in order,
    assign the smallest color not used by any conflicting vertex.

    Args:
        num_vertices: Number of vertices to color
        conflicts: List of sets where conflicts[v] contains
            all vertices that conflict with vertex v

    Returns:
        Tuple of (colors, num_colors) where:
        - colors: Array of shape (num_vertices,) with color assignments
        - num_colors: Total number of colors used
    """
    if num_vertices == 0:
        return np.array([], dtype=np.int32), 0

    # LargestFirst ordering: sort vertices by decreasing degree
    order = sorted(range(num_vertices), key=lambda v: len(conflicts[v]), reverse=True)

    colors = np.full(num_vertices, -1, dtype=np.int32)
    num_colors = 0

    for v in order:
        # Find colors used by conflicting vertices
        used_colors: set[int] = set()
        for neighbor in conflicts[v]:
            if colors[neighbor] >= 0:
                used_colors.add(colors[neighbor])

        # Assign smallest unused color
        color = 0
        while color in used_colors:
            color += 1

        colors[v] = color
        num_colors = max(num_colors, color + 1)

    return colors, num_colors


def color(
    sparsity: SparsityPattern,
    partition: Literal["row", "column", "auto"] = "auto",
) -> ColoringResult:
    """Color a sparsity pattern for sparse Jacobian computation.

    Assigns colors so that same-colored rows (or columns) can be
    computed together in a single VJP (or JVP).

    Args:
        sparsity: Sparsity pattern of shape (m, n).
        partition: Which partition to color.
            ``"row"`` colors rows (uses VJPs),
            ``"column"`` colors columns (uses JVPs),
            ``"auto"`` picks whichever needs fewer colors
            (ties go to column coloring since JVPs are cheaper).

    Returns:
        A :class:`ColoringResult` with the color assignment,
        number of colors, and which partition was used.
    """
    if partition == "row":
        colors_arr, num = color_rows(sparsity)
        return ColoringResult(colors=colors_arr, num_colors=num, partition="row")

    if partition == "column":
        colors_arr, num = color_cols(sparsity)
        return ColoringResult(colors=colors_arr, num_colors=num, partition="column")

    # Auto: pick whichever uses fewer colors.
    # Ties go to column coloring (JVPs are cheaper than VJPs).
    row_colors, num_row = color_rows(sparsity)
    col_colors, num_col = color_cols(sparsity)

    if num_col <= num_row:
        return ColoringResult(colors=col_colors, num_colors=num_col, partition="column")
    else:
        return ColoringResult(colors=row_colors, num_colors=num_row, partition="row")


def _build_row_conflict_sets(sparsity: SparsityPattern) -> list[set[int]]:
    """Build conflict graph: rows conflict if they share a non-zero column.

    For each column, all rows with non-zeros in that column conflict with each other.

    Args:
        sparsity: SparsityPattern of shape (m, n)

    Returns:
        List of sets where conflicts[i] contains all rows that conflict with row i
    """
    m = sparsity.m
    conflicts: list[set[int]] = [set() for _ in range(m)]

    # Use cached col_to_rows mapping
    col_to_rows = sparsity.col_to_rows

    # For each column, mark all pairs of rows as conflicting
    for rows_in_col in col_to_rows.values():
        for i, row_i in enumerate(rows_in_col):
            for row_j in rows_in_col[i + 1 :]:
                conflicts[row_i].add(row_j)
                conflicts[row_j].add(row_i)

    return conflicts


def _build_col_conflict_sets(sparsity: SparsityPattern) -> list[set[int]]:
    """Build conflict graph: columns conflict if they share a non-zero row.

    For each row, all columns with non-zeros in that row conflict with each other.

    Args:
        sparsity: SparsityPattern of shape (m, n)

    Returns:
        List of sets where conflicts[j] contains all columns that conflict with column j
    """
    n = sparsity.n
    conflicts: list[set[int]] = [set() for _ in range(n)]

    # Use cached row_to_cols mapping
    row_to_cols = sparsity.row_to_cols

    # For each row, mark all pairs of columns as conflicting
    for cols_in_row in row_to_cols.values():
        for i, col_i in enumerate(cols_in_row):
            for col_j in cols_in_row[i + 1 :]:
                conflicts[col_i].add(col_j)
                conflicts[col_j].add(col_i)

    return conflicts


def color_rows(sparsity: SparsityPattern) -> tuple[NDArray[np.int32], int]:
    """Greedy row-wise coloring for sparse Jacobian computation.

    Assigns colors to rows such that no two rows sharing a non-zero column
    have the same color.
    This enables computing multiple Jacobian rows in a single VJP
    by using a combined seed vector.

    Uses LargestFirst vertex ordering for fewer colors.

    Args:
        sparsity: SparsityPattern of shape (m, n) representing the
            Jacobian sparsity pattern

    Returns:
        Tuple of (colors, num_colors) where:
        - colors: Array of shape (m,) with color assignment for each row
        - num_colors: Total number of colors used
    """
    m = sparsity.m

    if m == 0:
        return np.array([], dtype=np.int32), 0

    conflicts = _build_row_conflict_sets(sparsity)
    return _greedy_color(m, conflicts)


def color_cols(sparsity: SparsityPattern) -> tuple[NDArray[np.int32], int]:
    """Greedy column-wise coloring for sparse Jacobian computation.

    Assigns colors to columns such that no two columns sharing a non-zero row
    have the same color.
    This enables computing multiple Jacobian columns in a single JVP
    by using a combined tangent vector.

    Uses LargestFirst vertex ordering for fewer colors.

    Args:
        sparsity: SparsityPattern of shape (m, n) representing the
            Jacobian sparsity pattern

    Returns:
        Tuple of (colors, num_colors) where:
        - colors: Array of shape (n,) with color assignment for each column
        - num_colors: Total number of colors used
    """
    n = sparsity.n

    if n == 0:
        return np.array([], dtype=np.int32), 0

    conflicts = _build_col_conflict_sets(sparsity)
    return _greedy_color(n, conflicts)


def star_color(sparsity: SparsityPattern) -> tuple[NDArray[np.int32], int]:
    """Star coloring for symmetric sparse Hessian computation.

    A star coloring is a distance-1 coloring with the additional constraint
    that every path on 4 vertices uses at least 3 colors.
    This enables symmetric decompression using fewer colors than row coloring.

    Requires a square sparsity pattern (Hessians are always square).
    Uses LargestFirst vertex ordering.

    Based on Gebremedhin et al. (2005), "What Color Is Your Jacobian?"

    Args:
        sparsity: SparsityPattern of shape (n, n) representing the
            symmetric Hessian sparsity pattern

    Returns:
        Tuple of (colors, num_colors) where:
        - colors: Array of shape (n,) with color assignment for each row/column
        - num_colors: Total number of colors used

    Raises:
        ValueError: If pattern is not square
    """
    if sparsity.m != sparsity.n:
        msg = f"Star coloring requires a square pattern, got shape {sparsity.shape}"
        raise ValueError(msg)

    n = sparsity.n

    if n == 0:
        return np.array([], dtype=np.int32), 0

    # Build adjacency graph (undirected, exclude diagonal)
    adj: list[set[int]] = [set() for _ in range(n)]
    for i, j in zip(sparsity.rows, sparsity.cols, strict=True):
        i, j = int(i), int(j)
        if i != j:
            adj[i].add(j)
            adj[j].add(i)

    # LargestFirst ordering
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)

    colors = np.full(n, -1, dtype=np.int32)
    num_colors = 0

    for v in order:
        # Forbidden colors from distance-1 constraint
        forbidden: set[int] = set()
        for w in adj[v]:
            if colors[w] >= 0:
                forbidden.add(colors[w])

        # Star constraint: for each colored neighbor w of v,
        # check w's other colored neighbors u.
        # If colors[w] == colors[u] would create a 2-colored path v-w-u-?,
        # we need to forbid the color of any neighbor of u that has color[w]'s color.
        # More precisely: for neighbor w (colored), for neighbor u of w (u != v, colored),
        # if colors[u] is not in forbidden yet, check if u has a neighbor x (x != w)
        # with colors[x] == colors[w]. If so, colors[u] is forbidden for v because
        # x-u-w-v would be a 4-path with only 2 colors.
        for w in adj[v]:
            if colors[w] < 0:
                continue
            for u in adj[w]:
                if u == v or colors[u] < 0:
                    continue
                if colors[u] in forbidden:
                    continue
                # Check if u has a neighbor x != w with colors[x] == colors[w]
                for x in adj[u]:
                    if x != w and colors[x] == colors[w]:
                        # Path x-u-w-v would use only colors[w] and colors[u]
                        # if we assign colors[u] to v. Forbid colors[u].
                        forbidden.add(colors[u])
                        break

        # Assign smallest non-forbidden color
        color = 0
        while color in forbidden:
            color += 1

        colors[v] = color
        num_colors = max(num_colors, color + 1)

    return colors, num_colors

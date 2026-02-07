"""Pattern data structures for the detection->coloring->decompression pipeline.

Pretty-printing adapted from SparseArrays.jl (MIT license)
Copyright (c) 2018-2024 SparseArrays.jl contributors: https://github.com/JuliaSparse/SparseArrays.jl/contributors
https://github.com/JuliaSparse/SparseArrays.jl/
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from jax.experimental.sparse import BCOO


@dataclass(frozen=True)
class SparsityPattern:
    """Sparse matrix pattern storing only structural information (no values).

    Optimized for the sparsity detection -> coloring -> decompression pipeline.
    Stores row and column indices separately for efficient access.

    Attributes:
        rows: Row indices of non-zero entries, shape (nnz,)
        cols: Column indices of non-zero entries, shape (nnz,)
        shape: Matrix dimensions (m, n)
    """

    rows: NDArray[np.int32]
    cols: NDArray[np.int32]
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        """Validate inputs."""
        if len(self.rows) != len(self.cols):
            msg = f"rows and cols must have same length, got {len(self.rows)} and {len(self.cols)}"
            raise ValueError(msg)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def nse(self) -> int:
        """Number of stored elements."""
        return len(self.rows)

    @property
    def nnz(self) -> int:
        """Number of non-zero elements (alias for nse)."""
        return self.nse

    @property
    def m(self) -> int:
        """Number of rows."""
        return self.shape[0]

    @property
    def n(self) -> int:
        """Number of columns."""
        return self.shape[1]

    @property
    def density(self) -> float:
        """Fraction of non-zero entries."""
        total = self.m * self.n
        return self.nse / total if total > 0 else 0.0

    @cached_property
    def col_to_rows(self) -> dict[int, list[int]]:
        """Mapping from column index to list of row indices with non-zeros in that column.

        Used by the coloring algorithm to build the row conflict graph.
        """
        result: dict[int, list[int]] = defaultdict(list)
        for row, col in zip(self.rows, self.cols, strict=True):
            result[int(col)].append(int(row))
        return dict(result)

    @cached_property
    def row_to_cols(self) -> dict[int, list[int]]:
        """Mapping from row index to list of column indices with non-zeros in that row.

        Used by the coloring algorithm to build the column conflict graph.
        """
        result: dict[int, list[int]] = defaultdict(list)
        for row, col in zip(self.rows, self.cols, strict=True):
            result[int(row)].append(int(col))
        return dict(result)

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def from_coordinates(
        cls,
        rows: NDArray[np.int32] | list[int],
        cols: NDArray[np.int32] | list[int],
        shape: tuple[int, int],
    ) -> SparsityPattern:
        """Create pattern from row and column index arrays.

        Args:
            rows: Row indices of non-zero entries
            cols: Column indices of non-zero entries
            shape: Matrix dimensions (m, n)

        Returns:
            SparsityPattern instance
        """
        return cls(
            rows=np.asarray(rows, dtype=np.int32),
            cols=np.asarray(cols, dtype=np.int32),
            shape=shape,
        )

    @classmethod
    def from_bcoo(cls, bcoo: BCOO) -> SparsityPattern:
        """Create pattern from JAX BCOO sparse matrix.

        Args:
            bcoo: JAX BCOO sparse matrix

        Returns:
            SparsityPattern instance
        """
        indices = np.asarray(bcoo.indices)
        shape = (bcoo.shape[0], bcoo.shape[1])
        if indices.size == 0:
            return cls(
                rows=np.array([], dtype=np.int32),
                cols=np.array([], dtype=np.int32),
                shape=shape,
            )
        return cls(
            rows=indices[:, 0].astype(np.int32),
            cols=indices[:, 1].astype(np.int32),
            shape=shape,
        )

    @classmethod
    def from_dense(cls, dense: NDArray) -> SparsityPattern:
        """Create pattern from dense boolean/numeric matrix.

        Args:
            dense: 2D array where non-zero entries indicate pattern positions

        Returns:
            SparsityPattern instance
        """
        dense = np.asarray(dense)
        rows, cols = np.nonzero(dense)
        return cls(
            rows=rows.astype(np.int32),
            cols=cols.astype(np.int32),
            shape=(dense.shape[0], dense.shape[1]),
        )

    # -------------------------------------------------------------------------
    # Conversion methods
    # -------------------------------------------------------------------------

    @cached_property
    def _bcoo_indices(self) -> jnp.ndarray:
        """BCOO index array of shape ``(nse, 2)``, cached for reuse."""
        if self.nse == 0:
            return jnp.zeros((0, 2), dtype=jnp.int32)
        return jnp.stack([self.rows, self.cols], axis=1)

    def to_bcoo(self, data: jnp.ndarray | None = None) -> BCOO:
        """Convert to JAX BCOO sparse matrix.

        Args:
            data: Optional data values. If None, uses all 1s.

        Returns:
            JAX BCOO sparse matrix
        """
        from jax.experimental.sparse import BCOO

        indices = self._bcoo_indices
        if data is None:
            if self.nse == 0:
                data = jnp.array([])
            else:
                data = jnp.ones(self.nse, dtype=jnp.int8)
        return BCOO((data, indices), shape=self.shape)

    def todense(self) -> NDArray:
        """Convert to dense numpy array (1s at pattern positions).

        Returns:
            Dense boolean array of shape (m, n)
        """
        result = np.zeros(self.shape, dtype=np.int8)
        if self.nse > 0:
            result[self.rows, self.cols] = 1
        return result

    def astype(self, dtype: type) -> NDArray:
        """Return dense array with specified dtype.

        For compatibility with existing test patterns like `.todense().astype(int)`.

        Args:
            dtype: Target numpy dtype

        Returns:
            Dense array of specified dtype
        """
        return self.todense().astype(dtype)

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    # Thresholds for switching from dot display to braille (Julia-style heuristics)
    _SMALL_ROWS = 16  # Max rows to show with dot display
    _SMALL_COLS = 40  # Max cols to show with dot display

    def _render_dots(self) -> str:
        """Render small matrix using dots and bullets (Julia-style).

        Uses '⋅' for zeros and '●' for non-zeros.

        Returns:
            String containing dot visualization
        """
        if self.m == 0 or self.n == 0:
            return "(empty)"

        dense = self.todense()
        lines = []
        for i in range(self.m):
            row_chars = []
            for j in range(self.n):
                row_chars.append("●" if dense[i, j] else "⋅")
            lines.append(" ".join(row_chars))
        return "\n".join(lines)

    def _render_braille(self, max_height: int = 20, max_width: int = 40) -> str:
        """Render sparsity pattern using Unicode braille characters.

        Each braille character represents a 4x2 block of the matrix.
        Dots are lit where the pattern has non-zero entries.
        Large matrices are downsampled by linearly interpolating each
        non-zero position to the output grid,
        following Julia's SparseArrays approach.

        Braille dot bits indexed by ``(col_offset % 2) * 4 + (row_offset % 4)``::

            [0,0]=0x01  [0,1]=0x08
            [1,0]=0x02  [1,1]=0x10
            [2,0]=0x04  [2,1]=0x20
            [3,0]=0x40  [3,1]=0x80

        Args:
            max_height: Maximum number of braille characters vertically
            max_width: Maximum number of braille characters horizontally

        Returns:
            String containing braille visualization
        """
        if self.m == 0 or self.n == 0:
            return "(empty)"

        # Target size in dot space (each braille char is 4 rows × 2 cols)
        scale_height = min(self.m, max_height * 4)
        scale_width = min(self.n, max_width * 2)

        # Output braille grid dimensions
        out_rows = (scale_height - 1) // 4 + 1
        out_cols = (scale_width - 1) // 2 + 1

        # Braille dot bits: index = (col_offset % 2) * 4 + (row_offset % 4)
        braille_bits = [0x01, 0x02, 0x04, 0x40, 0x08, 0x10, 0x20, 0x80]

        grid = [[0] * out_cols for _ in range(out_rows)]

        # Scale each non-zero to the output grid via linear interpolation
        row_denom = max(self.m - 1, 1)
        col_denom = max(self.n - 1, 1)
        for i, j in zip(self.rows, self.cols, strict=True):
            si = round(int(i) * (scale_height - 1) / row_denom)
            sj = round(int(j) * (scale_width - 1) / col_denom)
            grid[si // 4][sj // 2] |= braille_bits[(sj % 2) * 4 + (si % 4)]

        lines = []
        for row in grid:
            lines.append("".join(chr(0x2800 + bits) for bits in row))
        return "\n".join(lines)

    def _render(self) -> str:
        """Return visualization string without header.

        Uses dot display (●/⋅) for small matrices, braille for large ones.
        Follows Julia's SparseArrays display heuristics.
        """
        if self.m <= self._SMALL_ROWS and self.n <= self._SMALL_COLS:
            return self._render_dots()

        braille = self._render_braille()
        braille_lines = braille.split("\n")
        if braille_lines and braille_lines[0] != "(empty)":
            n_lines = len(braille_lines)
            bordered = []
            for i, line in enumerate(braille_lines):
                if i == 0:
                    bordered.append("⎡" + line + "⎤")
                elif i == n_lines - 1:
                    bordered.append("⎣" + line + "⎦")
                else:
                    bordered.append("⎢" + line + "⎥")
            return "\n".join(bordered)
        return braille

    def __str__(self) -> str:
        """Return string representation with visualization.

        Uses dot display (●/⋅) for small matrices, braille for large ones.
        Follows Julia's SparseArrays display heuristics.
        """
        header = f"SparsityPattern({self.m}×{self.n}, nnz={self.nse}, sparsity={1 - self.density:.1%})"
        return f"{header}\n{self._render()}"

    def __repr__(self) -> str:
        """Return compact representation."""
        return f"SparsityPattern(shape={self.shape}, nnz={self.nse})"


@dataclass(frozen=True, repr=False)
class ColoredPattern:
    """Result of a graph coloring for sparse differentiation.

    Attributes:
        sparsity: The sparsity pattern that was colored.
        colors: Color assignment array.
            Shape ``(m,)`` for ``"VJP"`` mode,
            ``(n,)`` for ``"JVP"`` and ``"HVP"`` modes.
        num_colors: Total number of colors used.
        mode: The AD primitive used per color.
            ``"VJP"`` for row-colored Jacobians,
            ``"JVP"`` for column-colored Jacobians,
            ``"HVP"`` for star-colored Hessians.
    """

    sparsity: SparsityPattern
    colors: NDArray[np.int32]
    num_colors: int
    mode: Literal["JVP", "VJP", "HVP"]

    @property
    def _compresses_columns(self) -> bool:
        """Whether coloring compresses columns (JVP/HVP) or rows (VJP)."""
        return self.mode in ("JVP", "HVP")

    # -------------------------------------------------------------------------
    # Cached arrays for fast decompression
    # -------------------------------------------------------------------------

    @cached_property
    def _extraction_indices(
        self,
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Indices for extracting sparse entries from compressed gradient rows.

        Returns ``(color_idx, elem_idx)`` such that for a compressed matrix
        ``C`` of shape ``(num_colors, dim)``::

            data = C[color_idx, elem_idx]

        gives the nnz values in sparsity-pattern order.

        For VJP: ``color_idx = colors[rows]``, ``elem_idx = cols``.
        For JVP: ``color_idx = colors[cols]``, ``elem_idx = rows``.
        For HVP: delegates to :meth:`_star_extraction_indices`.
        """
        if self.mode == "HVP":
            return self._star_extraction_indices

        rows = self.sparsity.rows
        cols = self.sparsity.cols

        if self.mode == "VJP":
            color_idx = self.colors[rows].astype(np.intp)
            elem_idx = cols.astype(np.intp)
        else:  # JVP
            color_idx = self.colors[cols].astype(np.intp)
            elem_idx = rows.astype(np.intp)

        return color_idx, elem_idx

    @cached_property
    def _star_extraction_indices(
        self,
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Pre-compute HVP extraction indices with star-coloring direction choice.

        For each nonzero ``(i, j)``:
        - diagonal (``i == j``): use ``compressed[colors[i]][i]``
        - off-diagonal: use ``compressed[colors[i]][j]`` if ``colors[i]``
          is unique among column ``j``'s neighbors;
          otherwise ``compressed[colors[j]][i]``.
        """
        rows = self.sparsity.rows
        cols = self.sparsity.cols
        col_to_rows = self.sparsity.col_to_rows

        color_idx = np.empty(len(rows), dtype=np.intp)
        elem_idx = np.empty(len(rows), dtype=np.intp)

        for k, (i, j) in enumerate(zip(rows, cols, strict=True)):
            i, j = int(i), int(j)
            if i == j:
                color_idx[k] = self.colors[i]
                elem_idx[k] = i
            else:
                color_i = self.colors[i]
                unique = True
                for r in col_to_rows.get(j, []):
                    if r != i and self.colors[r] == color_i:
                        unique = False
                        break
                if unique:
                    color_idx[k] = color_i
                    elem_idx[k] = j
                else:
                    color_idx[k] = self.colors[j]
                    elem_idx[k] = i

        return color_idx, elem_idx

    @cached_property
    def _seed_matrix(self) -> NDArray[np.bool_]:
        """Boolean seed matrix of shape ``(num_colors, dim)``.

        Row ``c`` is the mask ``colors == c``,
        used as the seed/tangent vector for the ``c``-th AD evaluation.
        """
        dim = self.sparsity.m if self.mode == "VJP" else self.sparsity.n
        seeds = np.zeros((self.num_colors, dim), dtype=np.bool_)
        for c in range(self.num_colors):
            seeds[c] = self.colors == c
        return seeds

    def __repr__(self) -> str:
        """Return compact representation."""
        sp = self.sparsity
        m, n = sp.shape
        c = self.num_colors
        return (
            f"ColoredPattern({m}×{n}, nnz={sp.nse}, sparsity={1 - sp.density:.1%}, "
            f"{self.mode}, {c} {'color' if c == 1 else 'colors'})"
        )

    def _compressed_pattern(self) -> SparsityPattern:
        """Build the compressed sparsity pattern after coloring.

        For column compression (JVP/HVP, shape ``(m, num_colors)``):
        entry ``(i, c)`` is present iff any column ``j``
        with ``colors[j] == c`` has a nonzero at ``(i, j)``.

        For row compression (VJP, shape ``(num_colors, n)``):
        entry ``(c, j)`` is present iff any row ``i``
        with ``colors[i] == c`` has a nonzero at ``(i, j)``.
        """
        comp_rows: list[int] = []
        comp_cols: list[int] = []

        if self._compresses_columns:
            # Compress columns: (m, n) → (m, num_colors)
            seen: set[tuple[int, int]] = set()
            for i, j in zip(self.sparsity.rows, self.sparsity.cols, strict=True):
                c = int(self.colors[j])
                entry = (int(i), c)
                if entry not in seen:
                    seen.add(entry)
                    comp_rows.append(entry[0])
                    comp_cols.append(entry[1])
            shape = (self.sparsity.m, self.num_colors)
        else:
            # Compress rows: (m, n) → (num_colors, n)
            seen = set()
            for i, j in zip(self.sparsity.rows, self.sparsity.cols, strict=True):
                c = int(self.colors[i])
                entry = (c, int(j))
                if entry not in seen:
                    seen.add(entry)
                    comp_rows.append(entry[0])
                    comp_cols.append(entry[1])
            shape = (self.num_colors, self.sparsity.n)

        return SparsityPattern.from_coordinates(comp_rows, comp_cols, shape)

    def __str__(self) -> str:
        """Return string with AD savings summary and visualization.

        Column compression (JVP/HVP) shows side-by-side with ``→``.
        Row compression (VJP) shows stacked with ``↓``.
        """
        m, n = self.sparsity.shape
        c = self.num_colors
        s = "" if c == 1 else "s"

        def _plural(count: int, word: str) -> str:
            return f"{count} {word}" if count == 1 else f"{count} {word}s"

        if self.mode == "HVP":
            instead = f"instead of {_plural(n, 'HVP')}"
        else:
            instead = f"instead of {_plural(m, 'VJP')} or {_plural(n, 'JVP')}"
        header = f"{repr(self)}\n  {c} {self.mode}{s} ({instead})"

        compressed = self._compressed_pattern()
        left_lines = self.sparsity._render().split("\n")
        right_lines = compressed._render().split("\n")

        if self._compresses_columns:
            viz = _render_side_by_side(left_lines, right_lines)
        else:
            viz = _render_stacked(left_lines, right_lines)

        return f"{header}\n{viz}"


def _render_side_by_side(left_lines: list[str], right_lines: list[str]) -> str:
    """Join two visualizations side-by-side with ``→`` on the middle line."""
    max_left = max((len(line) for line in left_lines), default=0)
    n_lines = max(len(left_lines), len(right_lines))
    mid = n_lines // 2

    result = []
    for i in range(n_lines):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""
        sep = " → " if i == mid else "   "
        result.append(f"{left:<{max_left}}{sep}{right}")
    return "\n".join(result)


def _render_stacked(top_lines: list[str], bottom_lines: list[str]) -> str:
    """Join two visualizations stacked with centered ``↓`` between them."""
    top_width = max((len(line) for line in top_lines), default=0)
    bottom_width = max((len(line) for line in bottom_lines), default=0)
    full_width = max(top_width, bottom_width)

    result = list(top_lines)
    arrow = "↓"
    pad = full_width // 2
    result.append(" " * pad + arrow)
    result.extend(bottom_lines)
    return "\n".join(result)

"""SparsityPattern data structure optimized for the detection->coloring->decompression pipeline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

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

        Used by the coloring algorithm to build the conflict graph.
        """
        result: dict[int, list[int]] = defaultdict(list)
        for row, col in zip(self.rows, self.cols, strict=True):
            result[int(col)].append(int(row))
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

    def to_bcoo(self, data: jnp.ndarray | None = None) -> BCOO:
        """Convert to JAX BCOO sparse matrix.

        Args:
            data: Optional data values. If None, uses all 1s.

        Returns:
            JAX BCOO sparse matrix
        """
        from jax.experimental.sparse import BCOO

        if self.nse == 0:
            indices = jnp.zeros((0, 2), dtype=jnp.int32)
            if data is None:
                data = jnp.array([])
            return BCOO((data, indices), shape=self.shape)

        indices = jnp.stack([self.rows, self.cols], axis=1)
        if data is None:
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

        Braille dot positions and bit values:
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

        # Each braille char covers 4 rows x 2 cols
        braille_rows = (self.m + 3) // 4
        braille_cols = (self.n + 1) // 2

        # Downsample if needed
        row_scale = max(1, (braille_rows + max_height - 1) // max_height)
        col_scale = max(1, (braille_cols + max_width - 1) // max_width)

        out_rows = (braille_rows + row_scale - 1) // row_scale
        out_cols = (braille_cols + col_scale - 1) // col_scale

        # Build a dense boolean matrix for the pattern
        dense = self.todense()

        # Braille bit positions: maps (row_offset, col_offset) -> bit
        bit_map = {
            (0, 0): 0x01,
            (1, 0): 0x02,
            (2, 0): 0x04,
            (3, 0): 0x40,
            (0, 1): 0x08,
            (1, 1): 0x10,
            (2, 1): 0x20,
            (3, 1): 0x80,
        }

        lines = []
        for br in range(out_rows):
            line = []
            for bc in range(out_cols):
                # Compute which braille cells this output char covers
                br_start = br * row_scale
                br_end = min((br + 1) * row_scale, braille_rows)
                bc_start = bc * col_scale
                bc_end = min((bc + 1) * col_scale, braille_cols)

                # Check if any dot should be lit
                char_bits = 0
                for sub_br in range(br_start, br_end):
                    for sub_bc in range(bc_start, bc_end):
                        for dr in range(4):
                            for dc in range(2):
                                mat_r = sub_br * 4 + dr
                                mat_c = sub_bc * 2 + dc
                                if (
                                    mat_r < self.m
                                    and mat_c < self.n
                                    and dense[mat_r, mat_c]
                                ):
                                    char_bits |= bit_map[(dr, dc)]

                line.append(chr(0x2800 + char_bits))
            lines.append("".join(line))

        return "\n".join(lines)

    def __str__(self) -> str:
        """Return string representation with visualization.

        Uses dot display (●/⋅) for small matrices, braille for large ones.
        Follows Julia's SparseArrays display heuristics.
        """
        header = f"SparsityPattern({self.m}×{self.n}, nnz={self.nse}, density={self.density:.1%})"

        # Use dot display for small matrices, braille for large ones
        if self.m <= self._SMALL_ROWS and self.n <= self._SMALL_COLS:
            visualization = self._render_dots()
        else:
            braille = self._render_braille()
            # Add box drawing borders for braille
            braille_lines = braille.split("\n")
            if braille_lines and braille_lines[0] != "(empty)":
                width = max(len(line) for line in braille_lines)
                bordered = ["┌" + "─" * width + "┐"]
                for line in braille_lines:
                    bordered.append("│" + line.ljust(width) + "│")
                bordered.append("└" + "─" * width + "┘")
                visualization = "\n".join(bordered)
            else:
                visualization = braille

        return f"{header}\n{visualization}"

    def __repr__(self) -> str:
        """Return compact representation."""
        return f"SparsityPattern(shape={self.shape}, nnz={self.nse})"

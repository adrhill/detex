"""Pattern data structures for the detection->coloring->decompression pipeline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from numpy.typing import NDArray

from asdex._display import colored_repr, colored_str, sparsity_repr, sparsity_str


@dataclass(frozen=True)
class SparsityPattern:
    """Sparse matrix pattern storing only structural information (no values).

    Stores row and column indices separately for efficient access
    by the coloring and decompression stages.

    Attributes:
        rows: Row indices of non-zero entries, shape ``(nnz,)``
        cols: Column indices of non-zero entries, shape ``(nnz,)``
        shape: Matrix dimensions ``(m, n)``
        input_shape: Shape of the function input that produced this pattern.
            Defaults to ``(n,)`` if not specified.
    """

    rows: NDArray[np.int32]
    cols: NDArray[np.int32]
    shape: tuple[int, int]
    input_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate inputs and set defaults."""
        if len(self.rows) != len(self.cols):
            msg = f"rows and cols must have same length, got {len(self.rows)} and {len(self.cols)}"
            raise ValueError(msg)
        if self.input_shape is None:
            object.__setattr__(self, "input_shape", (self.n,))

    # Properties

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self.rows)

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
        return self.nnz / total if total > 0 else 0.0

    @cached_property
    def col_to_rows(self) -> dict[int, list[int]]:
        """Mapping from column index to list of row indices with non-zeros.

        Used by the coloring algorithm to build the row conflict graph.
        """
        result: dict[int, list[int]] = defaultdict(list)
        for row, col in zip(self.rows, self.cols, strict=True):
            result[int(col)].append(int(row))
        return dict(result)

    @cached_property
    def row_to_cols(self) -> dict[int, list[int]]:
        """Mapping from row index to list of column indices with non-zeros.

        Used by the coloring algorithm to build the column conflict graph.
        """
        result: dict[int, list[int]] = defaultdict(list)
        for row, col in zip(self.rows, self.cols, strict=True):
            result[int(row)].append(int(col))
        return dict(result)

    # Constructors

    @classmethod
    def from_coordinates(
        cls,
        rows: NDArray[np.int32] | list[int],
        cols: NDArray[np.int32] | list[int],
        shape: tuple[int, int],
        *,
        input_shape: tuple[int, ...] | None = None,
    ) -> SparsityPattern:
        """Create pattern from row and column index arrays.

        Args:
            rows: Row indices of non-zero entries.
            cols: Column indices of non-zero entries.
            shape: Matrix dimensions ``(m, n)``.
            input_shape: Shape of the function input.
                Defaults to ``(n,)`` if not specified.
        """
        return cls(
            rows=np.asarray(rows, dtype=np.int32),
            cols=np.asarray(cols, dtype=np.int32),
            shape=shape,
            input_shape=input_shape,
        )

    @classmethod
    def from_bcoo(cls, bcoo: BCOO) -> SparsityPattern:
        """Create pattern from JAX BCOO sparse matrix."""
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

        Non-zero entries indicate pattern positions.
        """
        dense = np.asarray(dense)
        rows, cols = np.nonzero(dense)
        return cls(
            rows=rows.astype(np.int32),
            cols=cols.astype(np.int32),
            shape=(dense.shape[0], dense.shape[1]),
        )

    # Conversion methods

    @cached_property
    def _bcoo_indices(self) -> jnp.ndarray:
        """BCOO index array of shape ``(nnz, 2)``, cached for reuse."""
        if self.nnz == 0:
            return jnp.zeros((0, 2), dtype=jnp.int32)
        return jnp.stack([self.rows, self.cols], axis=1)

    def to_bcoo(self, data: jnp.ndarray | None = None) -> BCOO:
        """Convert to JAX BCOO sparse matrix.

        Args:
            data: Optional data values.
                If None, uses all 1s.
        """
        indices = self._bcoo_indices
        if data is None:
            if self.nnz == 0:
                data = jnp.array([])
            else:
                data = jnp.ones(self.nnz, dtype=jnp.int8)
        return BCOO((data, indices), shape=self.shape)

    def todense(self) -> NDArray:
        """Convert to dense numpy array with 1s at pattern positions."""
        result = np.zeros(self.shape, dtype=np.int8)
        if self.nnz > 0:
            result[self.rows, self.cols] = 1
        return result

    # Display

    def __str__(self) -> str:
        """Render sparsity pattern with header and dot/braille grid."""
        return sparsity_str(self)

    def __repr__(self) -> str:
        """Return compact single-line representation."""
        return sparsity_repr(self)


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
            ``"HVP"`` for symmetrically colored Hessians.
    """

    sparsity: SparsityPattern
    colors: NDArray[np.int32]
    num_colors: int
    mode: Literal["JVP", "VJP", "HVP"]

    @property
    def _compresses_columns(self) -> bool:
        """Whether coloring compresses columns (JVP/HVP) or rows (VJP)."""
        return self.mode in ("JVP", "HVP")

    # Cached arrays for fast decompression

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
        For HVP: delegates to `_star_extraction_indices`.
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
        """Pre-compute HVP extraction indices with symmetric coloring direction choice.

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

    # Display

    def __repr__(self) -> str:
        """Return compact single-line representation."""
        return colored_repr(self)

    def __str__(self) -> str:
        """Render colored pattern with sparsity grid and color assignments."""
        return colored_str(self)

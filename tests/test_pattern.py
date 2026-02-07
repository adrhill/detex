"""Tests for SparsityPattern data structure."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from asdex import SparsityPattern, jacobian_sparsity


class TestValidation:
    """Test input validation."""

    def test_mismatched_rows_cols_raises(self):
        """rows and cols with different lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            SparsityPattern.from_coordinates([0, 1], [0], (2, 2))


class TestConstruction:
    """Test SparsityPattern construction methods."""

    def test_from_coordinates(self):
        """Basic construction from row/col arrays."""
        rows = [0, 0, 1, 2]
        cols = [0, 1, 1, 2]
        pattern = SparsityPattern.from_coordinates(rows, cols, (3, 3))

        assert pattern.shape == (3, 3)
        assert pattern.nse == 4
        assert pattern.nnz == 4
        assert pattern.m == 3
        assert pattern.n == 3
        np.testing.assert_array_equal(pattern.rows, [0, 0, 1, 2])
        np.testing.assert_array_equal(pattern.cols, [0, 1, 1, 2])

    def test_from_coordinates_empty(self):
        """Construction with no non-zeros."""
        pattern = SparsityPattern.from_coordinates([], [], (3, 4))

        assert pattern.shape == (3, 4)
        assert pattern.nse == 0
        assert pattern.m == 3
        assert pattern.n == 4

    def test_from_bcoo_roundtrip(self):
        """Convert from BCOO and back."""
        # Create a BCOO matrix
        data = jnp.array([1, 1, 1])
        indices = jnp.array([[0, 0], [1, 1], [2, 2]])
        bcoo = BCOO((data, indices), shape=(3, 3))

        # Convert to SparsityPattern
        pattern = SparsityPattern.from_bcoo(bcoo)
        assert pattern.shape == (3, 3)
        assert pattern.nse == 3

        # Convert back to BCOO
        bcoo2 = pattern.to_bcoo()
        assert bcoo2.shape == (3, 3)
        np.testing.assert_array_equal(bcoo2.todense(), bcoo.todense())

    def test_from_bcoo_empty(self):
        """Convert empty BCOO to SparsityPattern."""
        data = jnp.array([])
        indices = jnp.zeros((0, 2), dtype=jnp.int32)
        bcoo = BCOO((data, indices), shape=(3, 4))

        pattern = SparsityPattern.from_bcoo(bcoo)
        assert pattern.shape == (3, 4)
        assert pattern.nse == 0

    def test_from_dense(self):
        """Construction from dense matrix."""
        dense = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        pattern = SparsityPattern.from_dense(dense)

        assert pattern.shape == (3, 3)
        assert pattern.nse == 5
        np.testing.assert_array_equal(pattern.todense(), (dense != 0).astype(np.int8))


class TestConversion:
    """Test conversion methods."""

    def test_todense(self):
        """Convert to dense numpy array."""
        pattern = SparsityPattern.from_coordinates([0, 1, 2], [0, 1, 2], (3, 3))
        dense = pattern.todense()

        expected = np.eye(3, dtype=np.int8)
        np.testing.assert_array_equal(dense, expected)

    def test_todense_empty(self):
        """todense with no non-zeros."""
        pattern = SparsityPattern.from_coordinates([], [], (2, 3))
        dense = pattern.todense()

        expected = np.zeros((2, 3), dtype=np.int8)
        np.testing.assert_array_equal(dense, expected)

    def test_astype(self):
        """astype for test compatibility."""
        pattern = SparsityPattern.from_coordinates([0, 1], [0, 1], (2, 2))

        int_arr = pattern.astype(int)
        assert int_arr.dtype == int
        np.testing.assert_array_equal(int_arr, np.eye(2, dtype=int))

        float_arr = pattern.astype(np.float32)
        assert float_arr.dtype == np.float32

    def test_to_bcoo_with_data(self):
        """to_bcoo with custom data values."""
        pattern = SparsityPattern.from_coordinates([0, 1, 2], [0, 1, 2], (3, 3))
        data = jnp.array([2.0, 3.0, 4.0])
        bcoo = pattern.to_bcoo(data=data)

        expected = np.diag([2.0, 3.0, 4.0])
        np.testing.assert_array_equal(bcoo.todense(), expected)

    def test_to_bcoo_default_data(self):
        """to_bcoo uses 1s by default."""
        pattern = SparsityPattern.from_coordinates([0, 1], [0, 1], (2, 2))
        bcoo = pattern.to_bcoo()

        np.testing.assert_array_equal(bcoo.todense(), np.eye(2))

    def test_to_bcoo_empty(self):
        """to_bcoo with empty pattern produces zero matrix."""
        pattern = SparsityPattern.from_coordinates([], [], (3, 4))
        bcoo = pattern.to_bcoo()

        assert bcoo.shape == (3, 4)
        np.testing.assert_array_equal(bcoo.todense(), np.zeros((3, 4)))

    def test_to_bcoo_empty_with_data(self):
        """to_bcoo with empty pattern and custom data."""
        pattern = SparsityPattern.from_coordinates([], [], (2, 2))
        data = jnp.array([])
        bcoo = pattern.to_bcoo(data=data)

        assert bcoo.shape == (2, 2)
        np.testing.assert_array_equal(bcoo.todense(), np.zeros((2, 2)))


class TestProperties:
    """Test computed properties."""

    def test_density(self):
        """Density calculation."""
        # 2 non-zeros in 3x4 = 12 elements
        pattern = SparsityPattern.from_coordinates([0, 1], [0, 1], (3, 4))
        assert pattern.density == pytest.approx(2 / 12)

    def test_density_empty(self):
        """Density of empty pattern."""
        pattern = SparsityPattern.from_coordinates([], [], (3, 4))
        assert pattern.density == 0.0

    def test_density_zero_size(self):
        """Density with zero-size matrix."""
        pattern = SparsityPattern.from_coordinates([], [], (0, 4))
        assert pattern.density == 0.0

    def test_col_to_rows(self):
        """col_to_rows mapping."""
        # Pattern: row 0 has cols 0,1; row 1 has col 1; row 2 has col 2
        pattern = SparsityPattern.from_coordinates([0, 0, 1, 2], [0, 1, 1, 2], (3, 3))

        col_to_rows = pattern.col_to_rows
        assert col_to_rows == {0: [0], 1: [0, 1], 2: [2]}

    def test_col_to_rows_caching(self):
        """col_to_rows is cached."""
        pattern = SparsityPattern.from_coordinates([0, 1], [0, 1], (2, 2))

        # Access twice - should be same object
        first = pattern.col_to_rows
        second = pattern.col_to_rows
        assert first is second


class TestVisualization:
    """Test visualization (dots for small, braille for large)."""

    def test_small_matrix_uses_dots(self):
        """Small matrices use dot display (●/⋅)."""
        pattern = SparsityPattern.from_coordinates([0, 1, 2], [0, 1, 2], (3, 3))
        s = str(pattern)

        # Should have header line
        assert "SparsityPattern" in s
        assert "3×3" in s
        assert "nnz=3" in s
        # Should have dots, not braille
        assert "●" in s
        assert "⋅" in s

    def test_large_matrix_uses_braille(self):
        """Large matrices use braille display."""
        # Create 20x50 pattern (exceeds thresholds)
        rows = list(range(20))
        cols = list(range(20))
        pattern = SparsityPattern.from_coordinates(rows, cols, (20, 50))
        s = str(pattern)

        # Should have braille characters (Unicode block starting at 0x2800)
        assert any(ord(c) >= 0x2800 and ord(c) < 0x2900 for c in s)
        # Should have Julia-style bracket borders
        assert "⎡" in s
        assert "⎦" in s

    def test_repr_compact(self):
        """__repr__ is compact."""
        pattern = SparsityPattern.from_coordinates([0, 1], [0, 1], (10, 20))
        r = repr(pattern)

        assert "SparsityPattern" in r
        assert "shape=(10, 20)" in r
        assert "nnz=2" in r
        # Should be single line
        assert "\n" not in r

    def test_render_dots_empty_matrix(self):
        """Dot rendering of empty matrix."""
        pattern = SparsityPattern.from_coordinates([], [], (0, 0))
        dots = pattern._render_dots()
        assert dots == "(empty)"

    def test_render_dots_small_diagonal(self):
        """Dot rendering of small diagonal pattern."""
        pattern = SparsityPattern.from_coordinates([0, 1, 2], [0, 1, 2], (3, 3))
        dots = pattern._render_dots()

        # Should show diagonal pattern
        lines = dots.split("\n")
        assert len(lines) == 3
        assert "●" in lines[0]
        assert "⋅" in lines[0]

    def test_braille_empty_matrix(self):
        """Braille rendering of empty matrix."""
        pattern = SparsityPattern.from_coordinates([], [], (0, 0))
        braille = pattern._render_braille()
        assert braille == "(empty)"

    def test_braille_large_matrix_downsamples(self):
        """Large matrices are downsampled in braille."""
        # Create 100x100 diagonal
        rows = list(range(100))
        cols = list(range(100))
        pattern = SparsityPattern.from_coordinates(rows, cols, (100, 100))

        braille = pattern._render_braille(max_height=10, max_width=20)
        lines = braille.split("\n")

        # Should be within limits
        assert len(lines) <= 10
        assert all(len(line) <= 20 for line in lines)

    def test_large_zero_dim_matrix_str(self):
        """Large matrix with zero dimension uses braille "(empty)" fallback in __str__.

        When m or n is 0 but exceeds small-matrix thresholds,
        braille returns "(empty)" and __str__ uses it directly.
        """
        # n=50 exceeds _SMALL_COLS=40, forcing braille path; m=0 triggers "(empty)"
        pattern = SparsityPattern.from_coordinates([], [], (0, 50))
        s = str(pattern)

        assert "SparsityPattern" in s
        assert "nnz=0" in s
        assert "(empty)" in s


class TestIntegration:
    """Integration tests with detection pipeline."""

    def test_jacobian_sparsity_returns_pattern(self):
        """jacobian_sparsity returns SparsityPattern."""

        def f(x):
            return jnp.array([x[0] * x[1], x[1] + x[2], x[2]])

        result = jacobian_sparsity(f, input_shape=3)

        assert isinstance(result, SparsityPattern)
        assert result.shape == (3, 3)

        # Check sparsity pattern is correct
        expected = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
        np.testing.assert_array_equal(result.todense(), expected)

    def test_existing_tests_still_work(self):
        """Existing test patterns like .todense().astype(int) work."""

        def f(x):
            return x**2

        result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
        expected = np.eye(3, dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_print_sparsity_pattern(self):
        """Manual verification helper - prints sparsity pattern."""

        def f(x):
            return jnp.array([x[0] * x[1], x[1] + x[2], x[2]])

        pattern = jacobian_sparsity(f, input_shape=3)
        # This should print nicely with braille
        output = str(pattern)
        assert len(output) > 0

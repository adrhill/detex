"""Tests for the dot_general propagation handler.

Tests matrix multiplications, batched dots, dot products,
outer products, and higher-dimensional contractions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


def _dot_general_jacobian(f, n):
    """Compute the expected Jacobian via jax.jacobian (nonzero pattern)."""
    x = jnp.ones(n)
    J = jax.jacobian(f)(x)
    return (np.array(J).reshape(-1, n) != 0).astype(int)


# ── Vector dot product ──────────────────────────────────────────────────────


@pytest.mark.array_ops
def test_dot_1d():
    """Dot product of two copies of the same vector.

    out = x @ x = Σ_i x[i]^2, so out depends on all inputs.
    """

    def f(x):
        return jnp.dot(x, x).reshape(1)

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    expected = np.ones((1, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_dot_1d_separate():
    """Dot product x @ y where x and y are separate halves of input.

    out depends on all inputs.
    """

    def f(xy):
        x = xy[:3]
        y = xy[3:]
        return jnp.dot(x, y).reshape(1)

    result = jacobian_sparsity(f, input_shape=6).todense().astype(int)
    expected = np.ones((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


# ── Matrix-vector multiply ──────────────────────────────────────────────────


@pytest.mark.array_ops
def test_matvec():
    """Matrix-vector multiply: out[i] = Σ_k A[i,k] * v[k].

    A and v come from the same input, so out[i] depends on row i of A and all of v.
    """

    def f(x):
        # x[:6] is A (2x3), x[6:] is v (3,)
        A = x[:6].reshape(2, 3)
        v = x[6:]
        return A @ v

    result = jacobian_sparsity(f, input_shape=9).todense().astype(int)
    # out[0] depends on A row 0 ({0,1,2}) and v ({6,7,8})
    # out[1] depends on A row 1 ({3,4,5}) and v ({6,7,8})
    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(result, expected)


# ── Matrix-matrix multiply ──────────────────────────────────────────────────


@pytest.mark.array_ops
def test_matmul_2x3_3x2():
    """Matrix multiply A(2,3) @ B(3,2) → C(2,2).

    A and B are separate halves of the input.
    """

    def f(x):
        A = x[:6].reshape(2, 3)
        B = x[6:].reshape(3, 2)
        return (A @ B).flatten()

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    # out[i,j] depends on A row i and B col j.
    # A row 0: {0,1,2}, A row 1: {3,4,5}
    # B col 0: {6,8,10}, B col 1: {7,9,11}
    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],  # out[0,0]
            [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],  # out[0,1]
            [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0],  # out[1,0]
            [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],  # out[1,1]
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_matmul_self():
    """X @ x.T where x is a 2x2 matrix.

    Both operands share the same input, so deps are unioned.
    """

    def f(x):
        mat = x.reshape(2, 2)
        return (mat @ mat.T).flatten()

    result = jacobian_sparsity(f, input_shape=4).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
        ]
    )
    np.testing.assert_array_equal(result, expected)


# ── Batched matrix multiply ─────────────────────────────────────────────────


@pytest.mark.array_ops
def test_batched_matmul():
    """Batched matmul: batch dim is preserved, each batch independent.

    A(2,2,3) @ B(2,3,1) → C(2,2,1).
    Batch dim 0 is shared, contraction on dim 2 of A and dim 1 of B.
    """

    def f(x):
        A = x[:12].reshape(2, 2, 3)
        B = x[12:].reshape(2, 3, 1)
        return jnp.matmul(A, B).flatten()

    result = jacobian_sparsity(f, input_shape=18).todense().astype(int)
    # Batch 0: A[0] rows {0..5}, B[0] = {12,13,14}
    # Batch 1: A[1] rows {6..11}, B[1] = {15,16,17}
    # out[0,0,0] = A[0,0,:] @ B[0,:,0] → {0,1,2} ∪ {12,13,14}
    # out[0,1,0] = A[0,1,:] @ B[0,:,0] → {3,4,5} ∪ {12,13,14}
    # out[1,0,0] = A[1,0,:] @ B[1,:,0] → {6,7,8} ∪ {15,16,17}
    # out[1,1,0] = A[1,1,:] @ B[1,:,0] → {9,10,11} ∪ {15,16,17}
    expected = np.zeros((4, 18), dtype=int)
    expected[0, [0, 1, 2, 12, 13, 14]] = 1
    expected[1, [3, 4, 5, 12, 13, 14]] = 1
    expected[2, [6, 7, 8, 15, 16, 17]] = 1
    expected[3, [9, 10, 11, 15, 16, 17]] = 1
    np.testing.assert_array_equal(result, expected)


# ── Outer product ────────────────────────────────────────────────────────────


@pytest.mark.array_ops
def test_outer_product_via_dot_general():
    """Outer product via dot_general with no contracting dimensions.

    out[i,j] = a[i] * b[j], so each output depends on one element from each input.
    """

    def f(x):
        a = x[:3]
        b = x[3:]
        return jax.lax.dot_general(
            a, b, dimension_numbers=(([], []), ([], []))
        ).flatten()

    result = jacobian_sparsity(f, input_shape=5).todense().astype(int)
    # out[i,j] depends on {i} from a and {3+j} from b.
    expected = np.zeros((6, 5), dtype=int)
    for i in range(3):
        for j in range(2):
            expected[i * 2 + j, i] = 1
            expected[i * 2 + j, 3 + j] = 1
    np.testing.assert_array_equal(result, expected)


# ── Size-1 dimensions ───────────────────────────────────────────────────────


@pytest.mark.array_ops
def test_matmul_size_1_row():
    """Matrix multiply with size-1 row: (1,3) @ (3,2) → (1,2)."""

    def f(x):
        A = x[:3].reshape(1, 3)
        B = x[3:].reshape(3, 2)
        return (A @ B).flatten()

    result = jacobian_sparsity(f, input_shape=9).todense().astype(int)
    # out[0,0] depends on A[0,:] = {0,1,2} and B[:,0] = {3,5,7}
    # out[0,1] depends on A[0,:] = {0,1,2} and B[:,1] = {4,6,8}
    expected = np.array(
        [
            [1, 1, 1, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_matmul_size_1_contract():
    """Contraction over size-1 dimension: (2,1) @ (1,3) → (2,3).

    Each output depends on exactly one lhs element and one rhs element.
    """

    def f(x):
        A = x[:2].reshape(2, 1)
        B = x[2:].reshape(1, 3)
        return (A @ B).flatten()

    result = jacobian_sparsity(f, input_shape=5).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 1, 0, 0],  # out[0,0]: A[0,0]={0}, B[0,0]={2}
            [1, 0, 0, 1, 0],  # out[0,1]: A[0,0]={0}, B[0,1]={3}
            [1, 0, 0, 0, 1],  # out[0,2]: A[0,0]={0}, B[0,2]={4}
            [0, 1, 1, 0, 0],  # out[1,0]: A[1,0]={1}, B[0,0]={2}
            [0, 1, 0, 1, 0],  # out[1,1]: A[1,0]={1}, B[0,1]={3}
            [0, 1, 0, 0, 1],  # out[1,2]: A[1,0]={1}, B[0,2]={4}
        ]
    )
    np.testing.assert_array_equal(result, expected)


# ── Higher-dimensional contractions ─────────────────────────────────────────


@pytest.mark.array_ops
def test_einsum_ij_jk():
    """Einstein summation ij,jk->ik (standard matmul via einsum)."""

    def f(x):
        A = x[:6].reshape(2, 3)
        B = x[6:].reshape(3, 2)
        return jnp.einsum("ij,jk->ik", A, B).flatten()

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    expected = _dot_general_jacobian(f, 12)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_einsum_batched():
    """Batched einsum bij,bjk->bik."""

    def f(x):
        A = x[:12].reshape(2, 2, 3)
        B = x[12:].reshape(2, 3, 2)
        return jnp.einsum("bij,bjk->bik", A, B).flatten()

    result = jacobian_sparsity(f, input_shape=24).todense().astype(int)
    expected = _dot_general_jacobian(f, 24)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tensordot_axes_1():
    """Tensordot with axes=1: contracts the last axis of A with first of B."""

    def f(x):
        A = x[:6].reshape(2, 3)
        B = x[6:].reshape(3, 4)
        return jnp.tensordot(A, B, axes=1).flatten()

    result = jacobian_sparsity(f, input_shape=18).todense().astype(int)
    expected = _dot_general_jacobian(f, 18)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tensordot_axes_2():
    """Tensordot with axes=2: contracts last 2 axes of A with first 2 of B."""

    def f(x):
        A = x[:12].reshape(2, 2, 3)
        B = x[12:].reshape(2, 3, 4)
        return jnp.tensordot(A, B, axes=2).flatten()

    result = jacobian_sparsity(f, input_shape=36).todense().astype(int)
    expected = _dot_general_jacobian(f, 36)
    np.testing.assert_array_equal(result, expected)


# ── Non-contiguous input deps ────────────────────────────────────────────────


@pytest.mark.array_ops
def test_matmul_after_broadcast():
    """Matmul where inputs have non-trivial dep sets from a prior broadcast.

    Verifies that set merging handles non-singleton dep sets correctly.
    """

    def f(x):
        # Broadcast x (shape 3) to (2,3), then matmul with itself transposed.
        mat = jnp.broadcast_to(x, (2, 3))
        return (mat @ mat.T).flatten()

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # After broadcast, mat deps: each row is [{0},{1},{2}].
    # mat.T deps: each col is [{0},{1},{2}].
    # mat @ mat.T: all outputs depend on all inputs.
    expected = np.ones((4, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_matmul_with_constant():
    """Matmul where one operand is a constant (no input deps).

    out[i] = const @ x, so out[i] depends on all of x
    (handler unions over all contracting positions).
    """

    def f(x):
        W = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        return W @ x

    result = jacobian_sparsity(f, input_shape=3).todense().astype(int)
    # Dense because handler can't exploit value-level zeros.
    expected = np.ones((2, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


# ── Verify against jax.jacobian ──────────────────────────────────────────────


@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("desc", "f", "n"),
    [
        ("vecvec", lambda x: jnp.dot(x[:2], x[2:]).reshape(1), 4),
        ("matvec", lambda x: x[:6].reshape(2, 3) @ x[6:], 9),
        ("matmat", lambda x: (x[:6].reshape(2, 3) @ x[6:].reshape(3, 2)).flatten(), 12),
    ],
    ids=["vecvec", "matvec", "matmat"],
)
def test_against_jax_jacobian(desc, f, n):
    """Sparsity pattern matches the nonzero pattern of jax.jacobian."""
    result = jacobian_sparsity(f, input_shape=n).todense().astype(int)
    expected = _dot_general_jacobian(f, n)
    np.testing.assert_array_equal(result, expected)


# ── Adversarial edge cases ───────────────────────────────────────────────────


@pytest.mark.array_ops
def test_multi_contract_dims():
    """Multiple contracting dimensions: contract on 2 axes simultaneously."""

    def f(x):
        A = x[:12].reshape(2, 3, 2)
        B = x[12:].reshape(3, 2, 4)
        # Contract dims (1,2) of A with (0,1) of B → shape (2, 4)
        return jax.lax.dot_general(
            A, B, dimension_numbers=(((1, 2), (0, 1)), ((), ()))
        ).flatten()

    result = jacobian_sparsity(f, input_shape=36).todense().astype(int)
    expected = _dot_general_jacobian(f, 36)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_multi_batch_dims():
    """Multiple batch dimensions: batch on 2 axes."""

    def f(x):
        A = x[:24].reshape(2, 3, 4)
        B = x[24:].reshape(2, 3, 5)
        # Batch on (0,1), no contraction → shape (2, 3, 4, 5)
        return jax.lax.dot_general(
            A, B, dimension_numbers=(((), ()), ((0, 1), (0, 1)))
        ).flatten()

    result = jacobian_sparsity(f, input_shape=54).todense().astype(int)
    expected = _dot_general_jacobian(f, 54)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_batch_and_contract():
    """Both batch and contracting dimensions present."""

    def f(x):
        A = x[:24].reshape(2, 3, 4)
        B = x[24:].reshape(2, 4, 5)
        # Batch on (0,), contract on (2,) of A with (1,) of B → shape (2, 3, 5)
        return jax.lax.dot_general(
            A, B, dimension_numbers=(((2,), (1,)), ((0,), (0,)))
        ).flatten()

    result = jacobian_sparsity(f, input_shape=64).todense().astype(int)
    expected = _dot_general_jacobian(f, 64)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_size_1_contract():
    """Contraction over a size-1 dimension (effectively an outer product)."""

    def f(x):
        A = x[:6].reshape(3, 1, 2)
        B = x[6:].reshape(1, 4)
        # Contract dim 1 of A with dim 0 of B → shape (3, 2, 4)
        return jax.lax.dot_general(
            A, B, dimension_numbers=(((1,), (0,)), ((), ()))
        ).flatten()

    result = jacobian_sparsity(f, input_shape=10).todense().astype(int)
    expected = _dot_general_jacobian(f, 10)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_double_matmul():
    """Chained matmul: (A @ B) @ C."""

    def f(x):
        A = x[:6].reshape(2, 3)
        B = x[6:15].reshape(3, 3)
        C = x[15:].reshape(3, 2)
        return ((A @ B) @ C).flatten()

    result = jacobian_sparsity(f, input_shape=21).todense().astype(int)
    expected = _dot_general_jacobian(f, 21)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_vecdot():
    """jnp.vdot (flattened dot product) on matrices."""

    def f(x):
        A = x[:6].reshape(2, 3)
        B = x[6:].reshape(2, 3)
        return jnp.vdot(A, B).reshape(1)

    result = jacobian_sparsity(f, input_shape=12).todense().astype(int)
    expected = np.ones((1, 12), dtype=int)
    np.testing.assert_array_equal(result, expected)

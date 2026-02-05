"""Test asdex on convolutional layers, similar to SCT's performance example.

Reference: https://github.com/adrhill/SparseConnectivityTracer.jl/blob/main/docs/src/user/performance.md
"""

import jax
import numpy as np
import pytest
from flax import nnx

from asdex import jacobian_sparsity


def check_conv_sparsity(
    input_hw: tuple[int, int],
    kernel_size: tuple[int, int],
    channels: tuple[int, int],
    *,
    verify_exact: bool = True,
) -> None:
    """Test conv sparsity detection for given parameters.

    Args:
        input_hw: (height, width) of input image
        kernel_size: (kH, kW) kernel dimensions
        channels: (C_in, C_out) input and output channels
        verify_exact: If True, verify exact match with actual Jacobian.
                      Set False for large inputs where Jacobian is expensive.
    """
    H, W = input_hw
    C_in, C_out = channels
    input_size = H * W * C_in

    conv = nnx.Conv(C_in, C_out, kernel_size, padding="VALID", rngs=nnx.Rngs(0))

    def f(x_flat):
        x = x_flat.reshape(1, H, W, C_in)
        y = conv(x)
        return y.flatten()

    sparsity = jacobian_sparsity(f, n=input_size)

    # Expected output shape with VALID padding
    out_h = H - kernel_size[0] + 1
    out_w = W - kernel_size[1] + 1
    expected_out_size = out_h * out_w * C_out

    assert sparsity.shape == (expected_out_size, input_size)
    assert sparsity.nse > 0

    if verify_exact:
        x_test = jax.random.normal(jax.random.key(42), (input_size,))
        actual_jac = jax.jacobian(f)(x_test)
        actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)
        detected = sparsity.todense().astype(int)

        np.testing.assert_array_equal(
            detected, actual_nonzero, "Detected sparsity should match actual Jacobian"
        )


@pytest.mark.parametrize(
    "input_hw,kernel_size,channels",
    [
        ((4, 4), (2, 2), (1, 1)),
        ((4, 4), (3, 3), (1, 1)),
        ((6, 6), (2, 2), (1, 1)),
        ((6, 6), (3, 3), (2, 3)),
        ((8, 8), (3, 3), (3, 2)),
        ((8, 8), (5, 5), (1, 1)),
    ],
)
def test_conv_exact(input_hw, kernel_size, channels):
    """Verify exact sparsity match for small convolutions."""
    check_conv_sparsity(input_hw, kernel_size, channels, verify_exact=True)


def test_conv_sct_example():
    """SCT example: 28x28 input, 3x3 kernel, 3 in channels, 2 out channels."""
    check_conv_sparsity((28, 28), (3, 3), (3, 2), verify_exact=False)


def test_conv_with_padding():
    """Convolution with SAME padding exercises boundary coordinate checks."""
    H, W = 4, 4
    C_in, C_out = 1, 1
    input_size = H * W * C_in

    # SAME padding adds zeros around borders, so kernel windows at edges
    # will have coordinates that fall outside the valid input region
    conv = nnx.Conv(C_in, C_out, (3, 3), padding="SAME", rngs=nnx.Rngs(0))

    def f(x_flat):
        x = x_flat.reshape(1, H, W, C_in)
        y = conv(x)
        return y.flatten()

    sparsity = jacobian_sparsity(f, n=input_size)

    # With SAME padding, output size equals input size
    expected_out_size = H * W * C_out
    assert sparsity.shape == (expected_out_size, input_size)
    assert sparsity.nse > 0

    # Verify against actual Jacobian
    x_test = jax.random.normal(jax.random.key(42), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)
    detected = sparsity.todense().astype(int)

    np.testing.assert_array_equal(
        detected, actual_nonzero, "Detected sparsity should match actual Jacobian"
    )


def test_conv_with_strides():
    """Convolution with strides > 1."""
    H, W = 6, 6
    C_in, C_out = 1, 1
    input_size = H * W * C_in

    conv = nnx.Conv(
        C_in, C_out, (3, 3), strides=(2, 2), padding="VALID", rngs=nnx.Rngs(0)
    )

    def f(x_flat):
        x = x_flat.reshape(1, H, W, C_in)
        y = conv(x)
        return y.flatten()

    sparsity = jacobian_sparsity(f, n=input_size)

    # Output size with stride 2 and VALID padding
    out_h = (H - 3) // 2 + 1  # 2
    out_w = (W - 3) // 2 + 1  # 2
    expected_out_size = out_h * out_w * C_out

    assert sparsity.shape == (expected_out_size, input_size)
    assert sparsity.nse > 0

    # Verify against actual Jacobian
    x_test = jax.random.normal(jax.random.key(42), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)
    detected = sparsity.todense().astype(int)

    np.testing.assert_array_equal(
        detected, actual_nonzero, "Detected sparsity should match actual Jacobian"
    )


def test_conv_transpose():
    """Transposed convolution exercises lhs_dilation code path."""
    H, W = 4, 4
    C_in, C_out = 1, 1
    input_size = H * W * C_in

    # ConvTranspose uses lhs_dilation internally
    conv_t = nnx.ConvTranspose(
        C_in, C_out, (3, 3), strides=(2, 2), padding="VALID", rngs=nnx.Rngs(0)
    )

    def f(x_flat):
        x = x_flat.reshape(1, H, W, C_in)
        y = conv_t(x)
        return y.flatten()

    sparsity = jacobian_sparsity(f, n=input_size)

    assert sparsity.nse > 0

    # Verify against actual Jacobian
    x_test = jax.random.normal(jax.random.key(42), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)
    detected = sparsity.todense().astype(int)

    np.testing.assert_array_equal(
        detected, actual_nonzero, "Detected sparsity should match actual Jacobian"
    )

"""Propagation rules for convolution operations."""

from itertools import product

from jax._src.core import JaxprEqn

from ._commons import Deps, IndexSets, index_sets, numel, row_strides


def prop_conv_general_dilated(eqn: JaxprEqn, deps: Deps) -> None:
    """Convolution slides a kernel over the input, computing weighted sums.
    Each output element depends on a local spatial window of input elements
    across all input channels.

    For 2D conv with kernel size (kH, kW), stride s, and C_in input channels:
        out[n, h, w, c_out] = Σ_{kh, kw, c_in} in[n, h·s + kh, w·s + kw, c_in] · W[...]
    So out[n, h, w, :] depends on in[n, h·s : h·s+kH, w·s : w·s+kW, :].

    Example: 1D conv, kernel size 2, input [a, b, c, d]
        out[0] = a·w0 + b·w1  →  deps {0, 1}
        out[1] = b·w0 + c·w1  →  deps {1, 2}
        out[2] = c·w0 + d·w1  →  deps {2, 3}

    Jaxpr:
        invars[0]: input (lhs), invars[1]: kernel (rhs)
        dimension_numbers: specifies layout (batch, feature, spatial dims)
        window_strides, padding, lhs_dilation, rhs_dilation: conv parameters
    """
    lhs_indices = index_sets(deps, eqn.invars[0])  # Input image dependencies

    # Get shapes from avals
    lhs_shape = tuple(getattr(eqn.invars[0].aval, "shape", ()))
    rhs_shape = tuple(getattr(eqn.invars[1].aval, "shape", ()))
    out_shape = tuple(getattr(eqn.outvars[0].aval, "shape", ()))

    # Parse dimension numbers
    dim_nums = eqn.params["dimension_numbers"]
    lhs_spec, rhs_spec, out_spec = (
        dim_nums.lhs_spec,
        dim_nums.rhs_spec,
        dim_nums.out_spec,
    )

    # Extract dimension indices
    lhs_batch_dim, lhs_feature_dim = lhs_spec[0], lhs_spec[1]
    lhs_spatial_dims = lhs_spec[2:]
    out_batch_dim = out_spec[0]
    out_spatial_dims = out_spec[2:]
    rhs_spatial_dims = rhs_spec[2:]

    # Get parameters
    n_spatial = len(lhs_spatial_dims)
    window_strides = eqn.params.get("window_strides", (1,) * n_spatial)
    lhs_dilation = eqn.params.get("lhs_dilation", (1,) * n_spatial)
    rhs_dilation = eqn.params.get("rhs_dilation", (1,) * n_spatial)
    padding = eqn.params.get("padding", ((0, 0),) * n_spatial)

    lhs_strides = row_strides(lhs_shape)
    out_strides = row_strides(out_shape)

    # Get spatial sizes
    lhs_spatial_sizes = [lhs_shape[d] for d in lhs_spatial_dims]
    kernel_spatial_sizes = [rhs_shape[d] for d in rhs_spatial_dims]
    n_in_features = lhs_shape[lhs_feature_dim]

    out_indices: IndexSets = []

    for out_flat in range(numel(out_shape)):
        # Convert flat output index to coordinates
        out_coord = []
        remaining = out_flat
        for s in out_strides:
            out_coord.append(remaining // s)
            remaining %= s

        batch_idx = out_coord[out_batch_dim]
        out_spatial_coord = [out_coord[d] for d in out_spatial_dims]

        # Collect dependencies from input
        elem_deps: set[int] = set()

        # For each position in the kernel window
        for kernel_offsets in product(*[range(k) for k in kernel_spatial_sizes]):
            # Compute input spatial coordinates
            in_spatial_coord = []
            valid = True
            for i in range(n_spatial):
                in_c = (
                    out_spatial_coord[i] * window_strides[i]
                    + kernel_offsets[i] * rhs_dilation[i]
                    - padding[i][0]
                )
                if in_c < 0 or in_c >= lhs_spatial_sizes[i] * lhs_dilation[i]:
                    valid = False
                    break
                if lhs_dilation[i] > 1 and in_c % lhs_dilation[i] != 0:
                    valid = False
                    break
                in_spatial_coord.append(in_c // lhs_dilation[i])

            if not valid:
                continue

            # For each input feature channel
            for in_feature_idx in range(n_in_features):
                in_coord = [0] * len(lhs_shape)
                in_coord[lhs_batch_dim] = batch_idx
                in_coord[lhs_feature_dim] = in_feature_idx
                for i, d in enumerate(lhs_spatial_dims):
                    in_coord[d] = in_spatial_coord[i]

                in_flat = sum(c * s for c, s in zip(in_coord, lhs_strides, strict=True))
                if in_flat < len(lhs_indices):
                    elem_deps |= lhs_indices[in_flat]

        out_indices.append(elem_deps)

    deps[eqn.outvars[0]] = out_indices

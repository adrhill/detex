# TODO - Next Steps for asdex

## Immediate

- [ ] Add more test cases from SparseConnectivityTracer's "Jacobian Global" testset: https://github.com/adrhill/SparseConnectivityTracer.jl/blob/main/test/test_gradient.jl
- [ ] Handle `dynamic_slice` primitive precisely

## Primitive Coverage

Missing precise handlers for:
- [ ] `transpose` - track dimension permutation (`test_transpose_2d`)
- [ ] `dot_general` - matrix multiply sparsity (`test_matmul`, `test_iota_eye`)
- [ ] `reduce_*` variants (max, min, prod with axes)
- [ ] `rev` - reverse/flip array (`test_reverse`)
- [ ] `pad` - constant padding (`test_pad`)
- [ ] `dynamic_slice` - used by split (`test_split`)
- [ ] `dynamic_update_slice` - like scatter but contiguous
- [ ] `reduce_and`, `reduce_or`, `reduce_xor` - same structure as `reduce_sum`
- [ ] `clamp` - ternary element-wise
- [ ] `top_k` - conservative is correct (or close to it)
- [ ] `scatter_sub`, `scatter_mul`, `scatter_max`, `scatter_min` - extend existing `prop_scatter`
- [ ] `custom_jvp_call_jaxpr` - variant of already-handled `custom_jvp_call`
- [ ] `platform_index` - used by `jnp.diag` and other platform-dispatched ops

## Control Flow

- [ ] `cond` - requires unioning outputs across multiple branch jaxprs
- [ ] `scan` - iterative jaxpr application
- [ ] `while` - loop with unknown iteration count

## Architecture Improvements

- [ ] Cache jaxpr analysis for repeated calls

## Conservative Propagators

These propagators use conservative fallbacks that could be made precise:

- [ ] `prop_reshape` - Size mismatch unions all dependencies
- [ ] `prop_conservative_fallback` - Fallback for unhandled primitives (dot_general, dynamic_slice, transpose, etc.)

## Tests Using Conservative Fallbacks

These tests verify conservative behavior that could be made precise:
- [ ] `test_transpose_2d` - transpose produces dense, should be permutation matrix
- [ ] `test_matmul` - dot_general produces dense, should track row/column deps
- [ ] `test_reverse` - rev produces dense, should be anti-diagonal permutation
- [ ] `test_pad` - pad produces dense, should be sparse (pad values have no deps)
- [ ] `test_tile` - broadcast_in_dim produces dense, should track mod pattern
- [ ] `test_split` - dynamic_slice produces dense, should preserve structure
- [ ] `test_iota_eye` - dot_general produces dense, should be identity (iota is now precise)
- [ ] `test_stack` - block-wise deps instead of per-element (reshape limitation)

# TODO - Next Steps for asdex

## Immediate

- [ ] Add more test cases from SparseConnectivityTracer's "Jacobian Global" testset: https://github.com/adrhill/SparseConnectivityTracer.jl/blob/main/test/test_gradient.jl

## Primitive Coverage

Missing precise handlers for:
- [x] `transpose` - track dimension permutation (`test_transpose_2d`)
- [x] `dot_general` - matrix multiply sparsity (`test_matmul`, `test_iota_eye`)
- [ ] `reduce_max`, `reduce_min`, `reduce_prod` - reductions with axes
- [x] `rev` - reverse/flip array (`test_reverse`)
- [ ] `reduce_and`, `reduce_or`, `reduce_xor` - same structure as `reduce_sum`
- [ ] `top_k` - conservative is correct (or close to it)
- [ ] `scatter_sub`, `scatter_mul`, `scatter_max`, `scatter_min` - extend existing `prop_scatter`
- [ ] `platform_index` - used by `jnp.diag` and other platform-dispatched ops

## Control Flow

- [ ] `scan` - iterative jaxpr application
- [ ] `associative_scan` - parallel prefix scan

## Architecture Improvements

- [ ] Cache jaxpr analysis for repeated calls

## Conservative Propagators

These propagators use conservative fallbacks that could be made precise:

- [ ] `prop_reshape` - Size mismatch unions all dependencies
- [ ] `prop_conservative_fallback` - Fallback for unhandled primitives (dot_general, pad, transpose, etc.)

## Tests Using Conservative Fallbacks

These tests verify conservative behavior that could be made precise:
- [x] `test_transpose_2d` - transpose produces dense, should be permutation matrix
- [x] `test_matmul` - dot_general produces dense, should track row/column deps
- [x] `test_reverse` - rev produces dense, should be anti-diagonal permutation
- [x] `test_pad` - pad produces dense, should be sparse (pad values have no deps)
- [ ] `test_tile` - broadcast_in_dim produces dense, should track mod pattern
- [x] `test_iota_eye` - dot_general now precise (still dense due to contraction unions)
- [ ] `test_stack` - block-wise deps instead of per-element (reshape limitation)

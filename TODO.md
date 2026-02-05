# TODO - Next Steps for asdex

## Immediate

- [ ] Support multiple input arrays via input shape (currently assumes single 1D input)
- [ ] Support multi-dimensional outputs with proper indexing
- [ ] Add more test cases from SparseConnectivityTracer's "Jacobian Global" testset: https://github.com/adrhill/SparseConnectivityTracer.jl/blob/main/test/test_gradient.jl
- [ ] Handle `dynamic_slice` primitive precisely
- [ ] Handle `gather` and `scatter` with static indices

## Primitive Coverage

Missing precise handlers for:
- [ ] `transpose` - track dimension permutation (`test_transpose_2d`)
- [ ] `dot_general` - matrix multiply sparsity (`test_matmul`, `test_iota_eye`)
- [ ] `reduce_*` variants (max, min, prod with axes)
- [ ] `select` / `cond` - branch analysis
- [ ] `rev` - reverse/flip array (`test_reverse`)
- [ ] `pad` - constant padding (`test_pad`)
- [ ] `gather` - fancy indexing (`test_gather_fancy_indexing`)
- [ ] `scatter` - in-place updates (`test_scatter_at_set`)
- [ ] `dynamic_slice` - used by split (`test_split`)
- [ ] `iota` - constant array generation, add to ZERO_DERIVATIVE_PRIMITIVES (`test_iota_eye`)
- [ ] `argmax`/`argmin` - add to ZERO_DERIVATIVE_PRIMITIVES (`test_argmax`)

## Architecture Improvements

- [ ] Recursive jaxpr handling for `pjit`, `xla_call`, `cond`
- [ ] Cache jaxpr analysis for repeated calls

## Comparison with SCT

- [ ] Compare operator classification schemes

## Potential Extensions

- [ ] Local sparsity via Dual-number style tracking
- [ ] Integration with JAX's custom_vjp/custom_jvp
- [ ] Coloring algorithms for efficient Jacobian computation
- [ ] Export to sparse AD libraries

## Known Limitations

- Control flow unions all branches (global sparsity)
- Not all JAX primitives have precise handlers (falls back to conservative union)

## Conservative Propagators

These propagators use conservative fallbacks that could be made precise:

- [ ] `prop_reshape` - Size mismatch unions all dependencies
- [ ] `prop_conservative_fallback` - Fallback for unhandled primitives (dot_general, gather, scatter, dynamic_slice, transpose, etc.)

## Tests Using Conservative Fallbacks

These tests verify conservative behavior that could be made precise:
- [ ] `test_transpose_2d` - transpose produces dense, should be permutation matrix
- [ ] `test_matmul` - dot_general produces dense, should track row/column deps
- [ ] `test_argmax` - argmax falls to default, should have zero derivative
- [ ] `test_gather_fancy_indexing` - gather produces dense, should be permutation
- [ ] `test_reverse` - rev produces dense, should be anti-diagonal permutation
- [ ] `test_pad` - pad produces dense, should be sparse (pad values have no deps)
- [ ] `test_tile` - broadcast_in_dim produces dense, should track mod pattern
- [ ] `test_split` - dynamic_slice produces dense, should preserve structure
- [ ] `test_scatter_at_set` - scatter is partially precise but not fully accurate
- [ ] `test_iota_eye` - iota + dot_general produce dense, should be identity
- [ ] `test_stack` - block-wise deps instead of per-element (reshape limitation)

## References

- SparseConnectivityTracer.jl: https://github.com/adrhill/SparseConnectivityTracer.jl
- JAX jaxpr docs: https://jax.readthedocs.io/en/latest/jaxpr.html
- SparseDiffTools.jl for coloring algorithms

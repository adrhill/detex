# TODO - Next Steps for detex

## Immediate

- [ ] Add more test cases from SparseConnectivityTracer's "Jacobian Global" testset: https://github.com/adrhill/SparseConnectivityTracer.jl/blob/main/test/test_gradient.jl
- [ ] Add more test cases (matrix operations, neural network layers)
- [ ] Handle `dynamic_slice` primitive precisely
- [ ] Handle `gather` and `scatter` with static indices

## Primitive Coverage

Missing precise handlers for:
- [ ] `transpose` - track dimension permutation
- [ ] `dot_general` - matrix multiply sparsity
- [x] `conv_general_dilated` - convolution patterns (implemented)
- [ ] `reduce_*` variants (max, min, prod with axes)
- [ ] `select` / `cond` - branch analysis

## Architecture Improvements

- [ ] Support multiple input arrays (currently assumes single 1D input)
- [ ] Support multi-dimensional outputs with proper indexing
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

- Multi-dimensional slicing is conservative (see `test_multidim_slice`)
- Multi-dimensional broadcasting is conservative (see `test_array_broadcast`)
- Control flow unions all branches (global sparsity)
- Not all JAX primitives have precise handlers (falls back to conservative union)

## Conservative Propagators

These propagators in `src/detex/_propagate.py` use conservative fallbacks that could be made precise:

- [ ] `_propagate_slice` (lines 87-93) - Multi-dimensional slice unions all dependencies
- [ ] `_propagate_broadcast_in_dim` (lines 107-110) - Non-scalar broadcast unions all dependencies
- [ ] `_propagate_reshape` (lines 127-130) - Size mismatch unions all dependencies
- [ ] `_propagate_default` (lines 287-295) - Fallback for unhandled primitives (dot_general, gather, scatter, dynamic_slice, transpose, etc.)

## Tests Using Conservative Fallbacks

These tests verify conservative behavior that could be made precise:

- [ ] `test_multidim_slice` - exercises `_propagate_slice` fallback
- [ ] `test_array_broadcast` - exercises `_propagate_broadcast_in_dim` fallback

## References

- SparseConnectivityTracer.jl: https://github.com/adrhill/SparseConnectivityTracer.jl
- JAX jaxpr docs: https://jax.readthedocs.io/en/latest/jaxpr.html
- SparseDiffTools.jl for coloring algorithms

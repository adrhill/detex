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
- [ ] `conv_general_dilated` - convolution patterns
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

- Multi-dimensional slicing is conservative
- Control flow unions all branches (global sparsity)
- Not all JAX primitives have precise handlers (falls back to conservative union)

## References

- SparseConnectivityTracer.jl: https://github.com/adrhill/SparseConnectivityTracer.jl
- JAX jaxpr docs: https://jax.readthedocs.io/en/latest/jaxpr.html
- SparseDiffTools.jl for coloring algorithms

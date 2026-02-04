# TODO - Next Steps for detex

## Immediate

- [ ] Add more test cases from SparseConnectivityTracer's "Jacobian Global" testset: https://github.com/adrhill/SparseConnectivityTracer.jl/blob/main/test/test_gradient.jl
- [x] Add more test cases (matrix operations, neural network layers) - see edge case tests
- [ ] Handle `dynamic_slice` primitive precisely
- [ ] Handle `gather` and `scatter` with static indices

## Bugs

- [ ] **Empty array concatenate bug** (`test_empty_concatenate`): Empty arrays in concatenate produce invalid COO indices, causing `ValueError: index exceeds matrix dimension`

## Primitive Coverage

Missing precise handlers for:
- [ ] `transpose` - track dimension permutation (`test_transpose_2d`)
- [ ] `dot_general` - matrix multiply sparsity (`test_matmul`, `test_iota_eye`)
- [x] `conv_general_dilated` - convolution patterns (implemented)
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

- [ ] `prop_slice` (lines 87-93) - Multi-dimensional slice unions all dependencies
- [ ] `prop_broadcast_in_dim` (lines 107-110) - Non-scalar broadcast unions all dependencies
- [ ] `prop_reshape` (lines 127-130) - Size mismatch unions all dependencies
- [ ] `prop_default` (lines 287-295) - Fallback for unhandled primitives (dot_general, gather, scatter, dynamic_slice, transpose, etc.)

## Tests Using Conservative Fallbacks

These tests verify conservative behavior that could be made precise:

### Existing tests
- [ ] `test_multidim_slice` - exercises `prop_slice` fallback
- [ ] `test_array_broadcast` - exercises `prop_broadcast_in_dim` fallback

### New edge case tests (conservative fallback)
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

### Tests that work correctly
- [x] `test_roll` - correctly tracks cyclic permutation (precise!)
- [x] `test_reduce_max` - correctly produces dense (global sparsity is correct)
- [x] `test_sort` - correctly produces dense (sorting is global)
- [x] `test_nested_slice_concat` - 1D slices + concatenate preserve precise structure

## References

- SparseConnectivityTracer.jl: https://github.com/adrhill/SparseConnectivityTracer.jl
- JAX jaxpr docs: https://jax.readthedocs.io/en/latest/jaxpr.html
- SparseDiffTools.jl for coloring algorithms

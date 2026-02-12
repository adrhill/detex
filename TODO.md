# TODO - Next Steps for asdex

## Immediate

- [ ] Add more test cases from SparseConnectivityTracer's "Jacobian Global" testset: 
    - @testset "Global Jacobian" from https://raw.githubusercontent.com/adrhill/SparseConnectivityTracer.jl/refs/heads/main/test/test_gradient.jl
    - @testset "Global Hessian" from https://raw.githubusercontent.com/adrhill/SparseConnectivityTracer.jl/refs/heads/main/test/test_hessian.jl

## Primitive Coverage

Missing precise handlers for:
- [x] `top_k` - reduction along last axis for values, zero derivative for indices
- [x] `scatter_mul`, `scatter_max`, `scatter_min` - extend existing `prop_scatter`
- [ ] `platform_index` - used by `jnp.diag` and other platform-dispatched ops

## Control Flow

- [ ] `scan` - iterative jaxpr application
- [ ] `associative_scan` - parallel prefix scan

## Architecture Improvements

- [ ] Cache jaxpr analysis for repeated calls

## Conservative Propagators

These propagators use conservative fallbacks that could be made precise:

- [ ] `prop_reshape` - Size mismatch unions all dependencies
- [ ] `prop_conservative_fallback` - Fallback for unhandled primitives (reduce_max, sort, etc.)

## Tests Using Conservative Fallbacks

These tests verify conservative behavior that could be made precise:
- [ ] `test_tile` - broadcast_in_dim produces dense, should track mod pattern

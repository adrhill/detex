# TODO - Next Steps for asdex

## Usability

- [ ] Auto-squeeze scalar-valued functions with output shape `(1,)` in `hessian_sparsity` — currently fails with `TypeError: Gradient only defined for scalar-output functions`

## Conservative Propagators

These propagators use conservative fallbacks that could be made precise:

- [ ] `prop_reshape` - Size mismatch unions all dependencies
- [ ] `prop_conservative_fallback` - Fallback for unhandled primitives (reduce_max, sort, etc.)

## Tests Using Conservative Fallbacks

These tests verify conservative behavior that could be made precise:
- [ ] `test_tile` - broadcast_in_dim produces dense, should track mod pattern

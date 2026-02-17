# TODO

## Conservative fallback primitives

These primitives in `_interpret/__init__.py` use the conservative fallback.

### Correctly conservative

- **`pure_callback`**: Arbitrary Python callback via `jax.pure_callback()`.
  Opaque function body — no way to determine sparsity.
- **`nonbatchable`**: Annotation from `jax.custom_batching.custom_vmap`
  marking non-batched args. Opaque custom batching rule.

### Done

- **`select_if_vmap`**: Vmapped `lax.cond` lowered to element-wise select.
  Precise handler added — structurally identical to `select_n`.

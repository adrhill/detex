# Handler Hardening — PR #51 Post-Mortem

Handlers that silently fall back to conservative for untested cases.
Identified during the PR #51 post-mortem.

## High Priority

- [ ] `_gather.py`: only handles dim-0 gather precisely.
      `x[:, indices]` (gather along dim 1+) silently falls back to conservative.
      Plan: `.plan/gather.md`
- [ ] `_scatter.py`: handles two patterns (batched dim-0 scatter, full-window single-dim scatter).
      Multi-index scatters like `x[rows, cols]` silently fall back to conservative.
      Already marked `@pytest.mark.fallback` in `test_internals.py`.
      Plan: `.plan/scatter.md`

## Medium Priority

- [ ] `_conv.py`: grouped and depthwise convolutions fall back to conservative
      (`feature_group_count > 1` or `batch_group_count > 1`).
      Acknowledged in the docstring but not tracked here until now.
      Plan: `.plan/conv.md`

## Low Priority (tests only)

- [ ] `_dot_general.py`: handler is correct, but test suite needs
      conservative audit, expanded `jax.jacobian` verification, and batch broadcasting.
      Plan: `.plan/dot_general.md`
- [ ] `_pad.py`: handler is correct, but test suite needs
      `jax.jacobian` verification, non-square shapes, and 3D coverage.
      Plan: `.plan/pad.md`

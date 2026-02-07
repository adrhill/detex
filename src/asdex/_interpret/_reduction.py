"""Propagation rules for reduction operations."""

from jax._src.core import JaxprEqn

from ._commons import (
    Deps,
    IndexSets,
    atom_shape,
    flat_to_coords,
    index_sets,
    numel,
    row_strides,
    union_all,
)


def prop_reduce_sum(eqn: JaxprEqn, deps: Deps) -> None:
    """Sum reduction aggregates elements along specified axes.
    Each output depends on all input elements that were summed into it.

    This handles the ``reduce_sum`` primitive
    (emitted by ``jnp.sum``).
    The general ``jax.lax.reduce`` takes a ``computation`` function
    and a ``dimensions`` param;
    the ``reduce_sum`` primitive specializes this to addition
    and uses ``axes`` instead of ``dimensions``.

    Full reduction (no axes or all axes):
        out = Σᵢ x[i]  →  out depends on all inputs
    Partial reduction along axis k:
        out[i] = Σⱼ x[i, j]  →  out[i] depends on row i of input

    Example: y = sum(x, axis=1) where x.shape = (2, 3)
        Input deps:  [{0}, {1}, {2}, {3}, {4}, {5}]
        Output deps: [{0, 1, 2}, {3, 4, 5}]  (one set per row)

    Jaxpr:
        invars[0]: input array
        axes: tuple of axes to reduce (empty = full reduction)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce.html
    """
    in_indices = index_sets(deps, eqn.invars[0])
    axes = eqn.params.get("axes", ())
    in_shape = atom_shape(eqn.invars[0])

    # Full reduction: single output depends on all inputs
    if not axes or len(axes) == len(in_shape):
        deps[eqn.outvars[0]] = [union_all(in_indices)]
        return

    # Partial reduction: group input elements by their non-reduced coordinates
    out_shape = tuple(s for i, s in enumerate(in_shape) if i not in axes)
    in_strides = row_strides(in_shape)
    out_strides = row_strides(out_shape)
    out_size = numel(out_shape)

    out_indices: IndexSets = [set() for _ in range(out_size)]

    for in_flat, elem_deps in enumerate(in_indices):
        in_coord = flat_to_coords(in_flat, in_strides)

        # Project to output coordinates (drop reduced dimensions)
        out_coord = [c for i, c in enumerate(in_coord) if i not in axes]
        out_flat = sum(c * s for c, s in zip(out_coord, out_strides, strict=True))
        out_indices[out_flat] |= elem_deps

    deps[eqn.outvars[0]] = out_indices

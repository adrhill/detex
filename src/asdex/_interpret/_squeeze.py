"""Propagation rule for squeeze operations."""

from jax._src.core import JaxprEqn

from ._commons import Deps, index_sets


def prop_squeeze(eqn: JaxprEqn, deps: Deps) -> None:
    """Squeeze removes dimensions of size 1 without changing the data.
    Since it's a reshape with the same number of elements,
    dependencies pass through unchanged in flat order.

    For input shape (2, 1, 3) with squeeze on dim 1:
        out[i, k] = in[i, 0, k]
    The Jacobian is the identity matrix (permuted).

    Example: x.shape = (2, 1), y = squeeze(x) with shape (2,)
        Input deps:  [{0}, {1}]
        Output deps: [{0}, {1}]

    Jaxpr:
        invars[0]: input array
        dimensions: axes to squeeze (must have size 1)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.squeeze.html
    """
    deps[eqn.outvars[0]] = index_sets(deps, eqn.invars[0])

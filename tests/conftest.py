"""Pytest configuration and fixtures for detex tests."""


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "elementwise: simple element-wise operations")
    config.addinivalue_line(
        "markers", "array_ops: array manipulation (slice, concat, reshape, etc.)"
    )
    config.addinivalue_line(
        "markers", "control_flow: conditional operations (where, select)"
    )
    config.addinivalue_line(
        "markers", "reduction: reduction operations (sum, max, prod)"
    )
    config.addinivalue_line("markers", "vmap: batched/vmapped operations")
    config.addinivalue_line(
        "markers", "fallback: documents conservative fallback behavior (TODO)"
    )
    config.addinivalue_line("markers", "bug: documents known bugs")
    config.addinivalue_line("markers", "coloring: row coloring algorithm tests")
    config.addinivalue_line("markers", "jacobian: sparse Jacobian computation tests")
    config.addinivalue_line(
        "markers", "hessian: Hessian sparsity detection and computation"
    )

"""Shared helpers: construct networks directly and execute the real WebNN runtime."""

import pytest
import torch
from torch.utils import _pytree


@pytest.fixture
def seed():
    torch.manual_seed(42)
    return 42


def assert_export_matches(model, example_input, rtol=1e-5, atol=1e-5, debug=False):
    from webnn_torch_export import export_model

    model.eval()
    inputs = example_input if isinstance(example_input, tuple) else (example_input,)
    with torch.no_grad():
        expected = model(*inputs)
    exporter, _ = export_model(model, inputs, debug=debug)
    executor = exporter.to_executor()
    actual = executor(*inputs)
    expected_leaves, expected_spec = _pytree.tree_flatten(expected)
    actual_leaves, actual_spec = _pytree.tree_flatten(actual)
    assert actual_spec == expected_spec
    for a, e in zip(actual_leaves, expected_leaves):
        torch.testing.assert_close(a, e.cpu(), rtol=rtol, atol=atol)
    return executor, exporter


def assert_builds_webnn(model, example_input, debug=False):
    from webnn_torch_export import export_model

    model.eval()
    exporter, _ = export_model(model, example_input, debug=debug)
    assert exporter.webnn_graph is not None
    assert exporter.webnn_generator.checked_shapes
    return exporter


def validate_webnn_execution(model, example_input, debug=False, rtol=1e-3, atol=1e-3):
    return assert_export_matches(
        model, example_input, rtol=rtol, atol=atol, debug=debug
    )

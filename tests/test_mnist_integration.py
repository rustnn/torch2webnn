"""
Integration test: MNIST classifier with Conv, Activation, and Linear layers.
Tests a complete model export scenario.
"""

import pytest
import torch
from .models import MNISTClassifier, SimplerMNISTClassifier
from .conftest import assert_export_matches, validate_webnn_execution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simpler_mnist_model():
    return SimplerMNISTClassifier()


@pytest.fixture
def full_mnist_model():
    return MNISTClassifier()


@pytest.fixture
def mnist_input():
    return torch.randn(4, 1, 28, 28)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_simple_mnist_export(simpler_mnist_model, mnist_input):
    """Test exporting a simplified MNIST classifier"""
    torch._dynamo.reset()
    compiled_model, exporter = assert_export_matches(
        simpler_mnist_model, mnist_input, rtol=1e-3, atol=1e-5
    )

    with torch.no_grad():
        output = compiled_model(mnist_input)
    assert output.shape == (4, 10)
    assert len(exporter.exported_graphs) > 0

    validate_webnn_execution(simpler_mnist_model, mnist_input)


def test_full_mnist_export(full_mnist_model, mnist_input):
    """Test exporting a full MNIST classifier"""
    torch._dynamo.reset()
    compiled_model, _ = assert_export_matches(
        full_mnist_model, mnist_input, rtol=1e-3, atol=1e-5
    )

    with torch.no_grad():
        output = compiled_model(mnist_input)
    assert output.shape == (4, 10)

    validate_webnn_execution(full_mnist_model, mnist_input)


def test_mnist_inference_consistency(simpler_mnist_model):
    """Test that the exported model produces consistent results across multiple forward passes"""
    torch._dynamo.reset()
    inputs = [torch.randn(1, 1, 28, 28) for _ in range(3)]
    compiled_model, _ = assert_export_matches(simpler_mnist_model, inputs[0], rtol=1e-3, atol=1e-5)

    simpler_mnist_model.eval()
    for x in inputs[1:]:
        with torch.no_grad():
            expected = simpler_mnist_model(x)
            actual = compiled_model(x)
        assert torch.allclose(expected, actual, rtol=1e-3, atol=1e-5)

    validate_webnn_execution(simpler_mnist_model, inputs[0])


def test_mnist_graph_contains_expected_ops(simpler_mnist_model):
    """Test that exported MNIST graph contains expected operation types"""
    torch._dynamo.reset()
    x = torch.randn(1, 1, 28, 28)
    _, exporter = assert_export_matches(simpler_mnist_model, x)

    nodes = exporter.exported_graphs[0]['nodes']
    op_types = {node['op'] for node in nodes}

    assert 'placeholder' in op_types
    assert 'output' in op_types
    assert 'call_function' in op_types or 'call_module' in op_types

    validate_webnn_execution(simpler_mnist_model, x)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_mnist_batch_size_invariance(batch_size):
    """Test that model works with different batch sizes"""
    torch._dynamo.reset()
    model = SimplerMNISTClassifier()
    x = torch.randn(batch_size, 1, 28, 28)
    compiled_model, _ = assert_export_matches(model, x, rtol=1e-3, atol=1e-5)

    with torch.no_grad():
        output = compiled_model(x)
    assert output.shape == (batch_size, 10)

    validate_webnn_execution(model, x)


@pytest.mark.parametrize("model_class", [SimplerMNISTClassifier, MNISTClassifier])
def test_mnist_models(model_class):
    """Parametrized test for both MNIST model variants"""
    torch._dynamo.reset()
    model = model_class()
    x = torch.randn(2, 1, 28, 28)
    assert_export_matches(model, x, rtol=1e-3, atol=1e-5)
    validate_webnn_execution(model, x)

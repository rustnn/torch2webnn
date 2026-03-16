"""Test operations by functional category"""

import pytest
import torch
from .models import (
    PointwiseActivationsModel,
    PointwiseArithmeticModel,
    PointwiseMathModel,
    ReductionOpsModel,
    ShapeOpsModel,
    NormalizationOpsModel,
)
from .conftest import assert_export_matches, validate_webnn_execution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def activations_model():
    return PointwiseActivationsModel()


@pytest.fixture
def arithmetic_model():
    return PointwiseArithmeticModel()


@pytest.fixture
def math_model():
    return PointwiseMathModel()


@pytest.fixture
def reduction_model():
    return ReductionOpsModel()


@pytest.fixture
def shape_model():
    return ShapeOpsModel()


@pytest.fixture
def normalization_model():
    return NormalizationOpsModel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pointwise_activations(activations_model):
    """Test pointwise activation functions (sigmoid, tanh, softmax)"""
    x = torch.randn(1, 10) * 0.5
    assert_export_matches(activations_model, x, rtol=1e-4, atol=1e-4)
    validate_webnn_execution(activations_model, x, rtol=1e-4, atol=1e-4)


def test_pointwise_arithmetic(arithmetic_model):
    """Test pointwise arithmetic operations (add, sub, mul, div)"""
    x = torch.randn(1, 10) + 2
    y = torch.randn(1, 10) + 2
    assert_export_matches(arithmetic_model, (x, y), rtol=1e-4, atol=1e-4)
    validate_webnn_execution(arithmetic_model, (x, y), rtol=1e-4, atol=1e-4)


def test_pointwise_math(math_model):
    """Test pointwise math functions (abs, exp, log, sqrt)"""
    x = torch.randn(1, 10).abs() + 0.1
    assert_export_matches(math_model, x, rtol=1e-4, atol=1e-4)
    validate_webnn_execution(math_model, x, rtol=1e-4, atol=1e-4)


def test_reduction_ops(reduction_model):
    """Test reduction operations (avg_pool2d, max_pool2d)"""
    x = torch.randn(1, 3, 8, 8)
    assert_export_matches(reduction_model, x, rtol=1e-4, atol=1e-4)
    validate_webnn_execution(reduction_model, x, rtol=1e-4, atol=1e-4)


def test_shape_ops(shape_model):
    """Test shape manipulation operations (transpose, reshape)"""
    x = torch.randn(2, 3, 10)
    assert_export_matches(shape_model, x, rtol=1e-4, atol=1e-4)
    validate_webnn_execution(shape_model, x, rtol=1e-4, atol=1e-4)


def test_normalization_ops(normalization_model):
    """Test normalization operations (layer_norm)"""
    x = torch.randn(2, 10)
    assert_export_matches(normalization_model, x, rtol=1e-4, atol=1e-4)
    validate_webnn_execution(normalization_model, x, rtol=1e-4, atol=1e-4)

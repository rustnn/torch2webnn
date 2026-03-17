"""Test operations by functional category"""

import pytest
import torch
import torch.nn as nn
from .models import (
    ConcatModel,
    NormalizationOpsModel,
    PointwiseActivationsModel,
    PointwiseArithmeticModel,
    PointwiseMathModel,
    ReductionOpsModel,
    ShapeOpsModel,
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


# ---------------------------------------------------------------------------
# Chunk + getitem
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chunks,dim,shape", [
    (2, -1, (2, 8)),
    (3,  1, (1, 6, 4)),
    (6, -1, (1, 12)),   # Flux pattern: 6 chunks along last dim
], ids=["2chunks_last", "3chunks_dim1", "6chunks_flux"])
def test_chunk(chunks, dim, shape):
    """chunk splits a tensor and getitem indexes into the result."""
    torch._dynamo.reset()
    class ChunkSum(nn.Module):
        def __init__(self, n, d):
            super().__init__()
            self.n, self.d = n, d
        def forward(self, x):
            return sum(torch.chunk(x, self.n, dim=self.d))
    model = ChunkSum(chunks, dim)
    x = torch.randn(*shape)
    assert_export_matches(model, x, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Unbind + getitem
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim,shape", [
    (0, (3, 4)),
    (1, (2, 4, 5)),
], ids=["dim0", "dim1"])
def test_unbind(dim, shape):
    """unbind returns individual slices; getitem indexes them."""
    torch._dynamo.reset()
    class UnbindSum(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d = d
        def forward(self, x):
            return sum(torch.unbind(x, dim=self.d))
    model = UnbindSum(dim)
    x = torch.randn(*shape)
    assert_export_matches(model, x, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# select.int
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim,index,shape", [
    (0,  1, (4, 5)),
    (-1, 0, (2, 3, 4)),
    (2,  0, (2, 4, 6, 8)),
], ids=["dim0", "dim_neg", "higher_dim"])
def test_select(dim, index, shape):
    """select picks a single element along dim, removing that dimension."""
    torch._dynamo.reset()
    class SelectModel(nn.Module):
        def __init__(self, d, i):
            super().__init__()
            self.d, self.i = d, i
        def forward(self, x):
            return torch.select(x, self.d, self.i)
    model = SelectModel(dim, index)
    x = torch.randn(*shape)
    assert_export_matches(model, x, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# stack  (list passed as args[0])
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim,shape", [
    (0,  (4, 3)),
    (-1, (2, 3, 3)),
], ids=["dim0", "dim_neg"])
def test_stack(dim, shape):
    """stack assembles tensors along a new dimension."""
    torch._dynamo.reset()
    class StackModel(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d = d
        def forward(self, x):
            a, b, c = x[..., 0], x[..., 1], x[..., 2]
            return torch.stack([a, b, c], dim=self.d)
    model = StackModel(dim)
    x = torch.randn(*shape)
    assert_export_matches(model, x, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# concat  (aten.cat)
# ---------------------------------------------------------------------------

# 4-D shape (2, 3, 4, 5) — supports axes 0, 1, 3 and their negative equivalents
@pytest.mark.parametrize("n_inputs,axis", [
    (2, 0), (2, 1), (2, 3), (2, -1), (2, -3),
    (5, 0), (5, 1), (5, 3), (5, -1), (5, -3),
], ids=[f"{n}in_ax{a}" for n, a in [
    (2, 0), (2, 1), (2, 3), (2, -1), (2, -3),
    (5, 0), (5, 1), (5, 3), (5, -1), (5, -3),
]])
def test_concat(n_inputs, axis):
    torch._dynamo.reset()
    model = ConcatModel(axis)
    inputs = tuple(torch.randn(2, 3, 4, 5) for _ in range(n_inputs))
    assert_export_matches(model, inputs)
    validate_webnn_execution(model, inputs)


# ---------------------------------------------------------------------------
# type_as
# ---------------------------------------------------------------------------

def test_type_as_same_dtype():
    """type_as with matching dtype becomes an identity."""
    torch._dynamo.reset()
    class TypeAsModel(nn.Module):
        def forward(self, x, ref):
            return x.type_as(ref)
    model = TypeAsModel()
    x = torch.randn(3, 4)
    ref = torch.randn(1)
    assert_export_matches(model, (x, ref), rtol=1e-5, atol=1e-5)

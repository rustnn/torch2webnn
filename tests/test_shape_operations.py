"""Test shape operation export"""

import pytest
import torch
import torch._dynamo
from .models import RearrangeModel, SplitEven, SplitWithSizes
from .conftest import assert_export_matches, assert_generates_webnn, validate_webnn_execution


def test_rearrange_export():
    """Test exporting a model with einops rearrange (depth-to-space)"""
    rearrange_model = RearrangeModel()
    rearrange_input = torch.randn(1, 12, 4, 4)
    compiled_model, _ = assert_export_matches(rearrange_model, rearrange_input, rtol=1e-5, atol=1e-5)

    # Verify output shape: [1, 12, 4, 4] -> [1, 3, 8, 8]
    with torch.no_grad():
        output = compiled_model(rearrange_input)
    assert output.shape == (1, 3, 8, 8)

    validate_webnn_execution(rearrange_model, rearrange_input, rtol=1e-5, atol=1e-5)

# ---------------------------------------------------------------------------
# Even split (aten.split.Tensor)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split_size,dim,shape", [
    (2,  0, (6, 4)),       # 3 chunks along dim 0
    (4,  1, (2, 8, 3)),    # 2 chunks along dim 1
    (1, -1, (3, 4)),       # 4 chunks along last dim (negative index)
])
def test_split_even_export(split_size, dim, shape):
    torch._dynamo.reset()
    model = SplitEven(split_size, dim)
    x = torch.randn(*shape)
    assert_export_matches(model, x, rtol=1e-5, atol=1e-5)
    text = assert_generates_webnn(model, x)
    assert "slice" in text
    assert "split" not in text


# ---------------------------------------------------------------------------
# Uneven / variable-size split (aten.split_with_sizes.default)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split_sizes,dim,shape", [
    ([2, 3, 1],    0, (6, 4)),    # uneven along dim 0
    ([3, 5],       1, (2, 8, 3)), # uneven along dim 1
    ([1, 2, 1],   -1, (3, 4)),    # uneven along last dim
])
def test_split_with_sizes_export(split_sizes, dim, shape):
    torch._dynamo.reset()
    model = SplitWithSizes(split_sizes, dim)
    x = torch.randn(*shape)
    assert_export_matches(model, x, rtol=1e-5, atol=1e-5)
    text = assert_generates_webnn(model, x)
    assert "slice" in text
    assert "split" not in text


# ---------------------------------------------------------------------------
# WebNN execution tests
# ---------------------------------------------------------------------------

def test_split_even_webnn():
    torch._dynamo.reset()
    model = SplitEven(split_size=2, dim=0)
    x = torch.randn(6, 4)
    validate_webnn_execution(model, x, rtol=1e-4, atol=1e-4)


def test_split_with_sizes_webnn():
    torch._dynamo.reset()
    model = SplitWithSizes([2, 3, 1], dim=0)
    x = torch.randn(6, 4)
    validate_webnn_execution(model, x, rtol=1e-4, atol=1e-4)

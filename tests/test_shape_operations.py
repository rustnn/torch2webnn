"""Test shape operation export"""

import pytest
import torch
from .models import RearrangeModel
from .conftest import assert_export_matches, validate_webnn_execution


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

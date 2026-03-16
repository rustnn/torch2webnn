"""Test exporting models with multiple inputs"""

import pytest
import torch
import torch.nn as nn
from .models import MultiInputModel
from .conftest import assert_export_matches, validate_webnn_execution


def test_multiple_inputs():
    """Test exporting a model with multiple inputs"""
    x1, x2 = torch.randn(1, 10), torch.randn(1, 15)
    multi_input_model = MultiInputModel()
    assert_export_matches(multi_input_model, (x1, x2), rtol=1e-5, atol=1e-5)
    validate_webnn_execution(multi_input_model, (x1, x2), rtol=1e-5, atol=1e-5)

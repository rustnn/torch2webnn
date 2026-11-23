"""Test arange constant generation during model export"""

import pytest
import torch

from .conftest import assert_export_matches, validate_webnn_execution
from .models import ArangeModel


@pytest.fixture
def arange_model():
    return ArangeModel()


@pytest.fixture
def arange_input():
    return torch.randn(1, 10)

def test_arange(arange_model, arange_input):
    """Validate end-to-end WebNN execution for arange model."""
    assert_export_matches(arange_model, arange_input, rtol=1e-5, atol=1e-5)
    validate_webnn_execution(arange_model, arange_input, rtol=1e-5, atol=1e-5)

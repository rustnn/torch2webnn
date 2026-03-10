"""
Tests for batch normalization export.
Covers: BatchNorm2d with and without running mean/var.
"""

import pytest
import torch
from .models import SingleBatchNorm2d, SingleBatchNorm2dNoRunning
from .conftest import assert_export_matches, validate_webnn_execution


BATCH_NORM_CONFIGS = [
    (16, (1, 16, 8, 8)),
    (32, (1, 32, 4, 4)),
]


@pytest.mark.parametrize(
    "num_features,input_shape",
    BATCH_NORM_CONFIGS,
    ids=[f"bn_{c}c_{s[2]}x{s[3]}" for c, s in BATCH_NORM_CONFIGS],
)
def test_batch_norm_with_running_stats(num_features, input_shape):
    torch._dynamo.reset()
    model = SingleBatchNorm2d(num_features)
    x = torch.randn(*input_shape)
    assert_export_matches(model, x, rtol=1e-4, atol=1e-4)
    validate_webnn_execution(model, x, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "num_features,input_shape",
    BATCH_NORM_CONFIGS,
    ids=[f"bn_norunning_{c}c_{s[2]}x{s[3]}" for c, s in BATCH_NORM_CONFIGS],
)
def test_batch_norm_without_running_stats(num_features, input_shape):
    torch._dynamo.reset()
    model = SingleBatchNorm2dNoRunning(num_features)
    x = torch.randn(*input_shape)
    assert_export_matches(model, x, rtol=1e-4, atol=1e-4)
    validate_webnn_execution(model, x, rtol=1e-4, atol=1e-4)
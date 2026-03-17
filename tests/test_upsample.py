"""Tests for upsample / interpolate operations."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conftest import assert_export_matches, validate_webnn_execution


# ---------------------------------------------------------------------------
# Parametrize over (id, scale_or_size kwarg, mode)
# ---------------------------------------------------------------------------

UPSAMPLE_CASES = [
    # nearest — scale_factor
    ("nearest_scale2",   dict(scale_factor=2.0,        mode="nearest"),   (1, 4, 8, 8)),
    ("nearest_scale3",   dict(scale_factor=3.0,        mode="nearest"),   (1, 2, 4, 4)),
    ("nearest_scale_xy", dict(scale_factor=(2.0, 3.0), mode="nearest"),   (1, 4, 6, 4)),
    # nearest — explicit output size
    ("nearest_size",     dict(size=(16, 16),            mode="nearest"),   (1, 4, 8, 8)),
    # bilinear — scale_factor
    ("bilinear_scale2",  dict(scale_factor=2.0,         mode="bilinear", align_corners=False), (1, 4, 8, 8)),
    # bilinear — explicit output size
    ("bilinear_size",    dict(size=(20, 20),             mode="bilinear", align_corners=False), (1, 4, 10, 10)),
]


@pytest.mark.parametrize(
    "interp_kwargs,input_shape",
    [(c[1], c[2]) for c in UPSAMPLE_CASES],
    ids=[c[0] for c in UPSAMPLE_CASES],
)
def test_upsample(interp_kwargs, input_shape):
    class UpsampleModel(nn.Module):
        def forward(self, x):
            return F.interpolate(x, **interp_kwargs)

    model = UpsampleModel()
    x = torch.randn(*input_shape)
    assert_export_matches(model, x, rtol=1e-5, atol=1e-5)
    validate_webnn_execution(model, x, rtol=1e-4, atol=1e-4)

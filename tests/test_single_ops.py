"""
Parametrized tests for single-operator exports.
Covers: Conv2d, matmul, Linear × multiple configurations.
"""

import pytest
import torch
from .models import SingleConv, SingleMatmul, SingleLinear, SingleMM, SingleAddMM, SingleScaledDotProduct
from .conftest import assert_export_matches, validate_webnn_execution


class ConvOpConfig:
    def __init__(self, in_cannels, out_channels, kernel_size, stride, padding):
        self.in_cannels = in_cannels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def name(self):
        return f"conv_{self.in_cannels}x{self.out_channels}_{self.kernel_size}x{self.kernel_size}"


CONV_OPS = [ # config, input_shape
    (ConvOpConfig(16, 32, 3, stride=1, padding=1), (1, 16, 10, 10)),
    (ConvOpConfig(1,  32, 5, stride=1, padding=2), (1, 1,  28, 28)),
    (ConvOpConfig(3,  64, 3, stride=1, padding=1), (1, 3,  28, 28)),
    (ConvOpConfig(16, 32, 1, stride=1, padding=0), (1, 16, 28, 28)),
]

class GemmOpConfig:
    def __init__(self, in_features, out_features, batch_size, model_cls):
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.model_cls = model_cls

    def name(self):
        batch = "vec" if self.batch_size is None else f"b{self.batch_size}"
        return f"{self.model_cls.__name__}_{self.in_features}x{self.out_features}_{batch}"


GEMM_OPS = [ # config, input_shape
    (GemmOpConfig(784, 128,  32, SingleMatmul), (32, 784)),
    (GemmOpConfig(784, 128,  32, SingleLinear), (32, 784)),
    (GemmOpConfig(256,  64,   8, SingleMatmul), (8,  256)),
    (GemmOpConfig(512, 256,  16, SingleLinear), (16, 512)),
    (GemmOpConfig(128,  64, None, SingleMatmul), (128,)),   # mat*vec
    (GemmOpConfig(256,  64,   8, SingleMM),     (8,  256)),
    (GemmOpConfig(512, 128,  16, SingleMM),     (16, 512)),
    (GemmOpConfig(256,  64,   8, SingleAddMM),  (8,  256)),
    (GemmOpConfig(512, 128,  16, SingleAddMM),  (16, 512)),
]


class AttentionOpConfig:
    def __init__(self, batch, heads, seq_len, head_dim):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.head_dim = head_dim

    def name(self):
        return f"sdpa_b{self.batch}_h{self.heads}_s{self.seq_len}_d{self.head_dim}"


ATTN_OPS = [ # config, qkv_shape
    (AttentionOpConfig(1, 4,  16, 32), (1, 4,  16, 32)),
    (AttentionOpConfig(2, 8,  32, 64), (2, 8,  32, 64)),
    (AttentionOpConfig(1, 1,   8, 16), (1, 1,   8, 16)),  # minimal
]


@pytest.mark.parametrize(
    "conv_config,input_shape",
    CONV_OPS,
    ids=[op[0].name() for op in CONV_OPS],
)
def test_conv_op(conv_config, input_shape):
    torch._dynamo.reset()
    model = SingleConv(conv_config.in_cannels, conv_config.out_channels, conv_config.kernel_size, conv_config.stride)
    x = torch.randn(*input_shape)
    assert_export_matches(model, x, rtol=1e-3)
    validate_webnn_execution(model, x)


@pytest.mark.parametrize(
    "gemm_config,input_shape",
    GEMM_OPS,
    ids=[op[0].name() for op in GEMM_OPS],
)
def test_gemm_op(gemm_config, input_shape):
    torch._dynamo.reset()
    model = gemm_config.model_cls(gemm_config.in_features, gemm_config.out_features)
    x = torch.randn(*input_shape)
    assert_export_matches(model, x, rtol=1e-3)
    validate_webnn_execution(model, x)


@pytest.mark.parametrize(
    "attn_config,qkv_shape",
    ATTN_OPS,
    ids=[op[0].name() for op in ATTN_OPS],
)
def test_attention_op(attn_config, qkv_shape):
    torch._dynamo.reset()
    model = SingleScaledDotProduct()
    q = torch.randn(*qkv_shape)
    k = torch.randn(*qkv_shape)
    v = torch.randn(*qkv_shape)
    assert_export_matches(model, (q, k, v), rtol=1e-3, atol=1e-3)
    validate_webnn_execution(model, (q, k, v), rtol=1e-3, atol=1e-3)


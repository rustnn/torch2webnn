"""
Parametrized tests for single-operator exports.
Covers: Conv2d, matmul, Linear × multiple configurations.
"""

import pytest
import torch
from .models import SingleConv, SingleMatmul, SingleLinear, SingleMM, SingleAddMM, SingleScaledDotProduct, SingleEinsum
from .conftest import assert_export_matches, assert_generates_webnn, validate_webnn_execution


class ConvOpConfig:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def name(self):
        parts = [f"conv_{self.in_channels}x{self.out_channels}_k{self.kernel_size}"]
        if self.stride != 1:
            parts.append(f"s{self.stride}")
        if self.dilation != 1:
            parts.append(f"d{self.dilation}")
        if not self.bias:
            parts.append("nobias")
        return "_".join(parts)


CONV_OPS = [  # (config, input_shape)
    # baseline
    (ConvOpConfig(16, 32, 3, padding=1),                          (1, 16, 10, 10)),
    (ConvOpConfig(1,  32, 5, padding=2),                          (1, 1,  28, 28)),
    (ConvOpConfig(3,  64, 3, padding=1),                          (1, 3,  28, 28)),
    (ConvOpConfig(16, 32, 1, padding=0),                          (1, 16, 28, 28)),
    # no bias
    (ConvOpConfig(16, 32, 3, padding=1, bias=False),              (1, 16, 10, 10)),
    (ConvOpConfig(3,  64, 3, padding=1, bias=False),              (1, 3,  28, 28)),
    # stride > 1
    (ConvOpConfig(16, 32, 3, stride=2, padding=1),                (1, 16, 14, 14)),
    (ConvOpConfig(3,  32, 3, stride=2, padding=1),                (1, 3,  28, 28)),
    (ConvOpConfig(16, 32, 3, stride=2, padding=1, bias=False),    (1, 16, 14, 14)),
    # dilation > 1  (padding = dilation to preserve spatial size)
    (ConvOpConfig(16, 32, 3, dilation=2, padding=2),              (1, 16, 14, 14)),
    (ConvOpConfig(3,  32, 3, dilation=2, padding=2),              (1, 3,  28, 28)),
    (ConvOpConfig(16, 32, 3, dilation=2, padding=2, bias=False),  (1, 16, 14, 14)),
]

class GemmOpConfig:
    def __init__(self, in_features, out_features, input_shape, model_cls, tag=None):
        self.in_features = in_features
        self.out_features = out_features
        self.input_shape = input_shape
        self.model_cls = model_cls
        self.tag = tag

    def name(self):
        shape_str = "x".join(str(d) for d in self.input_shape)
        base = f"{self.model_cls.__name__}_{self.in_features}x{self.out_features}_{shape_str}"
        return f"{base}_{self.tag}" if self.tag else base


GEMM_OPS = [  # (config, input_shape)
    # 2-D (classic rank-2 GEMM)
    (GemmOpConfig(784, 128, (32, 784),  SingleMatmul), (32, 784)),
    (GemmOpConfig(784, 128, (32, 784),  SingleLinear),  (32, 784)),
    (GemmOpConfig(256,  64, (8,  256),  SingleMatmul), (8,  256)),
    (GemmOpConfig(512, 256, (16, 512),  SingleLinear),  (16, 512)),
    (GemmOpConfig(128,  64, (128,),     SingleMatmul, tag="vec"), (128,)),  # mat*vec
    (GemmOpConfig(256,  64, (8,  256),  SingleMM),     (8,  256)),
    (GemmOpConfig(512, 128, (16, 512),  SingleMM),     (16, 512)),
    (GemmOpConfig(256,  64, (8,  256),  SingleAddMM),  (8,  256)),
    (GemmOpConfig(512, 128, (16, 512),  SingleAddMM),  (16, 512)),
    # 3-D batched (tests that matmul handles batch dims; gemm would fail here)
    (GemmOpConfig(256,  64, (4, 8,  256), SingleLinear,  tag="batched"), (4, 8,  256)),
    (GemmOpConfig(512, 128, (2, 16, 512), SingleMatmul,  tag="batched"), (2, 16, 512)),
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
    model = SingleConv(
        conv_config.in_channels,
        conv_config.out_channels,
        conv_config.kernel_size,
        padding=conv_config.padding,
        stride=conv_config.stride,
        dilation=conv_config.dilation,
        bias=conv_config.bias,
    )
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
    text = assert_generates_webnn(model, x)
    assert "matmul" in text, f"Expected 'matmul' in graph:\n{text}"
    assert "gemm" not in text, f"Unexpected 'gemm' in graph:\n{text}"
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


# (id, a_shape, b_shape)  — pattern '...n,d->...nd'
EINSUM_OPS = [
    ("2d_8x16",  (4, 8),    (16,)),
    ("3d_2x4x8", (2, 4, 8), (16,)),
]


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [(c[1], c[2]) for c in EINSUM_OPS],
    ids=[c[0] for c in EINSUM_OPS],
)
def test_einsum_op(a_shape, b_shape):
    torch._dynamo.reset()
    model = SingleEinsum()
    a = torch.randn(*a_shape)
    b = torch.randn(*b_shape)
    assert_export_matches(model, (a, b), rtol=1e-4, atol=1e-4)


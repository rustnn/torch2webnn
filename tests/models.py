"""
Reusable nn.Module definitions for webnn_torch_export tests.
Import from this module rather than defining models inline in test files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Single-op wrappers (used in test_single_ops.py)
# ---------------------------------------------------------------------------

class SingleConv(nn.Module):
    """Wrapper for testing single Conv2d operation"""

    def __init__(self, in_channels=3, out_channels=16, kernel_size=3, padding=1,
                 stride=1, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, stride=stride, dilation=dilation, bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class SingleMatmul(nn.Module):
    """Wrapper for testing single matmul operation"""

    def __init__(self, in_features=784, out_features=128):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias


class SingleLinear(nn.Module):
    """Wrapper for testing nn.Linear (which uses matmul internally)"""

    def __init__(self, in_features=784, out_features=128):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        return self.linear(x)


class SingleMM(nn.Module):
    """Wrapper for testing torch.mm (strict 2-D multiply, no bias)."""

    def __init__(self, in_features=784, out_features=128):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x):
        return torch.mm(x, self.weight)


class SingleAddMM(nn.Module):
    """Wrapper for testing torch.addmm (bias + mat1 @ mat2)."""

    def __init__(self, in_features=784, out_features=128):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return torch.addmm(self.bias, x, self.weight)


class SingleScaledDotProduct(nn.Module):
    """Wrapper for testing F.scaled_dot_product_attention(q, k, v)."""

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)


class SingleEinsum(nn.Module):
    """Wrapper for testing torch.einsum with the '...n,d->...nd' pattern (used in Flux RoPE)."""

    def forward(self, a, b):
        return torch.einsum("...n,d->...nd", a, b)


# ---------------------------------------------------------------------------
# Multiple-input models (used in test_multiple_inputs.py)
# ---------------------------------------------------------------------------

class MultiInputModel(nn.Module):
    """Simple model that takes multiple inputs"""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(15, 20)

    def forward(self, x1, x2):
        out1 = self.linear1(x1)
        out2 = self.linear2(x2)
        return out1 + out2


# ---------------------------------------------------------------------------
# Operation-category models (used in test_operations.py)
# ---------------------------------------------------------------------------

class PointwiseActivationsModel(nn.Module):
    """Test pointwise activation functions"""
    def forward(self, x):
        x = torch.sigmoid(x)
        x = torch.tanh(x)
        x = F.softmax(x, dim=-1)
        return x


class PointwiseArithmeticModel(nn.Module):
    """Test pointwise arithmetic operations"""
    def forward(self, x, y):
        a = x + y
        b = x - y
        c = x * y
        d = x / y
        return a + b + c + d


class PointwiseMathModel(nn.Module):
    """Test pointwise math functions"""
    def forward(self, x):
        x = torch.abs(x)
        x = torch.exp(x)
        x = torch.log(x + 1)
        x = torch.sqrt(x)
        return x


class ReductionOpsModel(nn.Module):
    """Test reduction operations (pooling)"""
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.maxpool(x)
        return x


class ShapeOpsModel(nn.Module):
    """Test shape manipulation operations"""
    def forward(self, x):
        x = x.transpose(1, 2)
        batch = x.shape[0]
        x = x.reshape(batch, -1)
        return x


class NormalizationOpsModel(nn.Module):
    """Test normalization operations"""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(10)

    def forward(self, x):
        return self.norm(x)


# ---------------------------------------------------------------------------
# Concat models (used in test_concat.py)
# ---------------------------------------------------------------------------

class ConcatModel(nn.Module):
    """Concatenate any number of tensors along the given axis."""

    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, *tensors):
        return torch.cat(tensors, dim=self.axis)


# ---------------------------------------------------------------------------
# Batch norm models (used in test_batch_norm.py)
# ---------------------------------------------------------------------------

class SingleBatchNorm2d(nn.Module):
    """BatchNorm2d with running mean/var (eval mode uses pre-computed stats)."""

    def __init__(self, num_features: int = 16):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.bn(x)


class SingleBatchNorm2dNoRunning(nn.Module):
    """BatchNorm2d without running stats (always computes from the current batch)."""

    def __init__(self, num_features: int = 16):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, track_running_stats=False)

    def forward(self, x):
        return self.bn(x)


# ---------------------------------------------------------------------------
# Rearrange model (used in test_rearrange.py)
# ---------------------------------------------------------------------------

class RearrangeModel(nn.Module):
    """Simple model that uses einops rearrange (depth-to-space / pixel shuffle)"""

    def forward(self, x):
        # Input: [batch, c*4, h, w] -> Output: [batch, c, h*2, w*2]
        return rearrange(x, 'b (c pi pj) h w -> b c (h pi) (w pj)', pi=2, pj=2)


# ---------------------------------------------------------------------------
# Arange model (used in test_arange.py)
# ---------------------------------------------------------------------------

class ArangeModel(nn.Module):
    """Simple model that uses arange as a constant offset"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        indices = torch.arange(0, 10, dtype=torch.float32)
        return self.linear(x) + indices


# ---------------------------------------------------------------------------
# MNIST classifiers (used in test_mnist_integration.py)
# ---------------------------------------------------------------------------

class SimplerMNISTClassifier(nn.Module):
    """
    Simplified MNIST classifier for easier debugging:
    1 Conv + ReLU + 1 Linear
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SplitEven(nn.Module):
    """Split tensor into equal-sized chunks along a given dim."""

    def __init__(self, split_size: int, dim: int = 0):
        super().__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, x):
        chunks = torch.split(x, self.split_size, dim=self.dim)
        # Sum all chunks so the model has a single tensor output
        out = chunks[0]
        for c in chunks[1:]:
            out = out + c
        return out


class SplitWithSizes(nn.Module):
    """Split tensor into variable-sized sections along a given dim."""

    def __init__(self, split_sizes, dim: int = 0):
        super().__init__()
        self.split_sizes = split_sizes
        self.dim = dim

    def forward(self, x):
        chunks = torch.split(x, self.split_sizes, dim=self.dim)
        # Concatenate back so the model has a single tensor output
        return torch.cat(chunks, dim=self.dim)


class MNISTClassifier(nn.Module):
    """
    Full MNIST classifier:
    2 × (Conv + ReLU + MaxPool) + 2 Linear layers
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Activation function wrappers
# ---------------------------------------------------------------------------

class SingleGELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class SingleSiLU(nn.Module):
    def forward(self, x):
        return F.silu(x)


class SingleHardtanh(nn.Module):
    def forward(self, x):
        return F.hardtanh(x, min_val=-1.0, max_val=1.0)


class SingleClamp(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=-1.0, max=1.0)


# ---------------------------------------------------------------------------
# Elementwise math wrappers
# ---------------------------------------------------------------------------

class SingleNeg(nn.Module):
    def forward(self, x):
        return torch.neg(x)


class SingleCos(nn.Module):
    def forward(self, x):
        return torch.cos(x)


class SingleSin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class SinglePowTensor(nn.Module):
    """aten.pow.Tensor_Scalar — tensor raised to a scalar exponent."""
    def forward(self, x):
        return x ** 2


# ---------------------------------------------------------------------------
# Shape manipulation wrappers
# ---------------------------------------------------------------------------

class SingleUnsqueeze(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class SingleSqueeze(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class SingleCat(nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat([x, x], dim=self.dim)


class SingleStack(nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.stack([x, x], dim=self.dim)


class SingleChunk(nn.Module):
    def __init__(self, n: int = 3, dim: int = 0):
        super().__init__()
        self.n = n
        self.dim = dim

    def forward(self, x):
        chunks = torch.chunk(x, self.n, dim=self.dim)
        out = chunks[0]
        for c in chunks[1:]:
            out = out + c
        return out


class SingleUnbind(nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        parts = torch.unbind(x, dim=self.dim)
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out


class SingleSelect(nn.Module):
    def __init__(self, dim: int = 0, index: int = 0):
        super().__init__()
        self.dim = dim
        self.index = index

    def forward(self, x):
        return x.select(self.dim, self.index)


class SingleSlice(nn.Module):
    def forward(self, x):
        return x[:, 1:3]


class SingleExpand(nn.Module):
    """Input must be shape (1, D); expands to (N, D)."""
    def __init__(self, n: int = 3):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x.expand(self.n, -1)


class SinglePad(nn.Module):
    def __init__(self, padding=(1, 1)):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding)


class SingleTranspose(nn.Module):
    def __init__(self, dim0: int = 0, dim1: int = 1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


# ---------------------------------------------------------------------------
# Reduction / pooling wrappers
# ---------------------------------------------------------------------------

class SingleMeanDim(nn.Module):
    def __init__(self, dim=-1, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=self.keepdim)


class SingleGlobalAvgPool(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, 1)


class SingleGroupNorm(nn.Module):
    def __init__(self, num_groups: int = 2, num_channels: int = 4):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        return self.gn(x)


# ---------------------------------------------------------------------------
# Constant generation wrappers
# ---------------------------------------------------------------------------

class AddWithZeros(nn.Module):
    """Uses aten.zeros.default to create a zero tensor and adds it to x."""
    def forward(self, x):
        z = torch.zeros(x.shape[-1])
        return x + z


class AddWithOnes(nn.Module):
    """Uses aten.ones.default to create an all-ones tensor and adds it to x."""
    def forward(self, x):
        o = torch.ones(x.shape[-1])
        return x + o


class AddWithFull(nn.Module):
    """Uses aten.full.default to create a constant tensor and adds it to x."""
    def forward(self, x):
        c = torch.full((x.shape[-1],), 0.5)
        return x + c


# ---------------------------------------------------------------------------
# Type cast wrappers
# ---------------------------------------------------------------------------

class CastToFloat16AndBack(nn.Module):
    """Cast to float16 then back to float32 (exercises _convert_cast)."""
    def forward(self, x):
        return x.to(torch.float16).to(torch.float32)

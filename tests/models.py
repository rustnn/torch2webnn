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

    def __init__(self, in_channels=3, out_channels=16, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

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

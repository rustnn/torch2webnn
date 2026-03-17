"""
Parametrized tests for unary element-wise operations.
Covers: exp, abs, sqrt, log, sigmoid, tanh, relu, rsqrt, reciprocal,
        pow.Scalar × multiple input shapes.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conftest import assert_export_matches, validate_webnn_execution


def _make_unary_model(op):
    """Create a minimal nn.Module wrapping a single unary op."""

    class UnaryModel(nn.Module):
        def forward(self, x):
            return op(x)

    UnaryModel.__name__ = getattr(op, '__name__', repr(op))
    return UnaryModel()


# (id, model_class, input_factory)
# input_factory takes shape tuple and returns a tensor safe for the op
UNARY_OPS = [
    ("exp",        torch.exp,                      lambda s: torch.randn(*s)),
    ("abs",        torch.abs,                      lambda s: torch.randn(*s)),
    ("sqrt",       torch.sqrt,                     lambda s: torch.randn(*s).abs() + 1e-3),
    ("log",        torch.log,                      lambda s: torch.randn(*s).abs() + 1e-3),
    ("sigmoid",    torch.sigmoid,                  lambda s: torch.randn(*s)),
    ("tanh",       torch.tanh,                     lambda s: torch.randn(*s)),
    ("relu",       F.relu,                         lambda s: torch.randn(*s)),
    ("rsqrt",      torch.rsqrt,                    lambda s: torch.rand(*s) + 0.1),
    ("reciprocal", torch.reciprocal,               lambda s: torch.rand(*s) + 0.1),
    ("pow_scalar", lambda x: torch.pow(2.0, x),   lambda s: torch.randn(*s) * 0.5),
]

SHAPES = [
    (1, 10),
    (1, 20),
    (2, 5, 8),
]


@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
@pytest.mark.parametrize(
    "op_name,torch_fn,input_fn",
    UNARY_OPS,
    ids=[op[0] for op in UNARY_OPS],
)
def test_unary_op(op_name, torch_fn, input_fn, shape):
    """Export and validate a single unary op with the given input shape."""
    torch._dynamo.reset()  # clear compile cache so each shape gets a fresh trace
    model = _make_unary_model(torch_fn)
    x = input_fn(shape)
    assert_export_matches(model, x, rtol=1e-5, atol=1e-5)
    validate_webnn_execution(model, x, rtol=1e-5, atol=1e-5)

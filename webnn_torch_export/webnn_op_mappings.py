"""Mapping helpers from PyTorch FX call_function targets to converter methods."""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch.fx as fx

if TYPE_CHECKING:
    from .webnn_generator import WebNNGraphGenerator


ConverterFn = Callable[["WebNNGraphGenerator", fx.Node, str, List[str]], str]


EXACT_TARGET_TO_CONVERTER: Dict[str, ConverterFn] = {
    "<built-in function iadd>": lambda gen, node, output, inputs: gen._convert_arithmetric(node, output, inputs, "add"),
    "<built-in function mul>": lambda gen, node, output, inputs: gen._convert_arithmetric(node, output, inputs, "mul"),
    "<built-in function truediv>": lambda gen, node, output, inputs: gen._convert_arithmetric(node, output, inputs, "div"),
    "<built-in function getitem>": lambda gen, node, output, inputs: gen._convert_getitem(node, output, inputs),
    "<built-in function neg>": lambda gen, node, output, inputs: gen._convert_neg(node, output, inputs),
}


TARGET_NAME_TO_CONVERTER: Dict[str, ConverterFn] = {
    # Convolution and Linear
    "conv2d": lambda gen, node, output, inputs: gen._convert_conv2d(node, output, inputs),
    "linear": lambda gen, node, output, inputs: gen._convert_linear(node, output, inputs),
    "addmm": lambda gen, node, output, inputs: gen._convert_addmm(node, output, inputs),
    "matmul": lambda gen, node, output, inputs: gen._convert_matmul(node, output, inputs),
    "mm": lambda gen, node, output, inputs: gen._convert_matmul(node, output, inputs),

    # Normalization
    "batch_norm": lambda gen, node, output, inputs: gen._convert_batch_norm(node, output, inputs),
    "layer_norm": lambda gen, node, output, inputs: gen._convert_layer_norm(node, output, inputs),
    "group_norm": lambda gen, node, output, inputs: gen._convert_group_norm(node, output, inputs),

    # Activations
    "relu": lambda gen, node, output, inputs: gen._convert_relu(node, output, inputs),
    "sigmoid": lambda gen, node, output, inputs: gen._convert_sigmoid(node, output, inputs),
    "tanh": lambda gen, node, output, inputs: gen._convert_tanh(node, output, inputs),
    "softmax": lambda gen, node, output, inputs: gen._convert_softmax(node, output, inputs),
    "hardtanh": lambda gen, node, output, inputs: gen._convert_hardtanh(node, output, inputs),
    "clamp": lambda gen, node, output, inputs: gen._convert_clamp(node, output, inputs),
    "silu": lambda gen, node, output, inputs: gen._convert_silu(node, output, inputs),

    # Arithmetic
    "add": lambda gen, node, output, inputs: gen._convert_arithmetric(node, output, inputs, "add"),
    "sub": lambda gen, node, output, inputs: gen._convert_arithmetric(node, output, inputs, "sub"),
    "mul": lambda gen, node, output, inputs: gen._convert_arithmetric(node, output, inputs, "mul"),
    "div": lambda gen, node, output, inputs: gen._convert_arithmetric(node, output, inputs, "div"),

    # Math functions
    "sqrt": lambda gen, node, output, inputs: gen._convert_math(node, output, inputs, "sqrt"),
    "exp": lambda gen, node, output, inputs: gen._convert_math(node, output, inputs, "exp"),
    "abs": lambda gen, node, output, inputs: gen._convert_math(node, output, inputs, "abs"),
    "log": lambda gen, node, output, inputs: gen._convert_math(node, output, inputs, "log"),
    "cos": lambda gen, node, output, inputs: gen._convert_math(node, output, inputs, "cos"),
    "sin": lambda gen, node, output, inputs: gen._convert_math(node, output, inputs, "sin"),
    "pow": lambda gen, node, output, inputs: gen._convert_pow(node, output, inputs),

    # Pooling
    "adaptive_avg_pool": lambda gen, node, output, inputs: gen._convert_global_avg_pool(node, output, inputs),
    "avg_pool2d": lambda gen, node, output, inputs: gen._convert_avg_pool2d(node, output, inputs),
    "max_pool2d": lambda gen, node, output, inputs: gen._convert_max_pool2d(node, output, inputs),
    "mean": lambda gen, node, output, inputs: gen._convert_reduce_mean(node, output, inputs),

    # Tensor manipulation
    "flatten": lambda gen, node, output, inputs: gen._convert_reshape(node, output, inputs),
    "view": lambda gen, node, output, inputs: gen._convert_reshape(node, output, inputs),
    "reshape": lambda gen, node, output, inputs: gen._convert_reshape(node, output, inputs),
    "transpose": lambda gen, node, output, inputs: gen._convert_transpose(node, output, inputs),
    "t": lambda gen, node, output, inputs: gen._convert_transpose(node, output, inputs),
    "permute": lambda gen, node, output, inputs: gen._convert_transpose(node, output, inputs),
    "concat": lambda gen, node, output, inputs: gen._convert_concat(node, output, inputs),
    "cat": lambda gen, node, output, inputs: gen._convert_concat(node, output, inputs),
    "stack": lambda gen, node, output, inputs: gen._convert_stack(node, output, inputs),
    "split": lambda gen, node, output, inputs: gen._convert_split(node, output, inputs),
    "slice": lambda gen, node, output, inputs: gen._convert_slice(node, output, inputs),
    "expand": lambda gen, node, output, inputs: gen._convert_expand(node, output, inputs),
    "pad": lambda gen, node, output, inputs: gen._convert_pad(node, output, inputs),
    "tile": lambda gen, node, output, inputs: gen._convert_tile(node, output, inputs),
    "to": lambda gen, node, output, inputs: gen._convert_cast(node, output, inputs),
    "float": lambda gen, node, output, inputs: gen._convert_cast(node, output, inputs, "f32"),
    "half": lambda gen, node, output, inputs: gen._convert_cast(node, output, inputs, "f16"),

    # Special operations
    "rearrange": lambda gen, node, output, inputs: gen._convert_rearrange(node, output, inputs),
    "arange": lambda gen, node, output, inputs: gen._convert_arange(node, output, inputs),
    # "dropout": lambda gen, node, output, inputs: gen._convert_identity(node, output, inputs),
    "einsum": lambda gen, node, output, inputs: gen._convert_einsum(node, output, inputs),
    "scaled_dot_product_attention": lambda gen, node, output, inputs: gen._convert_scaled_dot_product_attention(node, output, inputs),
    "interpolate": lambda gen, node, output, inputs: gen._convert_interpolate(node, output, inputs),

    # No OP
    "identity": lambda gen, node, output, inputs: gen._convert_identity(node, output, inputs),
    "contiguous": lambda gen, node, output, inputs: gen._convert_identity(node, output, inputs),
}

SCHEMA_CONTAINS_TO_CONVERTER: Dict[str, ConverterFn] = {
    "convolution": lambda gen, node, output, inputs: gen._convert_conv2d(node, output, inputs),
    "relu": lambda gen, node, output, inputs: gen._convert_relu(node, output, inputs),
    "add": lambda gen, node, output, inputs: gen._convert_arithmetric(node, output, inputs, "add"),
}


def resolve_pytorch_converter(target) -> Optional[ConverterFn]:
    """Resolve converter callable for a PyTorch FX call_function target."""
    target_str = str(target)

    if target_str in EXACT_TARGET_TO_CONVERTER:
        return EXACT_TARGET_TO_CONVERTER[target_str]

    if target_str in TARGET_NAME_TO_CONVERTER:
        return TARGET_NAME_TO_CONVERTER[target_str]

    target_name = getattr(target, "__name__", None)
    if target_name in TARGET_NAME_TO_CONVERTER:
        return TARGET_NAME_TO_CONVERTER[target_name]

    schema = getattr(target, "_schema", None)
    if schema is not None:
        schema_str = str(schema)
        if schema_str in SCHEMA_CONTAINS_TO_CONVERTER.items():
            return SCHEMA_CONTAINS_TO_CONVERTER[schema_str]

    return None

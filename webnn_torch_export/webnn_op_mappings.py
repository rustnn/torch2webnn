"""ATen op → WebNN converter dispatch table.

All targets are matched by str(node.target) which for ATen ops gives
the canonical name like "aten.conv2d.default".
"""

from typing import Dict, Optional

# Maps str(aten_op) → converter method name on WebNNGraphGenerator.
ATEN_OP_TABLE: Dict[str, str] = {
    # Convolution
    "aten.conv2d.default": "_convert_conv2d",
    "aten.convolution.default": "_convert_convolution",

    # Linear / matmul
    "aten.linear.default": "_convert_linear",
    "aten.addmm.default": "_convert_addmm",
    "aten.mm.default": "_convert_matmul",
    "aten.matmul.default": "_convert_matmul",
    "aten.t.default": "_convert_t",

    # Activations
    "aten.relu.default": "_convert_relu",
    "aten.relu_.default": "_convert_relu",
    "aten.sigmoid.default": "_convert_sigmoid",
    "aten.tanh.default": "_convert_tanh",
    "aten.silu.default": "_convert_silu",
    "aten.silu_.default": "_convert_silu",
    "aten.hardtanh.default": "_convert_hardtanh",
    "aten.hardtanh_.default": "_convert_hardtanh",
    "aten.clamp.default": "_convert_clamp",
    "aten.clamp_.default": "_convert_clamp",
    "aten.gelu.default": "_convert_gelu",

    # Normalization
    "aten.batch_norm.default": "_convert_batch_norm_aten",
    "aten._native_batch_norm_legit_no_training.default": "_convert_batch_norm_no_training",
    "aten.layer_norm.default": "_convert_layer_norm",
    "aten.group_norm.default": "_convert_group_norm",

    # Pooling
    "aten.max_pool2d.default": "_convert_max_pool2d",
    "aten.max_pool2d_with_indices.default": "_convert_max_pool2d",
    "aten.avg_pool2d.default": "_convert_avg_pool2d",
    "aten.adaptive_avg_pool2d.default": "_convert_global_avg_pool",
    "aten.mean.dim": "_convert_reduce_mean",
    "aten.mean.default": "_convert_reduce_mean",

    # Arithmetic
    "aten.add.Tensor": "_convert_add",
    "aten.add.Scalar": "_convert_add",
    "aten.add_.Tensor": "_convert_add",
    "aten.sub.Tensor": "_convert_sub",
    "aten.sub.Scalar": "_convert_sub",
    "aten.mul.Tensor": "_convert_mul",
    "aten.mul.Scalar": "_convert_mul",
    "aten.div.Tensor": "_convert_div",
    "aten.div.Scalar": "_convert_div",
    "aten.neg.default": "_convert_neg",
    "aten.pow.Tensor_Scalar": "_convert_pow",
    "aten.pow.Tensor_Tensor": "_convert_pow",
    "aten.pow.Scalar": "_convert_pow_scalar",

    # Elementwise math
    "aten.sqrt.default": "_convert_math_sqrt",
    "aten.exp.default": "_convert_math_exp",
    "aten.abs.default": "_convert_math_abs",
    "aten.log.default": "_convert_math_log",
    "aten.cos.default": "_convert_math_cos",
    "aten.sin.default": "_convert_math_sin",
    "aten.rsqrt.default": "_convert_rsqrt",
    "aten.reciprocal.default": "_convert_reciprocal",

    # Tensor shape manipulation
    "aten.reshape.default": "_convert_reshape",
    "aten.view.default": "_convert_reshape",
    "aten._unsafe_view.default": "_convert_reshape",
    "aten.flatten.using_ints": "_convert_reshape",
    "aten.permute.default": "_convert_permute",
    "aten.transpose.int": "_convert_transpose",
    "aten.unsqueeze.default": "_convert_unsqueeze",
    "aten.squeeze.dim": "_convert_squeeze",
    "aten.squeeze.default": "_convert_squeeze",
    "aten.cat.default": "_convert_concat",
    "aten.stack.default": "_convert_stack",
    "aten.split.Tensor": "_convert_split",
    "aten.split_with_sizes.default": "_convert_split",
    "aten.chunk.default": "_convert_chunk",
    "aten.unbind.int": "_convert_unbind",
    "aten.select.int": "_convert_select",
    "aten.einsum.default": "_convert_einsum",
    "aten.slice.Tensor": "_convert_slice",
    "aten.expand.default": "_convert_expand",
    "aten.expand_as.default": "_convert_expand_as",

    # Padding
    "aten.constant_pad_nd.default": "_convert_pad",
    "aten.pad.default": "_convert_pad",

    # Upsampling
    "aten.upsample_nearest2d.vec":     "_convert_upsample_nearest2d",
    "aten.upsample_nearest2d.default": "_convert_upsample_nearest2d",
    "aten.upsample_bilinear2d.vec":    "_convert_upsample_bilinear2d",
    "aten.upsample_bilinear2d.default":"_convert_upsample_bilinear2d",

    # Softmax / attention
    "aten.softmax.int": "_convert_softmax",
    "aten._softmax.default": "_convert_softmax_aten",
    "aten.scaled_dot_product_attention.default": "_convert_scaled_dot_product_attention",

    # Compile-time constant generation
    "aten.arange.default":      "_convert_arange",
    "aten.arange.start":        "_convert_arange",
    "aten.arange.start_step":   "_convert_arange",
    "aten.full.default":        "_convert_full",
    "aten.zeros.default":       "_convert_full_zeros",
    "aten.ones.default":        "_convert_full_ones",

    # No-ops / memory ops
    "aten.clone.default": "_convert_identity",
    "aten.contiguous.default": "_convert_identity",
    "aten._assert_tensor_metadata.default": "_convert_identity",
    "aten.contiguous.memory_format": "_convert_identity",
    "aten.dropout.default": "_convert_identity",
    "aten.alias.default": "_convert_identity",

    # Type cast
    "aten._to_copy.default":  "_convert_cast",
    "aten.to.dtype":          "_convert_cast",
    "aten.to.device":         "_convert_to_device",
    "aten.to.dtype_layout":   "_convert_cast",
    "aten.type_as.default":   "_convert_type_as",

    # Tuple/list indexing (from chunk, unbind, split)
    "<built-in function getitem>": "_convert_getitem",
}


def resolve_aten_converter(target) -> Optional[str]:
    """Return converter method name for an ATen (or other) op target, or None."""
    return ATEN_OP_TABLE.get(str(target))


if __name__ == "__main__":
    from torch._decomp import _core_aten_decompositions_post_autograd

    aten_ops = _core_aten_decompositions_post_autograd()
    aten_ops_as_str = set(".".join((op.name().split("::"))) for op in aten_ops.keys())
    supported_aten_str = set(ATEN_OP_TABLE.keys())

    missing_aten = aten_ops_as_str.difference(supported_aten_str)
    # filter any op that contains "backward"
    missing_aten_non_bwd = [op for op in missing_aten if "backward" not in op]

    print(missing_aten_non_bwd)
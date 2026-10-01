"""ATen op → WebNN converter dispatch table.

All targets are matched by str(node.target) which for ATen ops gives
the canonical name like "aten.conv2d.default".
"""

from typing import Dict, Optional

# Maps str(aten_op) → converter method name on WebNNGraphGenerator.
ATEN_OP_TABLE: Dict[str, str] = {
    "aten.conv2d.default": "_conv",
    "aten.linear.default": "_linear",
    "aten.addmm.default": "_linear",
    "aten.mm.default": "_linear",
    "aten.matmul.default": "_linear",
    "aten.t.default": "_shape",
    "aten.relu.default": "_unary",
    "aten.relu_.default": "_unary",
    "aten.sigmoid.default": "_unary",
    "aten.tanh.default": "_unary",
    "aten.silu.default": "_activation",
    "aten.silu_.default": "_activation",
    "aten.hardtanh.default": "_activation",
    "aten.hardtanh_.default": "_activation",
    "aten.clamp.default": "_activation",
    "aten.clamp_.default": "_activation",
    "aten.gelu.default": "_activation",
    "aten.batch_norm.default": "_normalization",
    "aten._native_batch_norm_legit_no_training.default": "_normalization",
    "aten.layer_norm.default": "_normalization",
    "aten.group_norm.default": "_normalization",
    "aten.max_pool2d.default": "_pool",
    "aten.max_pool2d_with_indices.default": "_pool",
    "aten.avg_pool2d.default": "_pool",
    "aten.adaptive_avg_pool2d.default": "_pool",
    "aten.mean.dim": "_pool",
    "aten.mean.default": "_pool",
    "aten.add.Tensor": "_arithmetic",
    "aten.add.Scalar": "_arithmetic",
    "aten.add_.Tensor": "_arithmetic",
    "aten.sub.Tensor": "_arithmetic",
    "aten.sub.Scalar": "_arithmetic",
    "aten.mul.Tensor": "_arithmetic",
    "aten.mul.Scalar": "_arithmetic",
    "aten.div.Tensor": "_arithmetic",
    "aten.div.Scalar": "_arithmetic",
    "aten.neg.default": "_unary",
    "aten.pow.Tensor_Scalar": "_arithmetic",
    "aten.pow.Tensor_Tensor": "_arithmetic",
    "aten.pow.Scalar": "_arithmetic",
    "aten.sqrt.default": "_unary",
    "aten.exp.default": "_unary",
    "aten.abs.default": "_unary",
    "aten.log.default": "_unary",
    "aten.cos.default": "_unary",
    "aten.sin.default": "_unary",
    "aten.rsqrt.default": "_activation",
    "aten.reciprocal.default": "_unary",
    "aten.reshape.default": "_shape",
    "aten.view.default": "_shape",
    "aten._unsafe_view.default": "_shape",
    "aten.flatten.using_ints": "_shape",
    "aten.permute.default": "_shape",
    "aten.transpose.int": "_shape",
    "aten.unsqueeze.default": "_shape",
    "aten.squeeze.dim": "_shape",
    "aten.squeeze.default": "_shape",
    "aten.cat.default": "_shape",
    "aten.stack.default": "_shape",
    "aten.split.Tensor": "_shape",
    "aten.split_with_sizes.default": "_shape",
    "aten.chunk.default": "_shape",
    "aten.unbind.int": "_shape",
    "aten.select.int": "_shape",
    "aten.einsum.default": "_shape",
    "aten.slice.Tensor": "_shape",
    "aten.expand.default": "_shape",
    "aten.expand_as.default": "_shape",
    "aten.constant_pad_nd.default": "_pad",
    "aten.pad.default": "_pad",
    "aten.upsample_nearest2d.vec": "_upsample",
    "aten.upsample_nearest2d.default": "_upsample",
    "aten.upsample_bilinear2d.vec": "_upsample",
    "aten.upsample_bilinear2d.default": "_upsample",
    "aten.softmax.int": "_attention",
    "aten._softmax.default": "_attention",
    "aten.scaled_dot_product_attention.default": "_attention",
    "aten.arange.default": "_constant_op",
    "aten.arange.start": "_constant_op",
    "aten.arange.start_step": "_constant_op",
    "aten.full.default": "_constant_op",
    "aten.zeros.default": "_constant_op",
    "aten.ones.default": "_constant_op",
    "aten.clone.default": "_identity",
    "aten.detach.default": "_identity",
    "aten.detach_.default": "_identity",
    "aten.lift_fresh_copy.default": "_identity",
    "aten.contiguous.default": "_identity",
    "aten._assert_tensor_metadata.default": "_identity",
    "aten.contiguous.memory_format": "_identity",
    "aten.dropout.default": "_identity",
    "aten.alias.default": "_identity",
    "aten._to_copy.default": "_type",
    "aten.to.dtype": "_type",
    "aten.to.device": "_type",
    "aten.to.dtype_layout": "_type",
    "aten.type_as.default": "_type",
    "<built-in function getitem>": "_getitem",
    "aten.bmm.default": "_linear",
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

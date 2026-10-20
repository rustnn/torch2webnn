"""Build fixed-size pywebnn networks from torch.export ATen graphs.

Every builder result is checked against a PyTorch meta tensor immediately, including
intermediate operands introduced by decompositions. FX results are also checked
against torch.export metadata before subsequent nodes can use them.
"""

from dataclasses import dataclass
import math
import operator

import torch
import torch.fx as fx
import torch.nn.functional as F
import webnn
from torch.utils import _pytree

from .webnn_op_mappings import resolve_aten_converter


class ShapeMismatchError(ValueError):
    """PyTorch and pywebnn inferred different shapes for an operand."""


@dataclass
class CheckedOperand:
    operand: object
    tensor: torch.Tensor


DTYPES = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.uint32: "uint32",
    torch.uint64: "uint64",
}


def fixed_shape(tensor):
    shape = tuple(tensor.shape)
    if any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError(
            f"Only fixed, positive tensor dimensions are supported; got {shape}"
        )
    return shape


def webnn_dtype(dtype):
    try:
        return DTYPES[dtype]
    except KeyError:
        raise NotImplementedError(f"Unsupported WebNN dtype: {dtype}") from None


class WebNNGraphGenerator:
    """Construct an in-memory network; no graph text or parser is involved."""

    def __init__(self, context=None):
        self.context = context
        self.builder = None
        self.node = None
        self.checked_shapes = []
        self.input_names = []
        self.input_specs = {}
        self.output_names = []
        self.output_specs = {}
        self.node_to_operand = {}

    def _check(self, operand, tensor, operation):
        expected = fixed_shape(tensor)
        actual = tuple(operand.shape)
        label = self.node.name if self.node is not None else "graph"
        if actual != expected:
            raise ShapeMismatchError(
                f"{label} ({operation}): PyTorch shape {expected} != pywebnn shape {actual}"
            )
        self.checked_shapes.append((label, operation, expected))
        return CheckedOperand(
            operand, torch.empty(expected, dtype=tensor.dtype, device="meta")
        )

    def _emit(self, operation, reference, *args, **options):
        def unwrap(value, attr):
            return _pytree.tree_map(
                lambda x: getattr(x, attr) if isinstance(x, CheckedOperand) else x,
                value,
            )

        # PyTorch independently infers the expected intermediate shape.
        expected = reference(*unwrap(args, "tensor"))
        operand = getattr(self.builder, operation)(
            *unwrap(args, "operand"), **unwrap(options, "operand")
        )
        return self._check(operand, expected, operation)

    def _constant(self, tensor):
        tensor = tensor.detach().cpu().contiguous()
        fixed_shape(tensor)
        dtype = webnn_dtype(tensor.dtype)
        operand = self.builder.constant(tensor.numpy(), data_type=dtype)
        return self._check(operand, tensor, "constant")

    def _scalar(self, value, like):
        return self._constant(
            torch.tensor(value, dtype=torch.result_type(like.tensor, value))
        )

    def _binary(self, name, a, b):
        if not isinstance(a, CheckedOperand):
            a = self._scalar(a, b)
        if not isinstance(b, CheckedOperand):
            b = self._scalar(b, a)
        if a.tensor.dtype != b.tensor.dtype:
            dtype = torch.promote_types(a.tensor.dtype, b.tensor.dtype)
            a, b = self._cast(a, dtype), self._cast(b, dtype)
        return self._emit(name, getattr(torch, name), a, b)

    def _cast(self, x, dtype):
        if x.tensor.dtype == dtype:
            return x
        return self._emit(
            "cast", lambda t: t.to(dtype), x, data_type=webnn_dtype(dtype)
        )

    def _reshape(self, x, shape):
        # Resolve -1 through PyTorch, then pass only fixed dimensions to WebNN.
        resolved = list(x.tensor.reshape(shape).shape)
        return self._emit("reshape", lambda t: t.reshape(shape), x, new_shape=resolved)

    def _transpose(self, x, permutation):
        return self._emit(
            "transpose",
            lambda t: t.permute(permutation),
            x,
            permutation=list(permutation),
        )

    def _matmul(self, a, b):
        expected = torch.matmul(a.tensor, b.tensor).shape
        a_vec, b_vec = a.tensor.ndim == 1, b.tensor.ndim == 1
        if a_vec:
            a = self._reshape(a, [1, a.tensor.shape[0]])
        if b_vec:
            b = self._reshape(b, [b.tensor.shape[0], 1])
        result = self._emit("matmul", torch.matmul, a, b)
        return self._reshape(result, expected) if a_vec or b_vec else result

    def _mean(self, x, axes, keep=True):
        axes = list(range(x.tensor.ndim)) if axes is None else list(axes)
        axes = [axis % x.tensor.ndim for axis in axes]
        return self._emit(
            "reduce_mean",
            lambda t: t.mean(tuple(axes), keepdim=keep),
            x,
            axes=axes,
            keep_dimensions=keep,
        )

    def _normalize(self, x, axes, eps):
        centered = self._binary("sub", x, self._mean(x, axes))
        variance = self._mean(self._binary("mul", centered, centered), axes)
        std = self._emit("sqrt", torch.sqrt, self._binary("add", variance, eps))
        return self._binary("div", centered, std)

    def _affine(self, x, weight, bias, shape=None):
        if weight is not None:
            x = self._binary(
                "mul", x, self._reshape(weight, shape) if shape else weight
            )
        if bias is not None:
            x = self._binary("add", x, self._reshape(bias, shape) if shape else bias)
        return x

    def generate(self, ep, graph_name="model"):
        """Return an MLGraph with embedded constants and checked static operands."""
        if ep.range_constraints:
            raise ValueError(
                "Dynamic shapes are unsupported; export with fixed example inputs"
            )
        self.builder = (
            self.context.create_graph_builder()
            if self.context is not None
            else webnn.MLGraphBuilder.new_uncompiled()
        )
        self.checked_shapes = []
        self.input_names, self.output_names = [], []
        self.input_specs, self.node_to_operand, self.output_specs = {}, {}, {}
        sig = ep.graph_signature
        if sig.buffers_to_mutate or sig.user_inputs_to_mutate:
            raise NotImplementedError(
                "Graphs with input or buffer mutations are unsupported"
            )
        constants = {**ep.state_dict, **ep.constants}
        from torch.export.graph_signature import ConstantArgument, InputKind

        user_specs = [
            spec for spec in sig.input_specs if spec.kind == InputKind.USER_INPUT
        ]
        user_names = {spec.arg.name for spec in user_specs}
        self.scalar_inputs = {
            i: spec.arg.value
            for i, spec in enumerate(user_specs)
            if isinstance(spec.arg, ConstantArgument)
        }
        targets = {
            **sig.inputs_to_parameters,
            **sig.inputs_to_buffers,
            **sig.inputs_to_lifted_tensor_constants,
        }
        for node in ep.graph.nodes:
            self.node = node
            if node.op == "placeholder":
                if node.name in targets:
                    result = self._constant(constants[targets[node.name]])
                elif node.name in user_names:
                    value = node.meta.get("val")
                    if not isinstance(value, torch.Tensor):
                        # Non-tensor arguments are specialized by torch.export.
                        result = value
                    else:
                        shape = fixed_shape(value)
                        name = node.name
                        operand = self.builder.input(
                            name, list(shape), webnn_dtype(value.dtype)
                        )
                        result = self._check(operand, value, "input")
                        self.input_names.append(name)
                        self.input_specs[name] = (shape, value.dtype)
                else:
                    raise NotImplementedError(f"Unsupported placeholder {node.name}")
            elif node.op == "call_function":
                result = self._convert(node)
                self._check_result(result, node.meta.get("val"))
            elif node.op == "get_attr":
                result = self._constant(getattr(ep.graph_module, node.target))
            elif node.op == "output":
                leaves, _ = _pytree.tree_flatten(self._resolve(node.args[0]))
                outputs = {}
                for i, leaf in enumerate(leaves):
                    if not isinstance(leaf, CheckedOperand):
                        raise NotImplementedError("Only tensor outputs are supported")
                    name = f"output_{i}"
                    # Distinct output operands preserve duplicate outputs and aliases.
                    result = self._emit("identity", lambda t: t, leaf)
                    outputs[name] = result.operand
                    self.output_names.append(name)
                    self.output_specs[name] = (
                        tuple(result.tensor.shape),
                        result.tensor.dtype,
                    )
                self.node = None
                self.outputs = outputs
                return self.builder.build(outputs)
            else:
                raise NotImplementedError(
                    f"Unsupported FX node: {node.op} ({node.name})"
                )
            self.node_to_operand[node.name] = result
        raise ValueError("Exported graph has no outputs")

    def _resolve(self, value):
        return _pytree.tree_map(
            lambda x: self.node_to_operand[x.name] if isinstance(x, fx.Node) else x,
            value,
        )

    def _check_result(self, result, expected):
        if isinstance(result, CheckedOperand):
            if not isinstance(expected, torch.Tensor):
                raise ValueError(
                    f"Missing PyTorch tensor metadata for {self.node.name}"
                )
            self._check(result.operand, expected, str(self.node.target))
        elif isinstance(result, (tuple, list)):
            if not isinstance(expected, (tuple, list)) or len(result) != len(expected):
                raise ValueError(f"Output structure mismatch at {self.node.name}")
            for item, value in zip(result, expected):
                if item is not None:
                    self._check_result(item, value)

    def _convert(self, node):
        converter = resolve_aten_converter(node.target)
        if converter is None:
            raise NotImplementedError(
                f"Unsupported ATen operation at {node.name}: {node.target}"
            )
        args = self._resolve(node.args)
        kwargs = self._resolve(node.kwargs)
        return getattr(self, converter)(node, args, kwargs)

    def _unary(self, node, args, kwargs):
        name = str(node.target).split(".")[1].rstrip("_")
        return self._emit(name, getattr(torch, name), args[0])

    def _arithmetic(self, node, args, kwargs):
        name = str(node.target).split(".")[1].rstrip("_")
        a, b = args[:2]
        alpha = kwargs.get("alpha", 1)
        if name in ("add", "sub") and alpha != 1:
            b = (
                self._binary("mul", b, alpha)
                if isinstance(b, CheckedOperand)
                else b * alpha
            )
        if name == "div" and kwargs.get("rounding_mode") is not None:
            raise NotImplementedError("Integer/rounded division is unsupported")
        return self._binary(name, a, b)

    def _activation(self, node, args, kwargs):
        name = str(node.target).split(".")[1].rstrip("_")
        x = args[0]
        if name == "silu":
            return self._binary("mul", x, self._emit("sigmoid", torch.sigmoid, x))
        if name == "rsqrt":
            return self._emit(
                "reciprocal", torch.reciprocal, self._emit("sqrt", torch.sqrt, x)
            )
        if name == "gelu":
            approximate = (
                args[1] if len(args) > 1 else kwargs.get("approximate", "none")
            )
            if approximate == "none":
                return self._emit("gelu", F.gelu, x)
            if approximate != "tanh":
                raise NotImplementedError(
                    f"Unsupported GELU approximation: {approximate}"
                )
            cube = self._binary("pow", x, 3)
            inner = self._binary(
                "mul",
                self._binary("add", x, self._binary("mul", cube, 0.044715)),
                math.sqrt(2 / math.pi),
            )
            return self._binary(
                "mul",
                self._binary("mul", x, 0.5),
                self._binary("add", self._emit("tanh", torch.tanh, inner), 1),
            )
        low = args[1] if len(args) > 1 else kwargs.get("min")
        high = args[2] if len(args) > 2 else kwargs.get("max")
        if name == "hardtanh":
            low, high = (
                (args[1] if len(args) > 1 else -1),
                (args[2] if len(args) > 2 else 1),
            )
        return self._emit(
            "clamp",
            lambda t: torch.clamp(t, min=low, max=high),
            x,
            min_value=low,
            max_value=high,
        )

    def _linear(self, node, args, kwargs):
        name = str(node.target).split(".")[1]
        if name == "linear":
            x, weight = args[:2]
            out = self._matmul(x, self._transpose(weight, [1, 0]))
            return (
                self._binary("add", out, args[2])
                if len(args) > 2 and args[2] is not None
                else out
            )
        if name == "addmm":
            bias, a, b = args[:3]
            out = self._matmul(a, b)
            alpha, beta = kwargs.get("alpha", 1), kwargs.get("beta", 1)
            if alpha != 1:
                out = self._binary("mul", out, alpha)
            if beta == 0:
                return out
            return self._binary(
                "add", out, self._binary("mul", bias, beta) if beta != 1 else bias
            )
        return self._matmul(*args[:2])

    def _conv(self, node, args, kwargs):
        x, weight = args[:2]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else [1, 1]
        padding = args[4] if len(args) > 4 else [0, 0]
        dilation = args[5] if len(args) > 5 else [1, 1]
        groups = args[6] if len(args) > 6 else 1
        if isinstance(padding, str):
            raise NotImplementedError("String convolution padding is unsupported")

        def pair(v):
            return list(v) if isinstance(v, (tuple, list)) else [v, v]

        stride, padding, dilation = pair(stride), pair(padding), pair(dilation)
        return self._emit(
            "conv2d",
            lambda t, w: F.conv2d(
                t,
                w,
                bias.tensor if bias is not None else None,
                stride,
                padding,
                dilation,
                groups,
            ),
            x,
            weight,
            bias=bias,
            strides=stride,
            pads=[padding[0]] * 2 + [padding[1]] * 2,
            dilations=dilation,
            groups=groups,
            input_layout="nchw",
            filter_layout="oihw",
        )

    def _shape(self, node, args, kwargs):
        name = str(node.target).split(".")[1]
        x = args[0]
        if name in (
            "reshape",
            "view",
            "_unsafe_view",
            "flatten",
            "unsqueeze",
            "squeeze",
        ):
            # ATen meta execution handles inferred dimensions and no-op squeeze.
            expected = node.target(x.tensor, *args[1:], **kwargs)
            return self._reshape(x, expected.shape)
        if name == "t":
            return self._transpose(x, list(reversed(range(x.tensor.ndim))))
        if name == "permute":
            return self._transpose(x, args[1])
        if name == "transpose":
            perm = list(range(x.tensor.ndim))
            a, b = args[1] % len(perm), args[2] % len(perm)
            perm[a], perm[b] = perm[b], perm[a]
            return self._transpose(x, perm)
        if name in ("cat", "stack"):
            tensors = x
            dim = (args[1] if len(args) > 1 else 0) % (
                tensors[0].tensor.ndim + (name == "stack")
            )
            if name == "stack":
                tensors = [
                    self._reshape(
                        t, list(t.tensor.shape[:dim]) + [1] + list(t.tensor.shape[dim:])
                    )
                    for t in tensors
                ]
            return self._emit(
                "concat", lambda ts: torch.cat(ts, dim=dim), tensors, axis=dim
            )
        if name in ("expand", "expand_as"):
            shape = args[1].tensor.shape if name == "expand_as" else args[1]
            resolved = list(x.tensor.expand(shape).shape)
            return self._emit(
                "expand", lambda t: t.expand(shape), x, new_shape=resolved
            )
        if name in ("split", "split_with_sizes", "chunk", "unbind"):
            results = node.target(x.tensor, *args[1:], **kwargs)
            dim = (
                args[1]
                if name == "unbind" and len(args) > 1
                else args[2]
                if name != "unbind" and len(args) > 2
                else 0
            ) % x.tensor.ndim
            offset, outputs = 0, []
            for value in results:
                size = 1 if name == "unbind" else value.shape[dim]
                part = self._slice(x, dim, offset, offset + size)
                outputs.append(
                    self._reshape(part, value.shape) if name == "unbind" else part
                )
                offset += size
            return tuple(outputs)
        if name == "select":
            dim, index = args[1] % x.tensor.ndim, args[2]
            index %= x.tensor.shape[dim]
            out = self._slice(x, dim, index, index + 1)
            return self._reshape(out, x.tensor.select(dim, index).shape)
        if name == "slice":
            dim = args[1] if len(args) > 1 else 0
            start = args[2] if len(args) > 2 else 0
            end = args[3] if len(args) > 3 else x.tensor.shape[dim]
            step = args[4] if len(args) > 4 else 1
            return self._slice(x, dim, start, end, step)
        if name == "einsum":
            equation, operands = args[:2]
            if equation != "...n,d->...nd":
                raise NotImplementedError(f"Unsupported einsum equation: {equation}")
            a, b = operands
            a = self._reshape(a, [*a.tensor.shape, 1])
            return self._binary("mul", a, b)
        raise NotImplementedError(f"Unsupported shape operation: {node.target}")

    def _slice(self, x, dim, start, end, step=1):
        dim %= x.tensor.ndim
        if step <= 0:
            raise NotImplementedError("Slice strides must be positive")
        start, end, step = slice(start, end, step).indices(x.tensor.shape[dim])
        slices = [slice(None)] * x.tensor.ndim
        slices[dim] = slice(start, end, step)
        starts, sizes, strides = (
            [0] * x.tensor.ndim,
            list(x.tensor.shape),
            [1] * x.tensor.ndim,
        )
        starts[dim], sizes[dim], strides[dim] = start, end - start, step
        return self._emit(
            "slice",
            lambda t: t[tuple(slices)],
            x,
            starts=starts,
            sizes=sizes,
            strides=strides,
        )

    def _normalization(self, node, args, kwargs):
        name = str(node.target).split(".")[1]
        x = args[0]
        if name == "layer_norm":
            normalized = args[1]
            weight = args[2] if len(args) > 2 else None
            bias = args[3] if len(args) > 3 else None
            eps = args[4] if len(args) > 4 else 1e-5
            axes = list(range(x.tensor.ndim - len(normalized), x.tensor.ndim))
            return self._affine(self._normalize(x, axes, eps), weight, bias)
        if name == "group_norm":
            groups = args[1]
            weight = args[2] if len(args) > 2 else None
            bias = args[3] if len(args) > 3 else None
            eps = args[4] if len(args) > 4 else 1e-5
            shape = list(x.tensor.shape)
            grouped = self._reshape(x, [shape[0], groups, -1])
            out = self._reshape(self._normalize(grouped, [2], eps), shape)
            return self._affine(
                out, weight, bias, [1, shape[1]] + [1] * (len(shape) - 2)
            )
        weight, bias, mean, variance = args[1:5]
        native = name == "_native_batch_norm_legit_no_training"
        training = False if native else args[5]
        eps = args[6] if native else args[7]
        affine_shape = [1, x.tensor.shape[1]] + [1] * (x.tensor.ndim - 2)
        if not training and mean is not None and variance is not None:
            mean, variance = (
                self._reshape(mean, affine_shape),
                self._reshape(variance, affine_shape),
            )
            std = self._emit("sqrt", torch.sqrt, self._binary("add", variance, eps))
            out = self._binary("div", self._binary("sub", x, mean), std)
        else:
            out = self._normalize(x, [0] + list(range(2, x.tensor.ndim)), eps)
        out = self._affine(out, weight, bias, affine_shape)
        if native:
            # Saved statistics in inference mode are empty tensors; WebNN does not
            # represent zero-sized operands. Reject attempts to consume these.
            return (out, None, None)
        return out

    def _pool(self, node, args, kwargs):
        name = str(node.target).split(".")[1]
        x = args[0]
        if name == "mean":
            axes = args[1] if len(args) > 1 else None
            keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
            dtype = kwargs.get("dtype", x.tensor.dtype)
            return self._mean(self._cast(x, dtype), axes, keep)
        if name == "adaptive_avg_pool2d":
            size = args[1]
            if list(size) != [1, 1]:
                raise NotImplementedError(
                    "Adaptive average pooling currently supports output size (1, 1)"
                )
            return self._mean(x, [2, 3])
        kernel = args[1]

        def pair(v):
            return list(v) if isinstance(v, (tuple, list)) else [v, v]

        kernel = pair(kernel)
        stride = pair(args[2]) if len(args) > 2 and args[2] else kernel
        padding = pair(args[3]) if len(args) > 3 else [0, 0]
        options = dict(
            window_dimensions=kernel,
            strides=stride,
            pads=[padding[0]] * 2 + [padding[1]] * 2,
            layout="nchw",
        )
        if name.startswith("max_pool"):
            dilation = pair(args[4]) if len(args) > 4 else [1, 1]
            ceil = args[5] if len(args) > 5 else False
            out = self._emit(
                "max_pool2d",
                lambda t: F.max_pool2d(t, kernel, stride, padding, dilation, ceil),
                x,
                dilations=dilation,
                output_shape_rounding="ceil" if ceil else "floor",
                **options,
            )
            if name == "max_pool2d_with_indices":
                return (out, None)
            return out
        ceil = args[4] if len(args) > 4 else False
        include_pad = args[5] if len(args) > 5 else True
        divisor = args[6] if len(args) > 6 else None
        if divisor is not None or (include_pad and any(padding)):
            raise NotImplementedError(
                "Average pooling with a custom divisor or included padding is unsupported"
            )
        return self._emit(
            "average_pool2d",
            lambda t: F.avg_pool2d(t, kernel, stride, padding, ceil, include_pad),
            x,
            output_shape_rounding="ceil" if ceil else "floor",
            **options,
        )

    def _pad(self, node, args, kwargs):
        x, padding = args[:2]
        mode = (
            args[2]
            if len(args) > 2 and str(node.target).split(".")[1] == "pad"
            else "constant"
        )
        value = (
            (args[3] if len(args) > 3 else 0)
            if mode != "constant" or str(node.target).split(".")[1] == "pad"
            else (args[2] if len(args) > 2 else 0)
        )
        value = 0 if value is None else value
        if mode != "constant" or any(v < 0 for v in padding):
            raise NotImplementedError("Only nonnegative constant padding is supported")
        before, after = [0] * x.tensor.ndim, [0] * x.tensor.ndim
        for i in range(len(padding) // 2):
            before[-1 - i], after[-1 - i] = padding[2 * i : 2 * i + 2]
        return self._emit(
            "pad",
            lambda t: F.pad(t, padding, mode, value),
            x,
            padding=before + after,
            mode=mode,
            value=value,
        )

    def _upsample(self, node, args, kwargs):
        x = args[0]
        bilinear = "bilinear" in str(node.target)
        if bilinear and args[2]:
            raise NotImplementedError("Bilinear align_corners=True is unsupported")
        vec = str(node.target).endswith(".vec")
        if vec:
            scales = (
                args[3 if bilinear else 2]
                if len(args) > (3 if bilinear else 2)
                else None
            )
        else:
            scale_args = args[3 if bilinear else 2 :]
            scales = (
                list(scale_args)
                if scale_args and all(s is not None for s in scale_args)
                else None
            )
        if scales is not None and any(float(scale) != int(scale) for scale in scales):
            # pywebnn's scale-based resample inference can round dimensions up
            # where ATen truncates. Fixed index tensors preserve ATen coordinates.
            for axis, scale, size in zip([2, 3], scales, node.meta["val"].shape[-2:]):
                x = self._resample_axis(x, axis, float(scale), int(size), bilinear)
            return x
        options = (
            {"scales": list(scales)} if scales is not None else {"sizes": list(args[1])}
        )
        return self._emit(
            "resample2d",
            lambda t: node.target(t, *args[1:], **kwargs),
            x,
            axes=[2, 3],
            mode="linear" if bilinear else "nearest-neighbor",
            **options,
        )

    def _resample_axis(self, x, axis, scale, size, linear):
        length = x.tensor.shape[axis]
        positions = torch.arange(size, dtype=torch.float64) / scale
        if linear:
            positions = (
                (torch.arange(size, dtype=torch.float64) + 0.5) / scale - 0.5
            ).clamp(min=0)
        lower = positions.floor().to(torch.int32)

        def gather(indices):
            indices = self._constant(indices.clamp(0, length - 1))
            return self._emit(
                "gather",
                lambda t, idx: torch.index_select(t, axis, idx),
                x,
                indices,
                axis=axis,
            )

        low = gather(lower)
        if not linear:
            return low
        high = gather(lower + 1)
        weights = (positions - lower).to(x.tensor.dtype)
        shape = [1] * x.tensor.ndim
        shape[axis] = size
        weights = self._constant(weights.reshape(shape))
        return self._binary(
            "add", low, self._binary("mul", self._binary("sub", high, low), weights)
        )

    def _attention(self, node, args, kwargs):
        name = str(node.target).split(".")[1]
        if name in ("softmax", "_softmax"):
            x, axis = args[:2]
            if name == "_softmax" and len(args) > 2 and args[2]:
                x = self._cast(x, torch.float32)
            if "dtype" in kwargs:
                x = self._cast(x, kwargs["dtype"])
            axis %= x.tensor.ndim
            return self._emit("softmax", lambda t: torch.softmax(t, axis), x, axis=axis)
        q, k, v = args[:3]
        mask = args[3] if len(args) > 3 else None
        dropout = args[4] if len(args) > 4 else 0
        causal = args[5] if len(args) > 5 else False
        if dropout or kwargs.get("enable_gqa", False):
            raise NotImplementedError(
                "Attention dropout and grouped query attention are unsupported"
            )
        perm = list(range(k.tensor.ndim))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        scores = self._matmul(q, self._transpose(k, perm))
        scale = kwargs.get("scale")
        scores = self._binary(
            "mul",
            scores,
            scale if scale is not None else 1 / math.sqrt(q.tensor.shape[-1]),
        )
        if causal:
            if mask is not None:
                raise NotImplementedError(
                    "Causal attention with an explicit mask is unsupported"
                )
            bias = torch.zeros(
                (q.tensor.shape[-2], k.tensor.shape[-2]), dtype=q.tensor.dtype
            )
            bias.masked_fill_(
                ~torch.ones_like(bias, dtype=torch.bool).tril(), float("-inf")
            )
            scores = self._binary("add", scores, self._constant(bias))
        if mask is not None:
            scores = self._binary("add", scores, mask)
        probabilities = self._emit(
            "softmax",
            lambda t: torch.softmax(t, -1),
            scores,
            axis=scores.tensor.ndim - 1,
        )
        return self._matmul(probabilities, v)

    def _constant_op(self, node, args, kwargs):
        # All arguments are fixed scalar dimensions; evaluate without model inputs.
        options = {**kwargs, "device": "cpu"}
        if any(isinstance(x, CheckedOperand) for x in _pytree.tree_leaves(args)):
            raise NotImplementedError(
                f"Data-dependent constant generation: {node.target}"
            )
        return self._constant(node.target(*args, **options))

    def _identity(self, node, args, kwargs):
        if "dropout" in str(node.target) and len(args) > 2 and args[2] and args[1] != 0:
            raise NotImplementedError("Training dropout is unsupported")
        if "_assert_tensor_metadata" in str(node.target):
            return None
        return self._emit("identity", lambda t: t, args[0])

    def _type(self, node, args, kwargs):
        x = args[0]
        value = node.meta["val"]
        return self._cast(x, value.dtype)

    def _getitem(self, node, args, kwargs):
        value = operator.getitem(*args)
        if value is None:
            raise NotImplementedError(
                f"Unused auxiliary output is consumed at {node.name}"
            )
        return value

"""
WebNN Graph Generator - Converts PyTorch ExportedProgram (ATen IR) to WebNN format.

Entry point: WebNNGraphGenerator().generate(ep, graph_name)
where ep is a torch.export.ExportedProgram.
"""

import math
import sys
import torch
import torch.fx as fx
import torch.export
from typing import Dict, List, Optional, Tuple

from .webnn_op_mappings import resolve_aten_converter


def isnumeric(obj):
    try:
        obj + 0
        return True
    except TypeError:
        return False


def throw_unsupported(kind, node):
    msg = (
        f"\n{'=' * 80}\n"
        f"UNSUPPORTED {kind}\n"
        f"{'=' * 80}\n"
        f"Node : {node.name}\n"
        f"Target: {node.target}\n"
        f"Args  : {[str(a) for a in node.args]}\n"
        f"Kwargs: {node.kwargs}\n"
        f"{'=' * 80}\n"
    )
    raise NotImplementedError(msg)


class WebNNGraphGenerator:
    """Generates WebNN graph format from a torch.export.ExportedProgram."""

    def __init__(self):
        self.operand_counter = 1
        self.node_to_operand: Dict[str, str] = {}
        # Maps FX placeholder node name → operand name (for parameters/buffers)
        self.weight_operands: Dict[str, str] = {}
        self.operand_shapes: Dict[str, List[int]] = {}
        self.inline_constants: Dict[str, object] = {}
        # Maps multi-output node name → list of per-output operand names (chunk/unbind/split)
        self.multi_output_operands: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        ep: torch.export.ExportedProgram,
        graph_name: str = "model",
    ) -> str:
        """Generate WebNN graph format from an ExportedProgram.

        Args:
            ep: Result of torch.export.export(model, example_inputs).
            graph_name: Name embedded in the .webnn file header.

        Returns:
            WebNN graph as a string.
        """
        # Reset state
        self.operand_counter = 1
        self.node_to_operand = {}
        self.weight_operands = {}
        self.operand_shapes = {}
        self.inline_constants = {}
        self.multi_output_operands = {}

        gm = ep.graph_module
        sig = ep.graph_signature

        # Set of placeholder names that are actual model inputs (not weights)
        user_inputs: set = set(sig.user_inputs)

        # Mapping: placeholder_node_name → state_dict key
        param_map: Dict[str, str] = {
            **sig.inputs_to_parameters,
            **sig.inputs_to_buffers,
        }

        # Named tensors for weight shape/dtype lookup
        named_params: Dict[str, torch.Tensor] = {
            **dict(ep.named_parameters()),
            **dict(ep.named_buffers()),
        }

        # Build sections
        inputs_section = self._extract_inputs(gm, user_inputs)
        consts_section = self._extract_weights(gm, param_map, named_params)
        nodes_section = self._convert_nodes(gm)
        inline_consts_section = self._extract_inline_constants()
        outputs_section = self._extract_outputs(gm)

        all_consts = ""
        if inline_consts_section:
            all_consts += inline_consts_section
        if consts_section:
            all_consts += consts_section

        graph = f'webnn_graph "{graph_name}" v1 {{\n'
        graph += f"  inputs {{ {inputs_section} }}\n"
        if all_consts:
            graph += f"  consts {{\n{all_consts}  }}\n"
        graph += f"  nodes {{\n{nodes_section}  }}\n"
        graph += f"  outputs {{ {outputs_section} }}\n"
        graph += "}\n"

        return graph

    # ------------------------------------------------------------------
    # Input / weight extraction
    # ------------------------------------------------------------------

    def _extract_inputs(
        self, gm: fx.GraphModule, user_inputs: set
    ) -> str:
        """Emit `inputs {}` section — only real model inputs, not parameters."""
        decls = []
        for node in gm.graph.nodes:
            if node.op != "placeholder":
                continue
            if node.name not in user_inputs:
                continue

            shape = self._get_node_shape(node)
            dtype = self._get_webnn_dtype(self._get_node_dtype(node))
            shape_str = ", ".join(map(str, shape))
            # Use the parameter name from the graph signature (strip trailing '_')
            name = node.name.rstrip("_")
            self.node_to_operand[node.name] = name
            decls.append(f"{name}: {dtype}[{shape_str}]")

        return "; ".join(decls) + ";" if decls else ""

    def _extract_weights(
        self,
        gm: fx.GraphModule,
        param_map: Dict[str, str],
        named_params: Dict[str, torch.Tensor],
    ) -> str:
        """Emit `consts {}` section for model parameters and buffers."""
        consts = []
        for node in gm.graph.nodes:
            if node.op != "placeholder":
                continue
            if node.name not in param_map:
                continue

            state_key = param_map[node.name]
            tensor = named_params.get(state_key)
            if tensor is None:
                continue

            operand_name = f"weight_{self.operand_counter}"
            self.operand_counter += 1

            shape = list(tensor.shape)
            dtype = self._get_webnn_dtype(tensor.dtype)
            shape_str = ", ".join(map(str, shape))

            self.weight_operands[node.name] = operand_name
            self.operand_shapes[operand_name] = shape

            consts.append(
                f'\t{operand_name}: {dtype}[{shape_str}] @weights("{state_key}");'
            )

        return "\n".join(consts) + "\n" if consts else ""

    # ------------------------------------------------------------------
    # Node conversion
    # ------------------------------------------------------------------

    def _convert_nodes(self, gm: fx.GraphModule) -> str:
        """Convert all call_function nodes to WebNN operations."""
        operations = []
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            try:
                op_str = self._map_aten_to_webnn_op(node)
                if op_str:
                    operations.append(f"\t{op_str}")
            except NotImplementedError as e:
                input_operands = [
                    self._get_input_operand(a)
                    for a in node.args
                    if isinstance(a, fx.Node)
                ]
                operations.append(
                    f"\t// unsupported: {node.target} "
                    f"inputs=[{', '.join(input_operands)}] args={list(node.args)}"
                )

        return "\n".join(operations) + "\n" if operations else ""

    def _map_aten_to_webnn_op(self, node: fx.Node) -> str:
        output = self._get_operand_name(node)
        inputs = [
            self._get_input_operand(a)
            for a in node.args
            if isinstance(a, fx.Node)
        ]

        method_name = resolve_aten_converter(node.target)
        if method_name is None:
            throw_unsupported("ATen op", node)

        method = getattr(self, method_name)
        return method(node, output, inputs)

    # ------------------------------------------------------------------
    # Converter methods — each takes (node, output_operand, input_operands)
    # ------------------------------------------------------------------

    # --- Convolution ---

    def _emit_conv2d(
        self, input_tensor, weight, bias_info, stride, padding, dilation, groups, output
    ) -> str:
        def as_pair(v):
            return list(v) if isinstance(v, (list, tuple)) else [v, v]

        stride = as_pair(stride)
        padding = as_pair(padding)
        dilation = as_pair(dilation)

        params = []
        if dilation != [1, 1]:
            params.append(f"dilations=[{dilation[0]}, {dilation[1]}]")
        params.append('filterLayout="oihw"')
        params.append(f"groups={groups}")
        params.append('inputLayout="nchw"')
        if padding != [0, 0]:
            params.append(f"pads=[{padding[0]}, {padding[0]}, {padding[1]}, {padding[1]}]")
        if stride != [1, 1]:
            params.append(f"strides=[{stride[0]}, {stride[1]}]")

        params_str = ", ".join(params)

        if bias_info is not None:
            bias_operand, c = bias_info
            reshaped_bias = f"operand_{self.operand_counter}"
            self.operand_counter += 1
            conv_out = f"operand_{self.operand_counter}"
            self.operand_counter += 1
            return (
                f"[{reshaped_bias}] = reshape({bias_operand}, newShape=[1, {c}, 1, 1]);\n"
                f"\t[{conv_out}] = conv2d({input_tensor}, {weight}, {params_str});\n"
                f"\t[{output}] = add({conv_out}, {reshaped_bias});"
            )
        return f"[{output}] = conv2d({input_tensor}, {weight}, {params_str});"

    def _convert_conv2d(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.conv2d.default(input, weight, bias, stride, padding, dilation, groups)"""
        args = node.args
        input_tensor = inputs[0] if inputs else "unknown"
        weight = self._get_input_operand(args[1]) if len(args) > 1 else "unknown"

        stride = args[3] if len(args) > 3 else [1, 1]
        padding = args[4] if len(args) > 4 else [0, 0]
        dilation = args[5] if len(args) > 5 else [1, 1]
        groups = args[6] if len(args) > 6 else 1

        bias_info = None
        bias_node = args[2] if len(args) > 2 else None
        if isinstance(bias_node, fx.Node):
            bias_operand = self._get_input_operand(bias_node)
            bias_shape = self.operand_shapes.get(bias_operand, [])
            bias_info = (bias_operand, bias_shape[0] if bias_shape else 0)

        return self._emit_conv2d(input_tensor, weight, bias_info, stride, padding, dilation, groups, output)

    def _convert_convolution(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.convolution.default(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)"""
        args = node.args
        input_tensor = inputs[0] if inputs else "unknown"
        weight = self._get_input_operand(args[1]) if len(args) > 1 else "unknown"

        stride = args[3] if len(args) > 3 else [1, 1]
        padding = args[4] if len(args) > 4 else [0, 0]
        dilation = args[5] if len(args) > 5 else [1, 1]
        groups = args[8] if len(args) > 8 else 1

        bias_info = None
        bias_node = args[2] if len(args) > 2 else None
        if isinstance(bias_node, fx.Node):
            bias_operand = self._get_input_operand(bias_node)
            bias_shape = self.operand_shapes.get(bias_operand, [])
            bias_info = (bias_operand, bias_shape[0] if bias_shape else 0)

        return self._emit_conv2d(input_tensor, weight, bias_info, stride, padding, dilation, groups, output)

    # --- Linear ---

    def _convert_linear(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.linear.default(input, weight, bias)"""
        if len(inputs) < 2:
            raise NotImplementedError("linear requires at least 2 inputs")
        input_tensor, weight = inputs[0], inputs[1]

        if len(inputs) >= 3:
            bias = inputs[2]
            mm_out = f"operand_{self.operand_counter}"
            self.operand_counter += 1
            return (
                f"[{mm_out}] = gemm({input_tensor}, {weight}, bTranspose=true);\n"
                f"\t[{output}] = add({mm_out}, {bias});"
            )
        return f"[{output}] = gemm({input_tensor}, {weight}, bTranspose=true);"

    def _convert_addmm(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.addmm.default(bias, mat1, mat2) = bias + mat1 @ mat2"""
        if len(inputs) < 3:
            return self._convert_identity(node, output, inputs)
        bias, mat1, mat2 = inputs[0], inputs[1], inputs[2]
        stmts = []

        input_node = node.args[1] if len(node.args) > 1 and isinstance(node.args[1], fx.Node) else None
        input_shape = self._get_node_shape(input_node) if input_node else []
        if input_shape and len(input_shape) != 2:
            batch = int(input_shape[0])
            features = int(math.prod(input_shape[1:]))
            tmp = f"operand_{self.operand_counter}"
            self.operand_counter += 1
            stmts.append(f"[{tmp}] = reshape({mat1}, newShape=[{batch}, {features}]);")
            mat1 = tmp

        mm_out = f"operand_{self.operand_counter}"
        self.operand_counter += 1
        stmts.append(f"[{mm_out}] = gemm({mat1}, {mat2});")
        stmts.append(f"[{output}] = add({mm_out}, {bias});")
        return "\n\t".join(stmts)

    def _convert_matmul(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.mm / aten.matmul"""
        if len(inputs) < 2:
            raise NotImplementedError("matmul requires 2 inputs")
        a, b = inputs[0], inputs[1]
        stmts = []

        a_node = node.args[0] if isinstance(node.args[0], fx.Node) else None
        b_node = node.args[1] if len(node.args) > 1 and isinstance(node.args[1], fx.Node) else None
        a_shape = self._get_node_shape(a_node) if a_node else self.operand_shapes.get(a, [])
        b_shape = self._get_node_shape(b_node) if b_node else self.operand_shapes.get(b, [])
        out_shape = self._get_node_shape(node)

        if a_shape and len(a_shape) == 1:
            tmp = f"operand_{self.operand_counter}"
            self.operand_counter += 1
            stmts.append(f"[{tmp}] = reshape({a}, newShape=[1, {a_shape[0]}]);")
            a = tmp
        if b_shape and len(b_shape) == 1:
            tmp = f"operand_{self.operand_counter}"
            self.operand_counter += 1
            stmts.append(f"[{tmp}] = reshape({b}, newShape=[{b_shape[0]}, 1]);")
            b = tmp

        needs_reshape = bool(out_shape) and len(out_shape) != 2
        gemm_out = output if not needs_reshape else f"operand_{self.operand_counter}"
        if needs_reshape:
            self.operand_counter += 1

        stmts.append(f"[{gemm_out}] = gemm({a}, {b});")
        if needs_reshape:
            shape_str = ", ".join(str(d) for d in out_shape)
            stmts.append(f"[{output}] = reshape({gemm_out}, newShape=[{shape_str}]);")
        return "\n\t".join(stmts)

    def _convert_t(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.t — transpose a 2-D matrix."""
        input_tensor = inputs[0] if inputs else "unknown"
        return f"[{output}] = transpose({input_tensor}, permutation=[1, 0]);"

    # --- Activations ---

    def _convert_relu(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        return f"[{output}] = clamp({inputs[0] if inputs else 'unknown'}, minValue=0.0);"

    def _convert_sigmoid(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        return f"[{output}] = sigmoid({inputs[0] if inputs else 'unknown'});"

    def _convert_tanh(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        return f"[{output}] = tanh({inputs[0] if inputs else 'unknown'});"

    def _convert_silu(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        x = inputs[0] if inputs else "unknown"
        sig = f"operand_{self.operand_counter}"
        self.operand_counter += 1
        return f"[{sig}] = sigmoid({x});\n\t[{output}] = mul({x}, {sig});"

    def _convert_gelu(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """GELU via tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))"""
        x = inputs[0] if inputs else "unknown"
        c1 = self._create_inline_constant(0.7978845608028654)   # sqrt(2/pi)
        c2 = self._create_inline_constant(0.044715)
        c3 = self._create_inline_constant(3.0)
        c_half = self._create_inline_constant(0.5)
        c_one = self._create_inline_constant(1.0)

        def tmp():
            name = f"operand_{self.operand_counter}"
            self.operand_counter += 1
            return name

        x3, inner, scaled, tanh_in, tanh_out, one_plus, half_x, r = (
            tmp(), tmp(), tmp(), tmp(), tmp(), tmp(), tmp(), output
        )
        stmts = [
            f"[{x3}] = pow({x}, {c3});",
            f"[{inner}] = mul({x3}, {c2});",
            f"[{inner}] = add({x}, {inner});",
            f"[{scaled}] = mul({inner}, {c1});",
            f"[{tanh_out}] = tanh({scaled});",
            f"[{one_plus}] = add({tanh_out}, {c_one});",
            f"[{half_x}] = mul({x}, {c_half});",
            f"[{r}] = mul({half_x}, {one_plus});",
        ]
        return "\n\t".join(stmts)

    def _convert_hardtanh(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        x = inputs[0] if inputs else "unknown"
        args = node.args
        min_val = args[1] if len(args) > 1 else 0.0
        max_val = args[2] if len(args) > 2 else 6.0
        return f"[{output}] = clamp({x}, minValue={min_val}, maxValue={max_val});"

    def _convert_clamp(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        x = inputs[0] if inputs else "unknown"
        args = node.args
        params = []
        min_val = args[1] if len(args) > 1 else node.kwargs.get("min")
        max_val = args[2] if len(args) > 2 else node.kwargs.get("max")
        if min_val is not None:
            params.append(f"minValue={min_val}")
        if max_val is not None:
            params.append(f"maxValue={max_val}")
        return f"[{output}] = clamp({x}, {', '.join(params)});"

    # --- Normalization ---

    def _convert_batch_norm_aten(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.batch_norm.default(input, weight, bias, running_mean, running_var,
                                    training, momentum, eps, cudnn_enabled)"""
        args = node.args

        def get_op(idx):
            n = args[idx] if len(args) > idx else None
            return self._get_input_operand(n) if isinstance(n, fx.Node) else None

        input_tensor = get_op(0) or "unknown"
        weight = get_op(1)       # gamma / scale
        bias_op = get_op(2)      # beta / bias
        mean_op = get_op(3)      # running_mean
        var_op = get_op(4)       # running_var
        eps = args[7] if len(args) > 7 else 1e-5

        if mean_op and var_op:
            params = [f"epsilon={eps}", "axis=1"]
            if weight:
                params.append(f"scale={weight}")
            if bias_op:
                params.append(f"bias={bias_op}")
            # TODO fix parser to accept mean and variance as named args as well
            # return f"[{output}] = batchNormalization({input_tensor}, mean={mean_op}, variance={var_op}, {', '.join(params)});"
            return f"[{output}] = batchNormalization({input_tensor}, {mean_op}, {var_op}, {', '.join(params)});"

        # No running stats — layer-norm style decomposition over NCHW [0,2,3]
        return self._batch_norm_decompose(input_tensor, weight, bias_op, eps, node, output)

    def _convert_batch_norm_no_training(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten._native_batch_norm_legit_no_training.default(input, weight, bias, running_mean, running_var, momentum, eps)"""
        args = node.args

        def get_op(idx):
            n = args[idx] if len(args) > idx else None
            return self._get_input_operand(n) if isinstance(n, fx.Node) else None

        input_tensor = get_op(0) or "unknown"
        weight = get_op(1)
        bias_op = get_op(2)
        mean_op = get_op(3)
        var_op = get_op(4)
        eps = args[6] if len(args) > 6 else 1e-5

        if mean_op and var_op:
            params = [f"epsilon={eps}", "axis=1"]
            if weight:
                params.append(f"scale={weight}")
            if bias_op:
                params.append(f"bias={bias_op}")
            return f"[{output}] = batchNormalization({input_tensor}, {mean_op}, {var_op}, {', '.join(params)});"

        return self._batch_norm_decompose(input_tensor, weight, bias_op, eps, node, output)

    def _batch_norm_decompose(self, input_tensor, weight, bias_op, eps, node, output) -> str:
        """Decompose batch norm into mean/var/normalize ops for NCHW [0,2,3]."""
        input_shape = self._get_node_shape(node.args[0]) if node.args and isinstance(node.args[0], fx.Node) else []
        C = input_shape[1] if len(input_shape) > 1 else 0
        eps_c = self._create_inline_constant(float(eps))

        def tmp():
            name = f"operand_{self.operand_counter}"
            self.operand_counter += 1
            return name

        mean_t, centered, sq, var_t, var_eps, std_t, norm_t = (
            tmp(), tmp(), tmp(), tmp(), tmp(), tmp(), tmp()
        )
        steps = [
            f"[{mean_t}] = reduceMean({input_tensor}, axes=[0, 2, 3], keepDimensions=true);",
            f"[{centered}] = sub({input_tensor}, {mean_t});",
            f"[{sq}] = mul({centered}, {centered});",
            f"[{var_t}] = reduceMean({sq}, axes=[0, 2, 3], keepDimensions=true);",
            f"[{var_eps}] = add({var_t}, {eps_c});",
            f"[{std_t}] = sqrt({var_eps});",
            f"[{norm_t}] = div({centered}, {std_t});",
        ]
        result = norm_t

        if weight and C:
            w_shaped, scaled = tmp(), tmp()
            steps += [
                f"[{w_shaped}] = reshape({weight}, newShape=[1, {C}, 1, 1]);",
                f"[{scaled}] = mul({result}, {w_shaped});",
            ]
            result = scaled

        if bias_op and C:
            b_shaped = tmp()
            steps += [
                f"[{b_shaped}] = reshape({bias_op}, newShape=[1, {C}, 1, 1]);",
                f"[{output}] = add({result}, {b_shaped});",
            ]
        else:
            steps.append(f"[{output}] = identity({result});")

        return "\n\t".join(steps)

    def _convert_layer_norm(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.layer_norm.default(input, normalized_shape, weight, bias, eps)"""
        x = inputs[0] if inputs else "unknown"
        args = node.args
        weight = self._get_input_operand(args[2]) if len(args) > 2 and isinstance(args[2], fx.Node) else None
        bias_op = self._get_input_operand(args[3]) if len(args) > 3 and isinstance(args[3], fx.Node) else None
        eps = args[4] if len(args) > 4 else 1e-5
        params = []
        if weight:
            params.append(f"scale={weight}")
        if bias_op:
            params.append(f"bias={bias_op}")
        params.append(f"epsilon={eps}")
        return f"[{output}] = layerNormalization({x}, {', '.join(params)});"

    def _convert_group_norm(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.group_norm.default(input, num_groups, weight, bias, eps)"""
        x = inputs[0] if inputs else "unknown"
        args = node.args
        num_groups = args[1] if len(args) > 1 else 32
        weight = self._get_input_operand(args[2]) if len(args) > 2 and isinstance(args[2], fx.Node) else None
        bias_op = self._get_input_operand(args[3]) if len(args) > 3 and isinstance(args[3], fx.Node) else None
        eps = args[4] if len(args) > 4 else 1e-5
        input_shape = self._get_node_shape(args[0]) if args and isinstance(args[0], fx.Node) else []

        if not input_shape or len(input_shape) < 2:
            params = []
            if weight:
                params.append(f"scale={weight}")
            if bias_op:
                params.append(f"bias={bias_op}")
            params.append(f"epsilon={eps}")
            return f"[{output}] = layerNormalization({x}, {', '.join(params)}); // approx group_norm"

        N, C = input_shape[0], input_shape[1]
        spatial = input_shape[2:]
        cpg = C // num_groups
        eps_c = self._create_inline_constant(float(eps))

        def tmp():
            name = f"operand_{self.operand_counter}"
            self.operand_counter += 1
            return name

        r1 = tmp()
        r1_shape = [N, num_groups, cpg] + spatial
        mean_axes = list(range(2, 2 + 1 + len(spatial)))
        axes_str = ", ".join(map(str, mean_axes))

        mean_t, centered, sq, var_t, var_eps, std_t, norm_t, r_back = (
            tmp(), tmp(), tmp(), tmp(), tmp(), tmp(), tmp(), tmp()
        )
        orig_str = ", ".join(map(str, input_shape))
        steps = [
            f"[{r1}] = reshape({x}, newShape=[{', '.join(map(str, r1_shape))}]);",
            f"[{mean_t}] = reduceMean({r1}, axes=[{axes_str}], keepDimensions=true);",
            f"[{centered}] = sub({r1}, {mean_t});",
            f"[{sq}] = mul({centered}, {centered});",
            f"[{var_t}] = reduceMean({sq}, axes=[{axes_str}], keepDimensions=true);",
            f"[{var_eps}] = add({var_t}, {eps_c});",
            f"[{std_t}] = sqrt({var_eps});",
            f"[{norm_t}] = div({centered}, {std_t});",
            f"[{r_back}] = reshape({norm_t}, newShape=[{orig_str}]);",
        ]
        result = r_back

        if weight:
            w_shaped, scaled = tmp(), tmp()
            w_shape = [1] * len(input_shape)
            w_shape[1] = self.operand_shapes.get(weight, [C])[0]
            steps += [
                f"[{w_shaped}] = reshape({weight}, newShape=[{', '.join(map(str, w_shape))}]);",
                f"[{scaled}] = mul({result}, {w_shaped});",
            ]
            result = scaled

        if bias_op:
            b_shaped = tmp()
            b_shape = [1] * len(input_shape)
            b_shape[1] = self.operand_shapes.get(bias_op, [C])[0]
            steps += [
                f"[{b_shaped}] = reshape({bias_op}, newShape=[{', '.join(map(str, b_shape))}]);",
                f"[{output}] = add({result}, {b_shaped});",
            ]
        else:
            steps.append(f"[{output}] = identity({result});")

        return "\n\t".join(steps)

    # --- Pooling ---

    def _convert_max_pool2d(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.max_pool2d.default / max_pool2d_with_indices.default"""
        x = inputs[0] if inputs else "unknown"
        args = node.args
        kernel = args[1] if len(args) > 1 else node.kwargs.get("kernel_size", [2, 2])
        stride = args[2] if len(args) > 2 else node.kwargs.get("stride", kernel)
        padding = args[3] if len(args) > 3 else node.kwargs.get("padding", [0, 0])
        if not isinstance(kernel, (list, tuple)):
            kernel = [kernel, kernel]
        if not isinstance(stride, (list, tuple)):
            stride = [stride, stride]
        if not isinstance(padding, (list, tuple)):
            padding = [padding, padding, padding, padding]
        elif len(padding) == 2:
            padding = [padding[0], padding[0], padding[1], padding[1]]

        params = [f"windowDimensions=[{kernel[0]}, {kernel[1]}]"]
        if stride != [1, 1]:
            params.append(f"strides=[{stride[0]}, {stride[1]}]")
        if padding != [0, 0, 0, 0]:
            params.append(f"padding=[{padding[0]}, {padding[1]}, {padding[2]}, {padding[3]}]")

        # max_pool2d_with_indices returns a tuple; the first element is the pooled tensor.
        # The FX node itself represents index 0 (values), index 1 (indices) via getitem.
        return f"[{output}] = maxPool2d({x}, {', '.join(params)});"

    def _convert_avg_pool2d(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        x = inputs[0] if inputs else "unknown"
        args = node.args
        kernel = args[1] if len(args) > 1 else [2, 2]
        stride = args[2] if len(args) > 2 else kernel
        padding = args[3] if len(args) > 3 else [0, 0]
        if not isinstance(kernel, (list, tuple)):
            kernel = [kernel, kernel]
        if not isinstance(stride, (list, tuple)):
            stride = [stride, stride]
        if not isinstance(padding, (list, tuple)):
            padding = [padding, padding, padding, padding]
        elif len(padding) == 2:
            padding = [padding[0], padding[0], padding[1], padding[1]]

        params = [f"windowDimensions=[{kernel[0]}, {kernel[1]}]"]
        if stride != [1, 1]:
            params.append(f"strides=[{stride[0]}, {stride[1]}]")
        if padding != [0, 0, 0, 0]:
            params.append(f"padding=[{padding[0]}, {padding[1]}, {padding[2]}, {padding[3]}]")
        return f"[{output}] = averagePool2d({x}, {', '.join(params)});"

    def _convert_global_avg_pool(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        x = inputs[0] if inputs else "unknown"
        return f"[{output}] = reduceMean({x}, axes=[2, 3], keepDimensions=true);"

    def _convert_reduce_mean(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        x = inputs[0] if inputs else "unknown"
        args = node.args
        axes = None
        keep = True
        if "dim" in node.kwargs:
            axes = node.kwargs["dim"]
        elif len(args) > 1:
            axes = args[1]
        if "keepdim" in node.kwargs:
            keep = node.kwargs["keepdim"]
        elif len(args) > 2:
            keep = args[2]
        if axes is not None:
            if not isinstance(axes, (list, tuple)):
                axes = [axes]
            axes_str = ", ".join(map(str, axes))
            return f"[{output}] = reduceMean({x}, axes=[{axes_str}], keepDimensions={'true' if keep else 'false'});"
        return f"[{output}] = reduceMean({x});"

    # --- Arithmetic ---

    def _make_arithmetic(self, op: str, node: fx.Node, output: str, inputs: List[str]) -> str:
        if len(inputs) == 2:
            return f"[{output}] = {op}({inputs[0]}, {inputs[1]});"
        if len(inputs) == 1:
            # scalar second operand
            for arg in node.args:
                if isnumeric(arg) and not isinstance(arg, fx.Node):
                    const = self._create_inline_constant(arg)
                    return f"[{output}] = {op}({inputs[0]}, {const});"
        raise NotImplementedError(f"Invalid {op} operation: inputs={inputs} args={node.args}")

    def _convert_add(self, node, output, inputs):
        return self._make_arithmetic("add", node, output, inputs)

    def _convert_sub(self, node, output, inputs):
        return self._make_arithmetic("sub", node, output, inputs)

    def _convert_mul(self, node, output, inputs):
        return self._make_arithmetic("mul", node, output, inputs)

    def _convert_div(self, node, output, inputs):
        return self._make_arithmetic("div", node, output, inputs)

    def _convert_neg(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        return f"[{output}] = neg({inputs[0] if inputs else 'unknown'});"

    def _convert_pow(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        if len(inputs) >= 2:
            return f"[{output}] = pow({inputs[0]}, {inputs[1]});"
        # Scalar exponent in args
        x = inputs[0] if inputs else "unknown"
        exp_val = node.args[1] if len(node.args) > 1 else None
        if exp_val is not None and not isinstance(exp_val, fx.Node):
            const = self._create_inline_constant(float(exp_val))
            return f"[{output}] = pow({x}, {const});"
        raise NotImplementedError("pow: cannot determine exponent")

    def _convert_pow_scalar(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.pow.Scalar(scalar_base, tensor_exponent) — scalar ** tensor.
        Decomposed as exp(log(base) * exponent).
        """
        base = node.args[0] if node.args else 1.0
        x = inputs[0] if inputs else "unknown"
        log_base = math.log(float(base))
        log_c = self._create_inline_constant(log_base)
        scaled = f"operand_{self.operand_counter}"
        self.operand_counter += 1
        return f"[{scaled}] = mul({x}, {log_c});\n\t[{output}] = exp({scaled});"

    # --- Elementwise math ---

    def _convert_math_sqrt(self, node, output, inputs):
        return f"[{output}] = sqrt({inputs[0] if inputs else 'unknown'});"

    def _convert_math_exp(self, node, output, inputs):
        return f"[{output}] = exp({inputs[0] if inputs else 'unknown'});"

    def _convert_math_abs(self, node, output, inputs):
        return f"[{output}] = abs({inputs[0] if inputs else 'unknown'});"

    def _convert_math_log(self, node, output, inputs):
        return f"[{output}] = log({inputs[0] if inputs else 'unknown'});"

    def _convert_math_cos(self, node, output, inputs):
        return f"[{output}] = cos({inputs[0] if inputs else 'unknown'});"

    def _convert_math_sin(self, node, output, inputs):
        return f"[{output}] = sin({inputs[0] if inputs else 'unknown'});"

    def _convert_rsqrt(self, node, output, inputs):
        """aten.rsqrt.default(x) = 1 / sqrt(x)"""
        x = inputs[0] if inputs else "unknown"
        sqrt_op = f"operand_{self.operand_counter}"
        self.operand_counter += 1
        one_c = self._create_inline_constant(1.0)
        return f"[{sqrt_op}] = sqrt({x});\n\t[{output}] = div({one_c}, {sqrt_op});"

    def _convert_reciprocal(self, node, output, inputs):
        """aten.reciprocal.default(x) = 1 / x"""
        x = inputs[0] if inputs else "unknown"
        one_c = self._create_inline_constant(1.0)
        return f"[{output}] = div({one_c}, {x});"

    # --- Shape manipulation ---

    def _convert_reshape(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        x = inputs[0] if inputs else "unknown"

        # Prefer static output shape from FX metadata
        meta_shape = self._get_node_shape(node)
        if meta_shape:
            return f"[{output}] = reshape({x}, newShape=[{', '.join(map(str, meta_shape))}]);"

        # aten.flatten.using_ints(input, start_dim, end_dim)
        if len(node.args) >= 2 and isinstance(node.args[1], int):
            start_dim = int(node.args[1])
            end_dim = int(node.args[2]) if len(node.args) > 2 and isinstance(node.args[2], int) else -1
            if node.args and isinstance(node.args[0], fx.Node):
                in_shape = self._get_node_shape(node.args[0])
                if in_shape:
                    rank = len(in_shape)
                    if end_dim < 0:
                        end_dim += rank
                    if 0 <= start_dim <= end_dim < rank:
                        flat_dim = math.prod(in_shape[start_dim:end_dim + 1])
                        new_shape = in_shape[:start_dim] + [int(flat_dim)] + in_shape[end_dim + 1:]
                        return f"[{output}] = reshape({x}, newShape=[{', '.join(map(str, new_shape))}]);"

        # aten.reshape / aten.view: second arg is the shape list
        if len(node.args) > 1:
            new_shape = node.args[1]
            if isinstance(new_shape, (list, tuple)):
                return f"[{output}] = reshape({x}, newShape=[{', '.join(map(str, new_shape))}]);"

        raise NotImplementedError(f"Cannot determine reshape target shape for {node}")

    def _convert_permute(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.permute.default(input, dims_list) — dims is a single list arg."""
        x = inputs[0] if inputs else "unknown"
        dims = node.args[1] if len(node.args) > 1 else None
        if isinstance(dims, (list, tuple)):
            perm_str = ", ".join(map(str, dims))
            return f"[{output}] = transpose({x}, permutation=[{perm_str}]);"
        raise NotImplementedError(f"permute: cannot determine dims from {node.args}")

    def _convert_transpose(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.transpose.int(input, dim0, dim1) — swaps two dims."""
        x = inputs[0] if inputs else "unknown"
        if len(node.args) >= 3:
            dim0, dim1 = int(node.args[1]), int(node.args[2])
            in_shape = self._get_node_shape(node.args[0]) if isinstance(node.args[0], fx.Node) else []
            rank = len(in_shape)
            if rank == 0:
                raise NotImplementedError("transpose: unknown rank")
            perm = list(range(rank))
            d0 = dim0 % rank
            d1 = dim1 % rank
            perm[d0], perm[d1] = perm[d1], perm[d0]
            return f"[{output}] = transpose({x}, permutation=[{', '.join(map(str, perm))}]);"
        return f"[{output}] = transpose({x});"

    def _convert_unsqueeze(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        x = inputs[0] if inputs else "unknown"
        in_shape = self._get_node_shape(node.args[0]) if isinstance(node.args[0], fx.Node) else []
        dim = int(node.args[1]) if len(node.args) > 1 else 0
        rank = len(in_shape)
        if dim < 0:
            dim = rank + 1 + dim
        new_shape = list(in_shape[:dim]) + [1] + list(in_shape[dim:])
        return f"[{output}] = reshape({x}, newShape=[{', '.join(map(str, new_shape))}]);"

    def _convert_squeeze(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        x = inputs[0] if inputs else "unknown"
        out_shape = self._get_node_shape(node)
        if out_shape:
            return f"[{output}] = reshape({x}, newShape=[{', '.join(map(str, out_shape))}]);"
        return f"[{output}] = identity({x});"

    def _convert_concat(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        axis = 0
        if "dim" in node.kwargs:
            axis = node.kwargs["dim"]
        elif len(node.args) > 1 and not isinstance(node.args[1], fx.Node):
            axis = node.args[1]
        if inputs:
            return f"[{output}] = concat([{', '.join(inputs)}], axis={axis});"
        if len(node.args) >= 1 and isinstance(node.args[0], (list, tuple)):
            ops = ", ".join(self._get_input_operand(n) for n in node.args[0] if isinstance(n, fx.Node))
            return f"[{output}] = concat([{ops}], axis={axis});"
        raise NotImplementedError("concat: no inputs")

    def _convert_stack(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        # dim may be in kwargs or in node.args[1]
        if "dim" in node.kwargs:
            dim = node.kwargs["dim"]
        elif len(node.args) > 1 and isinstance(node.args[1], int):
            dim = node.args[1]
        else:
            dim = 0

        # inputs may be empty when the tensor list is packed as node.args[0]
        tensors = node.args[0] if isinstance(node.args[0], (list, tuple)) else []
        tensor_inputs = [self._get_input_operand(n) for n in tensors if isinstance(n, fx.Node)]
        if not tensor_inputs:
            tensor_inputs = inputs
        if not tensor_inputs:
            raise NotImplementedError("stack: no inputs")

        first = tensors[0] if tensors and isinstance(tensors[0], fx.Node) else None
        in_shape = self._get_node_shape(first) if first else []
        steps = []
        unsqueezed = []
        for inp in tensor_inputs:
            us = f"operand_{self.operand_counter}"
            self.operand_counter += 1
            unsqueezed.append(us)
            if in_shape:
                pos = dim if dim >= 0 else len(in_shape) + 1 + dim
                new_shape = list(in_shape[:pos]) + [1] + list(in_shape[pos:])
                steps.append(f"[{us}] = reshape({inp}, newShape=[{', '.join(map(str, new_shape))}]);")
            else:
                steps.append(f"[{us}] = reshape({inp}, newShape=[1]);")
        concat_dim = dim if dim >= 0 else len(in_shape) + 1 + dim
        steps.append(f"[{output}] = concat([{', '.join(unsqueezed)}], axis={concat_dim});")
        return "\n\t".join(steps)

    def _convert_split(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        x = inputs[0] if inputs else "unknown"
        sections = node.args[1] if len(node.args) > 1 else None
        dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
        if isinstance(sections, (list, tuple)):
            # Multi-output split: pre-allocate one operand per section
            out_ops = []
            for _ in sections:
                op = f"operand_{self.operand_counter}"
                self.operand_counter += 1
                out_ops.append(op)
            self.multi_output_operands[node.name] = out_ops
            return f"[{', '.join(out_ops)}] = split({x}, splits=[{', '.join(map(str, sections))}], axis={dim});"
        # Even-split: need shape to compute sizes
        in_shape = self._get_node_shape(node.args[0]) if node.args and isinstance(node.args[0], fx.Node) else []
        if in_shape and sections is not None:
            dim_n = int(dim) % len(in_shape)
            dim_size = in_shape[dim_n]
            n = int(sections)
            base = dim_size // n
            rem = dim_size % n
            sizes = [base + (1 if i < rem else 0) for i in range(n)]
            out_ops = []
            for _ in sizes:
                op = f"operand_{self.operand_counter}"
                self.operand_counter += 1
                out_ops.append(op)
            self.multi_output_operands[node.name] = out_ops
            return f"[{', '.join(out_ops)}] = split({x}, splits=[{', '.join(map(str, sizes))}], axis={dim});"
        return f"[{output}] = split({x}, splits={sections}, axis={dim});"

    def _convert_chunk(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.chunk.default(tensor, chunks, dim) — decompose into slice ops per chunk."""
        x = inputs[0] if inputs else "unknown"
        n_chunks = int(node.args[1]) if len(node.args) > 1 else 1
        dim = int(node.args[2]) if len(node.args) > 2 else 0
        in_shape = self._get_node_shape(node.args[0]) if node.args and isinstance(node.args[0], fx.Node) else []
        if not in_shape:
            raise NotImplementedError(f"chunk: unknown input shape for {node.name}")
        rank = len(in_shape)
        dim = dim % rank
        dim_size = in_shape[dim]
        # Compute actual chunk sizes (last chunk may be smaller)
        base = (dim_size + n_chunks - 1) // n_chunks
        sizes = []
        remaining = dim_size
        for _ in range(n_chunks):
            s = min(base, remaining)
            if s <= 0:
                break
            sizes.append(s)
            remaining -= s
        actual_n = len(sizes)
        steps = []
        out_ops = []
        offset = 0
        for s in sizes:
            op = f"operand_{self.operand_counter}"
            self.operand_counter += 1
            out_ops.append(op)
            starts = [0] * rank
            slice_sizes = list(in_shape)
            starts[dim] = offset
            slice_sizes[dim] = s
            steps.append(
                f"[{op}] = slice({x}, starts=[{', '.join(map(str, starts))}], "
                f"sizes=[{', '.join(map(str, slice_sizes))}]);"
            )
            offset += s
        self.multi_output_operands[node.name] = out_ops
        # Register the first output as this node's primary operand (satisfies downstream identity)
        self.node_to_operand[node.name] = out_ops[0]
        return "\n\t".join(steps)

    def _convert_unbind(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.unbind.int(tensor, dim) — split into individual tensors along dim."""
        x = inputs[0] if inputs else "unknown"
        dim = int(node.args[1]) if len(node.args) > 1 else 0
        in_shape = self._get_node_shape(node.args[0]) if node.args and isinstance(node.args[0], fx.Node) else []
        if not in_shape:
            raise NotImplementedError(f"unbind: unknown input shape for {node.name}")
        rank = len(in_shape)
        dim = dim % rank
        dim_size = in_shape[dim]
        out_shape = list(in_shape[:dim]) + list(in_shape[dim + 1:])
        steps = []
        out_ops = []
        for i in range(dim_size):
            slice_op = f"operand_{self.operand_counter}"
            self.operand_counter += 1
            squeeze_op = f"operand_{self.operand_counter}"
            self.operand_counter += 1
            starts = [0] * rank
            sizes = list(in_shape)
            starts[dim] = i
            sizes[dim] = 1
            steps.append(
                f"[{slice_op}] = slice({x}, starts=[{', '.join(map(str, starts))}], "
                f"sizes=[{', '.join(map(str, sizes))}]);"
            )
            steps.append(
                f"[{squeeze_op}] = reshape({slice_op}, newShape=[{', '.join(map(str, out_shape))}]);"
            )
            out_ops.append(squeeze_op)
        self.multi_output_operands[node.name] = out_ops
        self.node_to_operand[node.name] = out_ops[0]
        return "\n\t".join(steps)

    def _convert_getitem(self, node: fx.Node, output: str, inputs: List[str]) -> Optional[str]:
        """operator.getitem — index into multi-output results (chunk/unbind/split)."""
        source = node.args[0] if node.args and isinstance(node.args[0], fx.Node) else None
        idx = node.args[1] if len(node.args) > 1 else 0
        if source is not None and source.name in self.multi_output_operands:
            operands = self.multi_output_operands[source.name]
            if isinstance(idx, int) and 0 <= idx < len(operands):
                # Alias this node to the pre-computed slice operand (no new op needed)
                self.node_to_operand[node.name] = operands[idx]
                return None
        # Fallback: treat as identity of the source
        if inputs:
            return f"[{output}] = identity({inputs[0]});"
        raise NotImplementedError(f"getitem: cannot resolve index {idx} for {node.name}")

    def _convert_select(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.select.int(tensor, dim, index) — select one slice along dim, removing it."""
        x = inputs[0] if inputs else "unknown"
        dim = int(node.args[1]) if len(node.args) > 1 else 0
        index = int(node.args[2]) if len(node.args) > 2 else 0
        in_shape = self._get_node_shape(node.args[0]) if node.args and isinstance(node.args[0], fx.Node) else []
        if not in_shape:
            raise NotImplementedError(f"select: unknown input shape for {node.name}")
        rank = len(in_shape)
        dim = dim % rank
        if index < 0:
            index = in_shape[dim] + index
        starts = [0] * rank
        sizes = list(in_shape)
        starts[dim] = index
        sizes[dim] = 1
        out_shape = list(in_shape[:dim]) + list(in_shape[dim + 1:])
        slice_op = f"operand_{self.operand_counter}"
        self.operand_counter += 1
        return (
            f"[{slice_op}] = slice({x}, starts=[{', '.join(map(str, starts))}], "
            f"sizes=[{', '.join(map(str, sizes))}]);\n"
            f"\t[{output}] = reshape({slice_op}, newShape=[{', '.join(map(str, out_shape))}]);"
        )

    def _convert_einsum(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.einsum.default(equation, operands) — decompose common patterns."""
        equation = node.args[0] if node.args else ""
        tensors_arg = node.args[1] if len(node.args) > 1 and isinstance(node.args[1], (list, tuple)) else []
        operands = [self._get_input_operand(n) for n in tensors_arg if isinstance(n, fx.Node)]

        # Pattern: '...n,d->...nd'  (broadcast outer product: [...,n] x [d] -> [...,n,d])
        if equation == "...n,d->...nd" and len(operands) == 2:
            a, b = operands[0], operands[1]
            a_node = tensors_arg[0] if tensors_arg and isinstance(tensors_arg[0], fx.Node) else None
            b_node = tensors_arg[1] if len(tensors_arg) > 1 and isinstance(tensors_arg[1], fx.Node) else None
            a_shape = self._get_node_shape(a_node) if a_node else []
            out_shape = self._get_node_shape(node)
            if not out_shape:
                raise NotImplementedError(f"einsum '{equation}': cannot determine output shape")
            rank_out = len(out_shape)

            def tmp():
                name = f"operand_{self.operand_counter}"
                self.operand_counter += 1
                return name

            # Unsqueeze a along last axis: [..., n] → [..., n, 1]
            a_us = tmp()
            a_us_shape = list(a_shape) + [1]
            # Reshape b to [1, ..., 1, d] matching output rank
            b_node_shape = self._get_node_shape(b_node) if b_node else []
            d = b_node_shape[0] if b_node_shape else out_shape[-1]
            b_rs = tmp()
            b_rs_shape = [1] * (rank_out - 1) + [d]
            return "\n\t".join([
                f"[{a_us}] = reshape({a}, newShape=[{', '.join(map(str, a_us_shape))}]);",
                f"[{b_rs}] = reshape({b}, newShape=[{', '.join(map(str, b_rs_shape))}]);",
                f"[{output}] = mul({a_us}, {b_rs});",
            ])

        raise NotImplementedError(f"einsum: unsupported equation '{equation}'")

    def _convert_slice(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.slice.Tensor(input, dim, start, end, step)
        start / end can be None (meaning 0 / full dimension respectively).
        end can also be 9223372036854775807 (sys.maxsize) meaning full dimension.
        """
        x = inputs[0] if inputs else "unknown"
        args = node.args
        dim = int(args[1]) if len(args) > 1 else 0
        start = args[2] if len(args) > 2 else None
        end = args[3] if len(args) > 3 else None
        step = int(args[4]) if len(args) > 4 and args[4] is not None else 1
        in_shape = self._get_node_shape(args[0]) if args and isinstance(args[0], fx.Node) else []

        if not in_shape:
            return f"// slice: unknown input shape for {node.name}"

        rank = len(in_shape)
        dim = dim % rank
        dim_size = in_shape[dim]

        # Resolve None / sentinel values
        start = 0 if start is None else int(start)
        end = dim_size if (end is None or end >= sys.maxsize) else min(int(end), dim_size)

        # Handle negative indices
        if start < 0:
            start = max(0, dim_size + start)
        if end < 0:
            end = max(0, dim_size + end)

        size = max(0, (end - start + step - 1) // step)  # ceiling div for step > 1

        # If the slice covers the full dimension it's a no-op
        if start == 0 and size == dim_size and step == 1:
            return f"[{output}] = identity({x});"

        starts = [0] * rank
        sizes = list(in_shape)
        starts[dim] = start
        sizes[dim] = size
        return f"[{output}] = slice({x}, starts=[{', '.join(map(str, starts))}], sizes=[{', '.join(map(str, sizes))}]);"

    def _convert_expand(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        x = inputs[0] if inputs else "unknown"
        shape = node.args[1] if len(node.args) > 1 else None
        if shape and isinstance(shape, (list, tuple)):
            return f"[{output}] = expand({x}, newShape=[{', '.join(map(str, shape))}]);"
        return f"// expand: unknown shape for {node.name}"

    def _convert_expand_as(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        x = inputs[0] if inputs else "unknown"
        target_node = node.args[1] if len(node.args) > 1 and isinstance(node.args[1], fx.Node) else None
        shape = self._get_node_shape(target_node) if target_node else []
        if shape:
            return f"[{output}] = expand({x}, newShape=[{', '.join(map(str, shape))}]);"
        return f"[{output}] = identity({x}); // expand_as: unknown target shape"

    def _convert_pad(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        x = inputs[0] if inputs else "unknown"
        padding = node.args[1] if len(node.args) > 1 else [0, 0, 0, 0]
        mode = node.kwargs.get("mode", "constant")
        value = node.kwargs.get("value", 0)
        if isinstance(padding, (list, tuple)):
            pad_str = ", ".join(map(str, padding))
            if mode == "constant":
                return f'[{output}] = pad({x}, padding=[{pad_str}], mode="constant", value={value});'
            return f'[{output}] = pad({x}, padding=[{pad_str}], mode="{mode}");'
        return f"// pad: unknown parameters for {node.name}"

    # --- Upsampling ---

    def _convert_upsample_nearest2d(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.upsample_nearest2d.vec(input, output_size, scale_factors)
           aten.upsample_nearest2d.default(input, output_size, scales_h, scales_w)
        """
        x = inputs[0] if inputs else "unknown"
        args = node.args
        target_str = str(node.target)

        if "vec" in target_str:
            output_size  = args[1] if len(args) > 1 else None  # int[] or None
            scale_factors = args[2] if len(args) > 2 else None  # float[] or None
            if scale_factors:
                scales_str = ", ".join(str(s) for s in scale_factors)
                return f'[{output}] = resample2d({x}, mode="nearest-neighbor", scales=[{scales_str}]);'
            if output_size:
                sizes_str = ", ".join(str(s) for s in output_size)
                return f'[{output}] = resample2d({x}, mode="nearest-neighbor", sizes=[{sizes_str}]);'
        else:
            # .default(input, output_size, scales_h=None, scales_w=None)
            output_size = args[1] if len(args) > 1 else None
            scales_h = args[2] if len(args) > 2 else None
            scales_w = args[3] if len(args) > 3 else None
            if scales_h is not None and scales_w is not None:
                return f'[{output}] = resample2d({x}, mode="nearest-neighbor", scales=[{scales_h}, {scales_w}]);'
            if output_size:
                sizes_str = ", ".join(str(s) for s in output_size)
                return f'[{output}] = resample2d({x}, mode="nearest-neighbor", sizes=[{sizes_str}]);'

        raise NotImplementedError(f"upsample_nearest2d: need output_size or scale_factors, got args={args}")

    def _convert_upsample_bilinear2d(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.upsample_bilinear2d.vec / .default → resample2d with mode="linear" """
        x = inputs[0] if inputs else "unknown"
        args = node.args
        target_str = str(node.target)

        if "vec" in target_str:
            output_size   = args[1] if len(args) > 1 else None
            scale_factors = args[3] if len(args) > 3 else None  # arg[2] is align_corners
            if scale_factors:
                scales_str = ", ".join(str(s) for s in scale_factors)
                return f'[{output}] = resample2d({x}, mode="linear", scales=[{scales_str}]);'
            if output_size:
                sizes_str = ", ".join(str(s) for s in output_size)
                return f'[{output}] = resample2d({x}, mode="linear", sizes=[{sizes_str}]);'
        else:
            output_size = args[1] if len(args) > 1 else None
            scales_h = args[3] if len(args) > 3 else None  # arg[2] is align_corners
            scales_w = args[4] if len(args) > 4 else None
            if scales_h is not None and scales_w is not None:
                return f'[{output}] = resample2d({x}, mode="linear", scales=[{scales_h}, {scales_w}]);'
            if output_size:
                sizes_str = ", ".join(str(s) for s in output_size)
                return f'[{output}] = resample2d({x}, mode="linear", sizes=[{sizes_str}]);'

        raise NotImplementedError(f"upsample_bilinear2d: need output_size or scale_factors, got args={args}")

    # --- Softmax / attention ---

    def _convert_softmax(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.softmax.int(input, dim, half_to_float)"""
        x = inputs[0] if inputs else "unknown"
        axis = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        if axis == -1:
            axis = len(self._get_node_shape(node.args[0])) - 1
        return f"[{output}] = softmax({x}, axis={axis});"

    def _convert_softmax_aten(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten._softmax.default(input, dim, half_to_float)"""
        return self._convert_softmax(node, output, inputs)

    def _convert_scaled_dot_product_attention(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        if len(inputs) < 3:
            raise NotImplementedError("SDPA requires Q, K, V inputs")
        Q, K, V = inputs[0], inputs[1], inputs[2]
        q_shape = self._get_node_shape(node.args[0]) if isinstance(node.args[0], fx.Node) else []
        head_dim = q_shape[-1] if q_shape else 64
        scale = 1.0 / math.sqrt(head_dim)
        scale_c = self._create_inline_constant(scale)

        def tmp():
            name = f"operand_{self.operand_counter}"
            self.operand_counter += 1
            return name

        k_shape = self._get_node_shape(node.args[1]) if len(node.args) > 1 and isinstance(node.args[1], fx.Node) else []
        if k_shape and len(k_shape) >= 2:
            perm = list(range(len(k_shape)))
            perm[-2], perm[-1] = perm[-1], perm[-2]
        else:
            perm = [0, 1, 3, 2]
        perm_str = ", ".join(map(str, perm))

        kt, qk, qk_sc, attn_w = tmp(), tmp(), tmp(), tmp()
        axis = len(q_shape) - 1
        return "\n\t".join([
            f"[{kt}] = transpose({K}, permutation=[{perm_str}]);",
            f"[{qk}] = matmul({Q}, {kt});",
            f"[{qk_sc}] = mul({qk}, {scale_c});",
            f"[{attn_w}] = softmax({qk_sc}, axis={axis});",
            f"[{output}] = matmul({attn_w}, {V});",
        ])

    # --- Compile-time constant generation ---

    def _convert_arange(self, node: fx.Node, output: str, inputs: List[str]) -> Optional[str]:
        """aten.arange.* — evaluate at export time, embed as @bytes(...) constant.

        Overload args:
          arange.default(end)
          arange.start(start, end)
          arange.start_step(start, end, step)
        dtype / device come in via kwargs.
        """
        args = node.args
        target_str = str(node.target)
        if "start_step" in target_str:
            start, end, step = args[0], args[1], args[2]
        elif "start" in target_str:
            start, end = args[0], args[1]
            step = node.kwargs.get("step", 1)
        else:
            start, end, step = 0, args[0], 1

        dtype = node.kwargs.get("dtype", torch.float32) or torch.float32
        values = torch.arange(start, end, step, dtype=dtype)

        const_name = f"const_arange_{self.operand_counter}"
        self.operand_counter += 1
        self.inline_constants[const_name] = values
        self.operand_shapes[const_name] = list(values.shape)
        # Override so downstream ops reference the constant operand
        self.node_to_operand[node.name] = const_name
        return None  # no entry emitted in nodes {}

    def _convert_full(self, node: fx.Node, output: str, inputs: List[str]) -> Optional[str]:
        """aten.full.default(size, fill_value, *, dtype) — bake into @bytes."""
        args = node.args
        size = list(args[0]) if args else []
        fill_value = args[1] if len(args) > 1 else 0
        dtype = node.kwargs.get("dtype", torch.float32) or torch.float32
        values = torch.full(size, fill_value, dtype=dtype)
        const_name = f"const_full_{self.operand_counter}"
        self.operand_counter += 1
        self.inline_constants[const_name] = values
        self.operand_shapes[const_name] = list(values.shape)
        self.node_to_operand[node.name] = const_name
        return None

    def _convert_full_zeros(self, node: fx.Node, output: str, inputs: List[str]) -> Optional[str]:
        """aten.zeros.default — bake into @bytes."""
        size = list(node.args[0]) if node.args else []
        dtype = node.kwargs.get("dtype", torch.float32) or torch.float32
        values = torch.zeros(size, dtype=dtype)
        const_name = f"const_zeros_{self.operand_counter}"
        self.operand_counter += 1
        self.inline_constants[const_name] = values
        self.operand_shapes[const_name] = list(values.shape)
        self.node_to_operand[node.name] = const_name
        return None

    def _convert_full_ones(self, node: fx.Node, output: str, inputs: List[str]) -> Optional[str]:
        """aten.ones.default — bake into @bytes."""
        size = list(node.args[0]) if node.args else []
        dtype = node.kwargs.get("dtype", torch.float32) or torch.float32
        values = torch.ones(size, dtype=dtype)
        const_name = f"const_ones_{self.operand_counter}"
        self.operand_counter += 1
        self.inline_constants[const_name] = values
        self.operand_shapes[const_name] = list(values.shape)
        self.node_to_operand[node.name] = const_name
        return None

    # --- No-ops / cast ---

    def _convert_identity(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        if inputs:
            return f"[{output}] = identity({inputs[0]});"
        raise NotImplementedError("identity: no input")

    def _emit_cast(self, node: fx.Node, output: str, inputs: List[str], target_dtype: torch.dtype) -> str:
        """Emit cast or identity depending on whether the WebNN dtype actually changes."""
        x = inputs[0] if inputs else "unknown"
        input_node = node.args[0] if node.args and isinstance(node.args[0], fx.Node) else None
        src_dtype = self._get_node_dtype(input_node) if input_node else torch.float32
        src = self._get_webnn_dtype(src_dtype)
        tgt = self._get_webnn_dtype(target_dtype)
        if src == tgt:
            return f"[{output}] = identity({x});"
        return f"[{output}] = cast({x}, type={tgt});"

    def _convert_cast(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten._to_copy.default / aten.to.dtype — dtype in args[1] or kwargs['dtype']."""
        x = inputs[0] if inputs else "unknown"
        # aten.to.dtype: args = (input, dtype, ...)
        # aten._to_copy: dtype lives in kwargs
        target_dtype = None
        if len(node.args) > 1 and isinstance(node.args[1], torch.dtype):
            target_dtype = node.args[1]
        if target_dtype is None:
            target_dtype = node.kwargs.get("dtype")
        if not isinstance(target_dtype, torch.dtype):
            return f"[{output}] = identity({x});"
        return self._emit_cast(node, output, inputs, target_dtype)

    def _convert_to_device(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.to.device(input, device, dtype, ...) — ignore device, cast dtype."""
        x = inputs[0] if inputs else "unknown"
        # args = (input, device, dtype, non_blocking, copy, memory_format)
        target_dtype = node.args[2] if len(node.args) > 2 else None
        if not isinstance(target_dtype, torch.dtype):
            return f"[{output}] = identity({x});"
        return self._emit_cast(node, output, inputs, target_dtype)

    def _convert_type_as(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """aten.type_as.default(input, other) — cast input to dtype of other."""
        x = inputs[0] if inputs else "unknown"
        other_node = node.args[1] if len(node.args) > 1 and isinstance(node.args[1], fx.Node) else None
        if other_node is None:
            return f"[{output}] = identity({x});"
        target_dtype = self._get_node_dtype(other_node)
        return self._emit_cast(node, output, inputs, target_dtype)

    # ------------------------------------------------------------------
    # Output extraction
    # ------------------------------------------------------------------

    def _extract_outputs(self, gm: fx.GraphModule) -> str:
        outputs = []
        for node in gm.graph.nodes:
            if node.op == "output":
                flat = node.args[0]
                if isinstance(flat, (list, tuple)):
                    for arg in flat:
                        if isinstance(arg, fx.Node):
                            outputs.append(self._get_input_operand(arg))
                elif isinstance(flat, fx.Node):
                    outputs.append(self._get_input_operand(flat))
        return "; ".join(outputs) + ";" if outputs else ""

    # ------------------------------------------------------------------
    # Inline constant helpers
    # ------------------------------------------------------------------

    def _create_inline_constant(self, value) -> str:
        for name, v in self.inline_constants.items():
            if type(v) is type(value):
                if isinstance(value, torch.Tensor) and isinstance(v, torch.Tensor):
                    if torch.allclose(value, v):
                        return name
                elif v == value:
                    return name
        name = f"const_scalar_{self.operand_counter}"
        self.operand_counter += 1
        self.inline_constants[name] = value
        return name

    def _extract_inline_constants(self) -> str:
        consts = []
        for name, value in self.inline_constants.items():
            if isinstance(value, torch.Tensor):
                dtype = self._get_webnn_dtype(value.dtype)
                shape = list(value.shape)
                raw = value.cpu().numpy().tobytes()
                byte_list = ", ".join(str(b) for b in raw)
                consts.append(f"\t{name}: {dtype}[{', '.join(map(str, shape))}] @bytes([{byte_list}]);")
            elif isinstance(value, float):
                consts.append(f"\t{name}: f32[] @scalar({value});")
            elif isinstance(value, int):
                consts.append(f"\t{name}: i32[] @scalar({value});")
            else:
                consts.append(f"\t{name}: f32[] @scalar({value});")
        return "\n".join(consts) + "\n" if consts else ""

    # ------------------------------------------------------------------
    # Operand name management
    # ------------------------------------------------------------------

    def _get_operand_name(self, node: fx.Node) -> str:
        if node.name not in self.node_to_operand:
            self.node_to_operand[node.name] = f"operand_{self.operand_counter}"
            self.operand_counter += 1
        return self.node_to_operand[node.name]

    def _get_input_operand(self, node) -> str:
        if isinstance(node, fx.Node):
            # Parameter / buffer placeholder → look up in weight_operands by node name
            if node.op == "placeholder" and node.name in self.weight_operands:
                return self.weight_operands[node.name]
            if node.name in self.node_to_operand:
                return self.node_to_operand[node.name]
            return self._get_operand_name(node)
        return str(node)

    # ------------------------------------------------------------------
    # Shape / dtype helpers
    # ------------------------------------------------------------------

    def _get_node_shape(self, node) -> List[int]:
        if node is None or not hasattr(node, "meta"):
            return []
        meta = node.meta
        val = meta.get("val")
        if val is not None and hasattr(val, "shape"):
            return [int(d) for d in val.shape]
        tm = meta.get("tensor_meta")
        if tm is not None and hasattr(tm, "shape"):
            return [int(d) for d in tm.shape]
        return []

    def _get_node_dtype(self, node) -> torch.dtype:
        if not hasattr(node, "meta"):
            return torch.float32
        meta = node.meta
        val = meta.get("val")
        if val is not None and hasattr(val, "dtype"):
            return val.dtype
        tm = meta.get("tensor_meta")
        if tm is not None and hasattr(tm, "dtype"):
            return tm.dtype
        return torch.float32

    def _get_webnn_dtype(self, dtype) -> str:
        if not isinstance(dtype, torch.dtype):
            return "f32"
        return {
            torch.float32: "f32",
            torch.float16: "f16",
            torch.bfloat16: "f32",  # cast down to f32
            torch.int32: "i32",
            torch.int64: "i64",
            torch.int8: "i8",
            torch.uint8: "u8",
        }.get(dtype, "f32")

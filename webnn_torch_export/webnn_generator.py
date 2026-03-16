"""
WebNN Graph Generator - Converts PyTorch FX graphs to WebNN format
"""

import inspect
import torch
import torch.fx as fx
from typing import Dict, List, Tuple, Any
import math
import numpy as np

from jinja2.lexer import ignored_tokens
from sympy import true

from .webnn_op_mappings import resolve_pytorch_converter


def isnumeric(obj):
    try:
        obj + 0
        return True
    except TypeError:
        return False


def throw_unsupported(type, node, module_type="", module_class=""):
    error_msg = (
        f"\n{'=' * 80}\n"
        f"UNSUPPORTED {type}\n"
        f"{'=' * 80}\n"
    )
    if module_type:
        error_msg += (
            f"{type} Type: {module_type}\n"
        )
    if module_class:
        error_msg += (
            f"{type} Class: {module_class}\n"
        )
    error_msg += (
        f"Node: {node.name}\n"
        f"Target: {node.target}\n"
        f"Args: {[str(arg) for arg in node.args]}\n"
        f"Kwargs: {node.kwargs}\n"
        f"{'=' * 80}\n"
    )
    raise NotImplementedError(error_msg)


IGNORED_PLACEHOLDER_TOKENS = {'modules', 'buffers', 'parameters'}


class WebNNGraphGenerator:
    """Generates WebNN graph format from PyTorch FX GraphModule"""

    def __init__(self):
        self.operand_counter = 1
        self.node_to_operand = {}
        self.weight_operands = {}
        self.operand_shapes = {}
        self.inline_constants = {}  # Store inline constants like scalars (embedded in .webnn file)

    def generate(self, gm: fx.GraphModule, model: torch.nn.Module, graph_name: str = "model") -> str:
        """
        Generate WebNN graph format from FX graph

        Args:
            gm: FX GraphModule from Dynamo
            model: Original PyTorch model for weight extraction
            graph_name: Name for the WebNN graph

        Returns:
            WebNN graph as string
        """
        self.operand_counter = 1
        self.node_to_operand = {}
        self.weight_operands = {}
        self.operand_shapes = {}
        self.inline_constants = {}  # Reset inline constants for each graph

        # Extract forward() parameter order so inputs are emitted in the right order
        try:
            sig = inspect.signature(model.forward)
            param_names = [p for p in sig.parameters if p != 'self']
        except (ValueError, TypeError):
            param_names = []

        # Extract sections
        inputs_section = self._extract_inputs(gm, param_names)
        consts_section, weight_map = self._extract_weights(model)
        nodes_section = self._convert_nodes(gm)
        inline_consts_section = self._extract_inline_constants()
        outputs_section = self._extract_outputs(gm)

        # Combine all constants: inline (scalars), generated (arange), and weights
        all_consts = ''
        if inline_consts_section:
            all_consts += inline_consts_section
        if consts_section:
            all_consts += consts_section

        # Build WebNN graph
        graph = f'webnn_graph "{graph_name}" v1 {{\n'
        graph += f'  inputs {{ {inputs_section} }}\n'
        if all_consts:
            graph += f'  consts {{\n{all_consts}  }}\n'
        graph += f'  nodes {{\n{nodes_section}  }}\n'
        graph += f'  outputs {{ {outputs_section} }}\n'
        graph += '}\n'

        return graph

    def _extract_inputs(self, gm: fx.GraphModule, param_names: List[str] = None) -> str:
        """Extract input tensor declarations (only actual model inputs, not weights).

        param_names: ordered list of parameter names from model.forward() signature.
        When provided, the inputs section is emitted in that order so the executor
        maps positional arguments correctly.
        """
        inputs = []  # list of (sort_key, declaration)
        for node in gm.graph.nodes:
            if node.op == 'placeholder':
                # Skip weight/parameter placeholders - only include actual inputs
                node_name = str(node.name)
                if set(node_name.split("_")).intersection(IGNORED_PLACEHOLDER_TOKENS):
                    continue

                if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                    tensor = node.meta['tensor_meta']
                    shape = list(tensor.shape)
                    dtype = self._get_webnn_dtype(tensor.dtype)
                    shape_str = ', '.join(map(str, shape))
                    name = node.name
                    if name.startswith('l_'):
                        name = name[len('l_'):]
                    name = name.rstrip("_")
                    self.node_to_operand[node.name] = name
                    sort_key = param_names.index(name) if (param_names and name in param_names) else len(inputs)
                    inputs.append((sort_key, f'{name}: {dtype}[{shape_str}]'))
                else:
                    raise NotImplementedError(f"Dynamic inputs are not supported: {node.name} does not have tensor_meta")

        inputs.sort(key=lambda x: x[0])
        decls = [decl for _, decl in inputs]
        return '; '.join(decls) + ';' if decls else ''

    def _extract_weights(self, model: torch.nn.Module) -> Tuple[str, Dict[str, str]]:
        """Extract weight constants and create mapping"""
        consts = []
        weight_map = {}

        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            operand_name = f'operand_{self.operand_counter}'
            self.operand_counter += 1

            shape = list(tensor.shape)
            dtype = self._get_webnn_dtype(tensor.dtype)
            shape_str = ', '.join(map(str, shape))

            consts.append(f'    {operand_name}: {dtype}[{shape_str}] @weights("{name}");')
            weight_map[name] = operand_name
            self.weight_operands[name] = operand_name
            self.operand_shapes[operand_name] = shape

        return '\n'.join(consts) + '\n' if consts else '', weight_map

    def _convert_nodes(self, gm: fx.GraphModule) -> str:
        """Convert FX nodes to WebNN operations"""
        operations = []

        for i, node in enumerate(gm.graph.nodes):
            if node.op == 'call_function':
                op_str = self._map_pytorch_to_webnn_op(node)
                if op_str:
                    operations.append(f'    {op_str}')
            elif node.op == 'call_module':
                op_str = self._map_module_to_webnn_op(node, gm)
                if op_str:
                    operations.append(f'    {op_str}')
            elif node.op == 'call_method':
                op_str = self._map_method_to_webnn_op(node)
                if op_str:
                    operations.append(f'    {op_str}')

        return '\n'.join(operations) + '\n' if operations else ''

    def _map_pytorch_to_webnn_op(self, node: fx.Node) -> str:
        """Map PyTorch function to WebNN operation"""
        # Get output operand
        output_operand = self._get_operand_name(node)

        # Get input operands
        input_operands = [self._get_input_operand(arg) for arg in node.args if isinstance(arg, fx.Node)]

        converter = resolve_pytorch_converter(node.target)
        if converter:
            return converter(self, node, output_operand, input_operands)

        # Raise error for unsupported operations
        target_str = str(node.target)
        target_name = getattr(node.target, "__name__", target_str)
        schema = getattr(node.target, "_schema", None)
        schema_str = str(schema) if schema else "N/A"
        throw_unsupported("Operation", node)

    def _map_method_to_webnn_op(self, node: fx.Node) -> str:
        """Map PyTorch function to WebNN operation"""
        # Get output operand
        output_operand = self._get_operand_name(node)

        # Get input operands
        input_operands = [self._get_input_operand(arg) for arg in node.args if isinstance(arg, fx.Node)]

        converter = resolve_pytorch_converter(node.target)
        if converter:
            return converter(self, node, output_operand, input_operands)

        # Raise error for unsupported operations
        target_str = str(node.target)
        target_name = getattr(node.target, "__name__", target_str)
        schema = getattr(node.target, "_schema", None)
        schema_str = str(schema) if schema else "N/A"
        throw_unsupported("Method", node)

    def _map_module_to_webnn_op(self, node: fx.Node, gm: fx.GraphModule) -> str:
        """Map PyTorch module call to WebNN operation"""
        module = self._get_module(gm, node.target)
        output_operand = self._get_operand_name(node)
        input_operands = [self._get_input_operand(arg) for arg in node.args if isinstance(arg, fx.Node)]

        if isinstance(module, torch.nn.Conv2d):
            return self._convert_conv2d_module(node, module, output_operand, input_operands)
        elif isinstance(module, torch.nn.ReLU):
            return f'[{output_operand}] = clamp({input_operands[0]}, minValue=0.0);'
        elif isinstance(module, torch.nn.Linear):
            return self._convert_linear_module(node, module, output_operand, input_operands)
        else:
            # Raise error for unsupported modules
            module_type = type(module).__name__
            module_class = f"{type(module).__module__}.{type(module).__name__}"
            throw_unsupported("Module", module_type, module_class, node)

    def _emit_conv2d(self, input_tensor: str, weight: str, bias_info, stride, padding, dilation, groups, output: str) -> str:
        """Emit WebNN conv2d nodes, including bias reshape+add when bias is present.

        bias_info: (bias_operand, num_channels) when a bias exists, else None.
        stride/padding/dilation accept both lists and tuples.
        """

        def as_pair(v):
            return list(v) if isinstance(v, (list, tuple)) else [v, v]

        stride = as_pair(stride)
        padding = as_pair(padding)
        dilation = as_pair(dilation)

        params = []
        if dilation != [1, 1]:
            params.append(f'dilations=[{dilation[0]}, {dilation[1]}]')
        params.append('filterLayout="oihw"')
        params.append(f'groups={groups}')
        params.append('inputLayout="nchw"')
        if padding != [0, 0]:
            params.append(f'pads=[{padding[0]}, {padding[0]}, {padding[1]}, {padding[1]}]')
        if stride != [1, 1]:
            params.append(f'strides=[{stride[0]}, {stride[1]}]')

        params_str = ', '.join(params)

        if bias_info is not None:
            bias_operand, c = bias_info
            # Reshape bias [C] → [1, C, 1, 1] for NCHW broadcast
            reshaped_bias = f'operand_{self.operand_counter}'
            self.operand_counter += 1
            conv_out = f'operand_{self.operand_counter}'
            self.operand_counter += 1
            return (
                f'[{reshaped_bias}] = reshape({bias_operand}, newShape=[1, {c}, 1, 1]);\n'
                f'    [{conv_out}] = conv2d({input_tensor}, {weight}, {params_str});\n'
                f'    [{output}] = add({conv_out}, {reshaped_bias});'
            )
        return f'[{output}] = conv2d({input_tensor}, {weight}, {params_str});'

    def _convert_conv2d(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert torch.conv2d (call_function) to WebNN."""
        # torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
        args = node.args
        input_tensor = inputs[0] if inputs else 'unknown'
        weight = self._get_input_operand(args[1]) if len(args) > 1 else 'unknown'

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

    def _convert_conv2d_module(self, node: fx.Node, module: torch.nn.Conv2d, output: str, inputs: List[str]) -> str:
        """Convert Conv2d (call_module) to WebNN — delegates to _emit_conv2d."""
        input_tensor = inputs[0] if inputs else 'unknown'
        weight = self.weight_operands.get(f'{node.target}.weight', 'unknown')

        bias_info = None
        if module.bias is not None:
            bias_operand = self.weight_operands.get(f'{node.target}.bias')
            if bias_operand:
                bias_shape = self.operand_shapes.get(bias_operand, [])
                bias_info = (bias_operand, bias_shape[0] if bias_shape else module.out_channels)

        return self._emit_conv2d(input_tensor, weight, bias_info, module.stride, module.padding, module.dilation, module.groups, output)

    def _convert_arithmetric(self, node: fx.Node, output: str, inputs: List[str], op: str) -> str:
        """Convert arithmetic operations (add, sub, mul, div) to WebNN"""
        if len(inputs) == 2:
            return f'[{output}] = {op}({inputs[0]}, {inputs[1]});'
        if len(node.args) == 2:
            if len(inputs) == 1 and isnumeric(node.args[1]):
                # Create an inline constant for the numeric value
                const_operand = self._create_inline_constant(node.args[1])
                return f'[{output}] = {op}({inputs[0]}, {const_operand});'
            elif len(inputs) == 1 and isnumeric(node.args[0]):
                # Create an inline constant for the numeric value
                const_operand = self._create_inline_constant(node.args[0])
                return f'[{output}] = {op}({inputs[0]}, {const_operand});'
        return f'// Invalid {op} operation'

    def _convert_math(self, node: fx.Node, output: str, inputs: List[str], op: str) -> str:
        """Convert math functions (sqrt, exp, log, cos, sin) to WebNN"""
        input_tensor = inputs[0] if inputs else 'unknown'
        return f'[{output}] = {op}({input_tensor});'

    def _convert_pow(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert power to WebNN pow"""
        if len(inputs) >= 2:
            return f'[{output}] = pow({inputs[0]}, {inputs[1]});'
        return f'// Invalid pow operation'

    def _convert_neg(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert negation to WebNN neg"""
        input_tensor = inputs[0] if inputs else 'unknown'
        return f'[{output}] = neg({input_tensor});'

    def _convert_cast(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert type casting (tensor.to) to WebNN cast"""
        input_tensor = inputs[0] if inputs else 'unknown'

        # Get target dtype from args
        # to(dtype) or to(device, dtype) or to(tensor, dtype, ...)
        target_dtype = None
        for arg in node.args[1:]:  # Skip first arg (input tensor)
            if isinstance(arg, torch.dtype):
                target_dtype = arg
                break

        # Also check kwargs
        if target_dtype is None and 'dtype' in node.kwargs:
            target_dtype = node.kwargs['dtype']

        # If no dtype found, just return identity (might be device-only cast)
        if target_dtype is None:
            return f'[{output}] = identity({input_tensor});'

        # Map PyTorch dtype to WebNN dtype
        webnn_dtype = self._get_webnn_dtype(target_dtype)

        return f'[{output}] = cast({input_tensor}, type={webnn_dtype});'

    def _convert_sigmoid(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert sigmoid to WebNN sigmoid"""
        input_tensor = inputs[0] if inputs else 'unknown'
        return f'[{output}] = sigmoid({input_tensor});'

    def _convert_tanh(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert tanh to WebNN tanh"""
        input_tensor = inputs[0] if inputs else 'unknown'
        return f'[{output}] = tanh({input_tensor});'

    def _convert_softmax(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert softmax to WebNN softmax"""
        input_tensor = inputs[0] if inputs else 'unknown'
        # Get axis from kwargs or args
        axis = node.kwargs.get('dim', -1)
        if len(node.args) > 1:
            axis = node.args[1]
        return f'[{output}] = softmax({input_tensor}, axis={axis});'

    def _convert_silu(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """
        Convert SiLU (Swish) activation to WebNN operations.

        SiLU(x) = x * sigmoid(x)

        This is a common activation function in modern neural networks,
        especially in diffusion models and transformers.
        """
        input_tensor = inputs[0] if inputs else 'unknown'

        # Step 1: Compute sigmoid(x)
        sigmoid_operand = f'operand_{self.operand_counter}'
        self.operand_counter += 1
        step1 = f'[{sigmoid_operand}] = sigmoid({input_tensor});'

        # Step 2: Multiply x * sigmoid(x)
        step2 = f'[{output}] = mul({input_tensor}, {sigmoid_operand});'

        return f'{step1}\n    {step2}'

    def _convert_layer_norm(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert layer normalization to WebNN layerNormalization"""
        input_tensor = inputs[0] if inputs else 'unknown'

        # Extract parameters from args
        args = node.args
        # layer_norm(input, normalized_shape, weight, bias, eps)
        weight = self._get_input_operand(args[2]) if len(args) > 2 and isinstance(args[2], fx.Node) else None
        bias = self._get_input_operand(args[3]) if len(args) > 3 and isinstance(args[3], fx.Node) else None
        eps = args[4] if len(args) > 4 else 1e-5

        params = []
        if weight:
            params.append(f'scale={weight}')
        if bias:
            params.append(f'bias={bias}')
        params.append(f'epsilon={eps}')

        params_str = ', '.join(params)
        return f'[{output}] = layerNormalization({input_tensor}, {params_str});'

    def _convert_group_norm(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """
        Convert group normalization to WebNN operations.

        Group normalization divides channels into groups and normalizes within each group.
        This is commonly used in diffusion models and other architectures.

        Since WebNN doesn't have a native groupNorm, we decompose it into:
        1. Reshape to separate groups
        2. Normalize per group
        3. Reshape back
        4. Apply affine transform
        """
        input_tensor = inputs[0] if inputs else 'unknown'

        # Extract parameters from args
        # group_norm(input, num_groups, weight, bias, eps)
        args = node.args
        num_groups = args[1] if len(args) > 1 else 32
        weight = self._get_input_operand(args[2]) if len(args) > 2 and isinstance(args[2], fx.Node) else None
        bias = self._get_input_operand(args[3]) if len(args) > 3 and isinstance(args[3], fx.Node) else None
        eps = args[4] if len(args) > 4 else 1e-5

        # Get input shape to understand dimensions
        input_shape = self._get_node_shape(node.args[0]) if node.args and isinstance(args[0], fx.Node) else []

        if not input_shape or len(input_shape) < 2:
            # Fallback: use layerNormalization as approximation
            # This is not exactly group norm but works for many cases
            params = []
            if weight:
                params.append(f'scale={weight}')
            if bias:
                params.append(f'bias={bias}')
            params.append(f'epsilon={eps}')
            params_str = ', '.join(params)
            return f'[{output}] = layerNormalization({input_tensor}, {params_str}); // approximation of group_norm'

        # Full decomposition for group normalization
        # Input shape: [N, C, *spatial]
        N = input_shape[0]
        C = input_shape[1]
        spatial_dims = input_shape[2:]

        # Calculate group parameters
        channels_per_group = C // num_groups

        # Step 1: Reshape to separate groups
        # [N, C, *spatial] -> [N, num_groups, channels_per_group, *spatial]
        reshape1 = f'operand_{self.operand_counter}'
        self.operand_counter += 1
        reshape1_shape = [N, num_groups, channels_per_group] + spatial_dims
        reshape1_str = ', '.join(map(str, reshape1_shape))
        step1 = f'[{reshape1}] = reshape({input_tensor}, newShape=[{reshape1_str}]);'

        # Step 2: Compute mean per group (reduce over channels_per_group and spatial dims)
        # This requires reduceMean over specific axes
        # Axes to reduce: [2, 3, 4, ...] (channels_per_group and all spatial dimensions)
        mean_axes = list(range(2, 2 + 1 + len(spatial_dims)))
        mean_operand = f'operand_{self.operand_counter}'
        self.operand_counter += 1
        axes_str = ', '.join(map(str, mean_axes))
        step2 = f'[{mean_operand}] = reduceMean({reshape1}, axes=[{axes_str}], keepDimensions=true);'

        # Step 3: Subtract mean
        centered = f'operand_{self.operand_counter}'
        self.operand_counter += 1
        step3 = f'[{centered}] = sub({reshape1}, {mean_operand});'

        # Step 4: Compute variance (mean of squared differences)
        squared = f'operand_{self.operand_counter}'
        self.operand_counter += 1
        step4a = f'[{squared}] = mul({centered}, {centered});'

        var_operand = f'operand_{self.operand_counter}'
        self.operand_counter += 1
        step4b = f'[{var_operand}] = reduceMean({squared}, axes=[{axes_str}], keepDimensions=true);'

        # Step 5: Compute std = sqrt(var + eps)
        var_eps = f'operand_{self.operand_counter}'
        self.operand_counter += 1

        # Create epsilon constant
        eps_const = f'const_eps_{self.operand_counter}'
        self.operand_counter += 1

        eps_tensor = torch.tensor(eps, dtype=torch.float32)
        self.inline_constants[eps_const] = eps_tensor

        step5a = f'[{var_eps}] = add({var_operand}, {eps_const});'

        std_operand = f'operand_{self.operand_counter}'
        self.operand_counter += 1
        step5b = f'[{std_operand}] = sqrt({var_eps});'

        # Step 6: Normalize: centered / std
        normalized = f'operand_{self.operand_counter}'
        self.operand_counter += 1
        step6 = f'[{normalized}] = div({centered}, {std_operand});'

        # Step 7: Reshape back to original shape
        reshaped_back = f'operand_{self.operand_counter}'
        self.operand_counter += 1
        orig_shape_str = ', '.join(map(str, input_shape))
        step7 = f'[{reshaped_back}] = reshape({normalized}, newShape=[{orig_shape_str}]);'

        # Step 8: Apply affine transform if weight/bias provided
        result = reshaped_back
        steps = [step1, step2, step3, step4a, step4b, step5a, step5b, step6, step7]

        if weight:
            scaled = f'operand_{self.operand_counter}'
            self.operand_counter += 1
            step8 = f'[{scaled}] = mul({result}, {weight});'
            steps.append(step8)
            result = scaled

        if bias:
            step9 = f'[{output}] = add({result}, {bias});'
            steps.append(step9)
        else:
            # Rename final result to output
            steps.append(f'[{output}] = identity({result});')

        return '\n    '.join(steps)

    def _convert_getitem(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """
        Convert Python's getitem (indexing/slicing) to WebNN operations.

        Common patterns:
        - tensor[:, None] -> unsqueeze
        - tensor[0] -> slice
        - tensor[:, 1:10] -> slice
        - tensor[..., 0] -> slice
        """
        input_tensor = inputs[0] if inputs else 'unknown'

        # Get the index/slice from args
        if len(node.args) < 2:
            return f'[{output}] = identity({input_tensor}); // getitem with no index'

        index = node.args[1]

        # Get input shape to help with dimension calculations
        input_shape = self._get_node_shape(node.args[0]) if node.args and isinstance(node.args[0], fx.Node) else []

        # Handle single None (add dimension)
        if index is None:
            # tensor[None] -> unsqueeze at dimension 0
            # WebNN uses reshape to add dimensions
            if input_shape:
                new_shape = [1] + input_shape
                shape_str = ', '.join(map(str, new_shape))
                return f'[{output}] = reshape({input_tensor}, newShape=[{shape_str}]);'
            else:
                return f'[{output}] = reshape({input_tensor}, newShape=[1, ...]);'

        # Handle tuple of indices/slices
        if isinstance(index, tuple):
            # Check for patterns like (:, None) which adds a dimension
            none_positions = [i for i, idx in enumerate(index) if idx is None]
            slice_positions = [i for i, idx in enumerate(index) if isinstance(idx, slice)]

            if none_positions and all(isinstance(idx, (slice, type(None))) for idx in index):
                # This is unsqueeze operation - adding dimensions
                # Example: tensor[:, None] adds dimension at position 1
                # Example: tensor[:, None, :] adds dimension at position 1

                if not input_shape:
                    return f'[{output}] = identity({input_tensor});'

                # Build new shape by inserting 1s at None positions
                output_shape = []
                input_dim = 0
                for i, idx in enumerate(index):
                    if idx is None:
                        output_shape.append(1)
                    elif isinstance(idx, slice):
                        if idx == slice(None, None, None):  # Full slice (:)
                            if input_dim < len(input_shape):
                                output_shape.append(input_shape[input_dim])
                                input_dim += 1
                        else:
                            # Partial slice - need to calculate size
                            if input_dim < len(input_shape):
                                dim_size = input_shape[input_dim]
                                start = idx.start if idx.start is not None else 0
                                stop = idx.stop if idx.stop is not None else dim_size
                                step = idx.step if idx.step is not None else 1
                                sliced_size = (stop - start) // step
                                output_shape.append(sliced_size)
                                input_dim += 1
                    elif isinstance(idx, int):
                        # Integer indexing removes the dimension
                        input_dim += 1
                        # Don't add to output_shape (dimension is removed)

                # Add remaining dimensions
                while input_dim < len(input_shape):
                    output_shape.append(input_shape[input_dim])
                    input_dim += 1

                shape_str = ', '.join(map(str, output_shape))
                return f'[{output}] = reshape({input_tensor}, newShape=[{shape_str}]);'

            # Handle pure slicing (no None)
            elif not none_positions and all(isinstance(idx, (slice, int)) for idx in index):
                # This is slice operation
                # Example: tensor[0, :, 1:10]

                if not input_shape:
                    return f'[{output}] = identity({input_tensor});'

                # Check if it's just integer indexing (removes dimensions)
                if all(isinstance(idx, int) for idx in index):
                    # All integer indices - results in a scalar or lower-rank tensor
                    # This requires gather operation which we may not have yet
                    return f'// TODO: getitem with all integer indices (gather operation needed)'

                # Mixed slice and integer indexing
                # Build starts, sizes, and output shape
                starts = []
                sizes = []
                output_shape = []
                for dim_idx, idx in enumerate(index):
                    if dim_idx >= len(input_shape):
                        break

                    dim_size = input_shape[dim_idx]

                    if isinstance(idx, slice):
                        start = idx.start if idx.start is not None else 0
                        stop = idx.stop if idx.stop is not None else dim_size
                        step = idx.step if idx.step is not None else 1

                        if step != 1:
                            return f'// TODO: getitem with step != 1 not supported yet'

                        starts.append(start)
                        size = stop - start
                        sizes.append(size)
                        output_shape.append(size)
                    elif isinstance(idx, int):
                        # Integer index - take single element
                        starts.append(idx)
                        sizes.append(1)
                        # Don't add to output_shape (dimension is squeezed)

                # Handle remaining dimensions (implicitly [:])
                for dim_idx in range(len(index), len(input_shape)):
                    starts.append(0)
                    sizes.append(input_shape[dim_idx])
                    output_shape.append(input_shape[dim_idx])

                # Generate slice operation
                starts_str = ', '.join(map(str, starts))
                sizes_str = ', '.join(map(str, sizes))

                sliced = f'operand_{self.operand_counter}'
                self.operand_counter += 1
                slice_op = f'[{sliced}] = slice({input_tensor}, starts=[{starts_str}], sizes=[{sizes_str}]);'

                # If we removed dimensions (integer indexing), reshape to squeeze them
                if len(output_shape) < len(sizes):
                    output_shape_str = ', '.join(map(str, output_shape))
                    reshape_op = f'[{output}] = reshape({sliced}, newShape=[{output_shape_str}]);'
                    return f'{slice_op}\n    {reshape_op}'
                else:
                    return f'[{output}] = slice({input_tensor}, starts=[{starts_str}], sizes=[{sizes_str}]);'

        # Handle single integer index (e.g., tensor[0])
        elif isinstance(index, int):
            if not input_shape:
                return f'[{output}] = identity({input_tensor});'

            # Slice first dimension at index
            starts = [index] + [0] * (len(input_shape) - 1)
            sizes = [1] + input_shape[1:]

            starts_str = ', '.join(map(str, starts))
            sizes_str = ', '.join(map(str, sizes))

            # Slice and then reshape to remove the first dimension
            sliced = f'operand_{self.operand_counter}'
            self.operand_counter += 1
            slice_op = f'[{sliced}] = slice({input_tensor}, starts=[{starts_str}], sizes=[{sizes_str}]);'

            output_shape = input_shape[1:]
            if output_shape:
                output_shape_str = ', '.join(map(str, output_shape))
                reshape_op = f'[{output}] = reshape({sliced}, newShape=[{output_shape_str}]);'
                return f'{slice_op}\n    {reshape_op}'
            else:
                return slice_op

        # Handle single slice (e.g., tensor[1:10])
        elif isinstance(index, slice):
            if not input_shape:
                return f'[{output}] = identity({input_tensor});'

            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else input_shape[0]
            step = index.step if index.step is not None else 1

            if step != 1:
                return f'// TODO: getitem with step != 1 not supported yet'

            # Build full starts and sizes for all dimensions
            starts = [start] + [0] * (len(input_shape) - 1)
            sizes = [stop - start] + input_shape[1:]

            starts_str = ', '.join(map(str, starts))
            sizes_str = ', '.join(map(str, sizes))

            return f'[{output}] = slice({input_tensor}, starts=[{starts_str}], sizes=[{sizes_str}]);'

        # Unknown index type
        return f'// TODO: getitem with index type {type(index).__name__} not supported yet'

    def _convert_rearrange(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """
        Convert einops rearrange to WebNN operations.

        Rearrange is a powerful operation from einops that can express many transformations.
        This implementation handles common patterns used in vision models.
        """
        raise NotImplementedError("Rearrange is not yet implemented.")

    def _convert_arange(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """
        Convert arange to a pre-computed constant.
        Since WebNN doesn't have arange, we compute it at export time and add as a constant.
        """
        # Extract parameters from kwargs
        kwargs = node.kwargs
        start = kwargs.get('start', 0)
        end = kwargs.get('end', None)
        step = kwargs.get('step', 1)
        dtype = kwargs.get('dtype', torch.float32)

        # Handle positional args if no kwargs
        if end is None and node.args:
            if len(node.args) == 1:
                end = node.args[0]
            elif len(node.args) >= 2:
                start = node.args[0]
                end = node.args[1]
            if len(node.args) >= 3:
                step = node.args[2]

        if end is None:
            raise ValueError(f"arange requires 'end' parameter: {node}")

        # Generate arange values
        values = torch.arange(start, end, step, dtype=dtype)

        # Store as a generated constant
        const_name = f'const_arange_{self.operand_counter}'
        self.operand_counter += 1

        self.inline_constants[const_name] = values
        self.operand_shapes[const_name] = list(values.shape)

        # Map this node to the constant operand
        self.node_to_operand[node.name] = const_name

    def _convert_einsum(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """
        Convert einsum (Einstein summation) to WebNN operations.

        Einsum is a powerful operation that can express many tensor operations
        through Einstein notation. Common patterns:
        - Matrix multiplication: 'ij,jk->ik'
        - Batch matrix multiply: 'bij,bjk->bik'
        - Outer product: 'i,j->ij'
        - Broadcasting: '...n,d->...nd'

        This implementation handles common patterns. Complex patterns may need
        additional decomposition.
        """
        # Get einsum pattern from args
        # einsum(pattern, *tensors)
        args = node.args
        if not args:
            return f'// Invalid einsum: no arguments'

        pattern = args[0] if isinstance(args[0], str) else None
        if not pattern:
            return f'// Invalid einsum: pattern not found'

        # Get input shapes
        input_shapes = []
        for i, inp in enumerate(inputs):
            if i < len(args) - 1:  # Skip the pattern
                node_arg = args[i + 1]
                if isinstance(node_arg, fx.Node):
                    shape = self._get_node_shape(node_arg)
                    input_shapes.append(shape)

        # Parse the einsum pattern
        # Pattern format: 'input1,input2,...->output'
        if '->' not in pattern:
            return f'// Unsupported einsum pattern (no ->): {pattern}'

        lhs, rhs = pattern.split('->')
        input_patterns = [p.strip() for p in lhs.split(',')]

        # Handle common patterns
        # Pattern: '...n,d->...nd' (outer product with broadcasting)
        if len(input_patterns) == 2 and pattern.endswith('nd') and input_patterns[0].endswith('n') and input_patterns[1] == 'd':
            # This is: [..., n] x [d] -> [..., n, d]
            # Can be implemented as unsqueeze + broadcast
            input1 = inputs[0] if len(inputs) > 0 else 'unknown'
            input2 = inputs[1] if len(inputs) > 1 else 'unknown'

            if not input_shapes or len(input_shapes) < 2:
                return f'// einsum {pattern}: shape information needed'

            shape1 = input_shapes[0]
            shape2 = input_shapes[1]

            # Step 1: Unsqueeze input2 to add a dimension
            # [d] -> [1, d]
            unsqueezed2 = f'operand_{self.operand_counter}'
            self.operand_counter += 1
            unsqueeze_shape = [1] + shape2
            unsqueeze_str = ', '.join(map(str, unsqueeze_shape))
            step1 = f'[{unsqueezed2}] = reshape({input2}, newShape=[{unsqueeze_str}]);'

            # Step 2: Unsqueeze input1 to add dimension for broadcasting
            # [..., n] -> [..., n, 1]
            unsqueezed1 = f'operand_{self.operand_counter}'
            self.operand_counter += 1
            unsqueeze1_shape = shape1 + [1]
            unsqueeze1_str = ', '.join(map(str, unsqueeze1_shape))
            step2 = f'[{unsqueezed1}] = reshape({input1}, newShape=[{unsqueeze1_str}]);'

            # Step 3: Multiply (will broadcast automatically)
            # [..., n, 1] * [1, d] -> [..., n, d]
            step3 = f'[{output}] = mul({unsqueezed1}, {unsqueezed2});'

            return f'{step1}\n    {step2}\n    {step3}'

        # Pattern: 'ij,jk->ik' (matrix multiplication)
        elif len(input_patterns) == 2 and len(input_patterns[0]) == 2 and len(input_patterns[1]) == 2 and len(rhs) == 2:
            # Standard matrix multiplication
            input1 = inputs[0] if len(inputs) > 0 else 'unknown'
            input2 = inputs[1] if len(inputs) > 1 else 'unknown'
            return f'[{output}] = matmul({input1}, {input2});'

        # Pattern: 'bij,bjk->bik' (batch matrix multiplication)
        elif len(input_patterns) == 2 and len(input_patterns[0]) == 3 and len(input_patterns[1]) == 3 and len(rhs) == 3:
            # Batch matrix multiplication
            input1 = inputs[0] if len(inputs) > 0 else 'unknown'
            input2 = inputs[1] if len(inputs) > 1 else 'unknown'
            return f'[{output}] = matmul({input1}, {input2});'

        # Pattern: 'i,j->ij' (outer product)
        elif len(input_patterns) == 2 and len(input_patterns[0]) == 1 and len(input_patterns[1]) == 1 and len(rhs) == 2:
            input1 = inputs[0] if len(inputs) > 0 else 'unknown'
            input2 = inputs[1] if len(inputs) > 1 else 'unknown'

            # Outer product: reshape to [n, 1] and [1, m], then multiply
            if not input_shapes or len(input_shapes) < 2:
                return f'// einsum {pattern}: shape information needed'

            shape1 = input_shapes[0]
            shape2 = input_shapes[1]

            # Reshape input1 to [n, 1]
            reshaped1 = f'operand_{self.operand_counter}'
            self.operand_counter += 1
            reshape1_shape = shape1 + [1]
            reshape1_str = ', '.join(map(str, reshape1_shape))
            step1 = f'[{reshaped1}] = reshape({input1}, newShape=[{reshape1_str}]);'

            # Reshape input2 to [1, m]
            reshaped2 = f'operand_{self.operand_counter}'
            self.operand_counter += 1
            reshape2_shape = [1] + shape2
            reshape2_str = ', '.join(map(str, reshape2_shape))
            step2 = f'[{reshaped2}] = reshape({input2}, newShape=[{reshape2_str}]);'

            # Multiply
            step3 = f'[{output}] = mul({reshaped1}, {reshaped2});'

            return f'{step1}\n    {step2}\n    {step3}'

        # Pattern: 'ii->i' (diagonal extraction)
        elif len(input_patterns) == 1 and len(input_patterns[0]) == 2 and len(rhs) == 1:
            # Diagonal extraction - not directly supported, would need gather
            return f'// TODO: einsum diagonal extraction pattern ({pattern}) not supported yet'

        # Pattern: 'ij->ji' (transpose)
        elif len(input_patterns) == 1 and len(input_patterns[0]) == 2 and len(rhs) == 2:
            # Check if it's a transpose
            if input_patterns[0][0] == rhs[1] and input_patterns[0][1] == rhs[0]:
                input1 = inputs[0] if len(inputs) > 0 else 'unknown'
                return f'[{output}] = transpose({input1}, permutation=[1, 0]);'

        # Unknown pattern
        return f'// TODO: einsum pattern not yet supported: {pattern}'

    def _convert_scaled_dot_product_attention(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """
        Convert scaled dot product attention to WebNN operations.

        Implements: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

        This is the core attention mechanism used in transformers and
        was introduced as an optimized primitive in PyTorch 2.0.

        Args:
            Q: Query tensor of shape [batch, num_heads, seq_len_q, head_dim]
            K: Key tensor of shape [batch, num_heads, seq_len_k, head_dim]
            V: Value tensor of shape [batch, num_heads, seq_len_v, head_dim]

        Returns:
            Output tensor of shape [batch, num_heads, seq_len_q, head_dim]
        """
        if len(inputs) < 3:
            return f'// Invalid scaled_dot_product_attention: need Q, K, V inputs'

        Q = inputs[0]
        K = inputs[1]
        V = inputs[2]

        # Get shapes to calculate scaling factor
        q_shape = self._get_node_shape(node.args[0]) if len(node.args) > 0 and isinstance(node.args[0], fx.Node) else []

        # head_dim is the last dimension of Q
        head_dim = q_shape[-1] if q_shape else 64  # default to 64 if unknown

        # Calculate scaling factor: 1 / sqrt(head_dim)
        import math
        scale_factor = 1.0 / math.sqrt(head_dim)

        steps = []

        # Step 1: Transpose K to get K^T
        # K shape: [batch, num_heads, seq_len_k, head_dim]
        # K^T shape: [batch, num_heads, head_dim, seq_len_k]
        # We transpose the last two dimensions: [... -2, -1] -> [... -1, -2]
        k_transposed = f'operand_{self.operand_counter}'
        self.operand_counter += 1

        if len(node.args) > 1 and isinstance(node.args[1], fx.Node):
            k_shape = self._get_node_shape(node.args[1])
            if k_shape and len(k_shape) >= 2:
                # Build permutation for transpose
                # For 4D: [0, 1, 2, 3] -> [0, 1, 3, 2]
                perm = list(range(len(k_shape)))
                perm[-2], perm[-1] = perm[-1], perm[-2]
                perm_str = ', '.join(map(str, perm))
                steps.append(f'[{k_transposed}] = transpose({K}, permutation=[{perm_str}]);')
            else:
                steps.append(f'[{k_transposed}] = transpose({K}, permutation=[0, 1, 3, 2]);')
        else:
            steps.append(f'[{k_transposed}] = transpose({K}, permutation=[0, 1, 3, 2]);')

        # Step 2: Compute Q @ K^T
        qk = f'operand_{self.operand_counter}'
        self.operand_counter += 1
        steps.append(f'[{qk}] = matmul({Q}, {k_transposed});')

        # Step 3: Scale by 1/sqrt(head_dim)
        # Create scale constant
        scale_const = f'const_scale_{self.operand_counter}'
        self.operand_counter += 1
        import torch
        scale_tensor = torch.tensor(scale_factor, dtype=torch.float32)
        self.inline_constants[scale_const] = scale_tensor.item()

        qk_scaled = f'operand_{self.operand_counter}'
        self.operand_counter += 1
        steps.append(f'[{qk_scaled}] = mul({qk}, {scale_const});')

        # Step 4: Apply softmax along the last dimension
        # softmax is computed over the key dimension (last dim)
        attention_weights = f'operand_{self.operand_counter}'
        self.operand_counter += 1
        steps.append(f'[{attention_weights}] = softmax({qk_scaled}, axis=-1);')

        # Step 5: Multiply attention weights with V
        # attention_weights @ V
        steps.append(f'[{output}] = matmul({attention_weights}, {V});')

        return '\n    '.join(steps)

    def _convert_interpolate(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """
        Convert interpolate (upsampling/downsampling) to WebNN operations.

        PyTorch's interpolate supports various modes:
        - 'nearest': Nearest neighbor interpolation
        - 'linear', 'bilinear', 'trilinear': Linear interpolation
        - 'bicubic': Bicubic interpolation

        WebNN has resample2d for this operation.
        """
        input_tensor = inputs[0] if inputs else 'unknown'

        # Get parameters
        kwargs = node.kwargs
        scale_factor = kwargs.get('scale_factor', None)
        size = kwargs.get('size', None)
        mode = kwargs.get('mode', 'nearest')
        align_corners = kwargs.get('align_corners', None)

        # Get input shape
        input_shape = self._get_node_shape(node.args[0]) if len(node.args) > 0 and isinstance(node.args[0], fx.Node) else []

        if scale_factor is not None:
            # Use scale factor
            if isinstance(scale_factor, (int, float)):
                # Same scale for all spatial dimensions
                scales = [scale_factor, scale_factor]
            elif isinstance(scale_factor, (list, tuple)):
                scales = list(scale_factor)
            else:
                scales = [2.0, 2.0]  # default

            scales_str = ', '.join(map(str, scales))

            # WebNN mode mapping
            webnn_mode = 'nearest-neighbor' if mode == 'nearest' else 'linear'

            return f'[{output}] = resample2d({input_tensor}, mode="{webnn_mode}", scales=[{scales_str}]);'

        elif size is not None:
            # Use target size
            if isinstance(size, (list, tuple)):
                target_size = list(size)
            else:
                target_size = [size, size]

            # Calculate scales from target size
            if input_shape and len(input_shape) >= 4:
                # Input shape: [N, C, H, W]
                current_h, current_w = input_shape[-2:]
                target_h, target_w = target_size
                scale_h = target_h / current_h
                scale_w = target_w / current_w
                scales_str = f'{scale_h}, {scale_w}'
            else:
                # Without shape info, use sizes directly
                size_str = ', '.join(map(str, target_size))
                return f'[{output}] = resample2d({input_tensor}, sizes=[{size_str}]);'

            webnn_mode = 'nearest-neighbor' if mode == 'nearest' else 'linear'
            return f'[{output}] = resample2d({input_tensor}, mode="{webnn_mode}", scales=[{scales_str}]);'

        else:
            return f'// interpolate: need either scale_factor or size'

    def _convert_relu(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert ReLU to WebNN clamp"""
        input_tensor = inputs[0] if inputs else 'unknown'
        return f'[{output}] = clamp({input_tensor}, minValue=0.0);'

    def _convert_clamp(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert clamp to WebNN clamp"""
        input_tensor = inputs[0] if inputs else 'unknown'
        args = node.args
        min_val = args[1] if len(args) > 1 else None
        max_val = args[2] if len(args) > 2 else None

        params = []
        if min_val is not None:
            params.append(f'minValue={min_val}')
        if max_val is not None:
            params.append(f'maxValue={max_val}')

        params_str = ', '.join(params)
        return f'[{output}] = clamp({input_tensor}, {params_str});'

    def _convert_hardtanh(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert hardtanh to WebNN clamp (typically ReLU6)"""
        input_tensor = inputs[0] if inputs else 'unknown'
        args = node.args
        # hardtanh(input, min_val, max_val)
        min_val = args[1] if len(args) > 1 else 0.0
        max_val = args[2] if len(args) > 2 else 6.0

        return f'[{output}] = clamp({input_tensor}, maxValue={max_val}, minValue={min_val});'

    def _convert_addmm(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert addmm to WebNN gemm + add.

        torch.addmm(bias, mat1, mat2) = bias + mat1 @ mat2
          mat1: (M, K)  — input
          mat2: (K, N)  — weight, already mm-ready (no transpose)
          bias: (N,) or broadcastable
        """
        if len(inputs) < 3:
            return self._convert_identity(node, output, inputs)

        bias = inputs[0]
        mat1 = inputs[1]
        mat2 = inputs[2]

        stmts = []

        # Flatten mat1 to rank-2 if needed (e.g. coming from a conv feature map)
        input_node = node.args[1] if len(node.args) > 1 and isinstance(node.args[1], fx.Node) else None
        input_shape = self._get_node_shape(input_node) if input_node is not None else []
        if input_shape and len(input_shape) != 2:
            batch = int(input_shape[0])
            features = int(math.prod(input_shape[1:]))
            tmp = f'operand_{self.operand_counter}'
            self.operand_counter += 1
            stmts.append(f'[{tmp}] = reshape({mat1}, newShape=[{batch}, {features}]);')
            mat1 = tmp

        # gemm(mat1, mat2) = mat1 @ mat2 — mat2 is already (K, N), no bTranspose
        # Add bias separately to avoid WebNN runtime broadcast issues with 1-D c.
        mm_out = f'operand_{self.operand_counter}'
        self.operand_counter += 1
        stmts.append(f'[{mm_out}] = gemm({mat1}, {mat2});')
        stmts.append(f'[{output}] = add({mm_out}, {bias});')

        return '\n    '.join(stmts)

    def _convert_matmul(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert matmul to WebNN gemm, reshaping rank-1 inputs to rank-2."""
        if len(inputs) < 2:
            raise ValueError("Gemm requires 2 inputs")

        a, b = inputs[0], inputs[1]
        stmts = []

        # Resolve shapes for both inputs
        a_node = node.args[0] if len(node.args) > 0 and isinstance(node.args[0], fx.Node) else None
        b_node = node.args[1] if len(node.args) > 1 and isinstance(node.args[1], fx.Node) else None
        a_shape = self._get_node_shape(a_node) if a_node is not None else self.operand_shapes.get(a, [])
        b_shape = self._get_node_shape(b_node) if b_node is not None else self.operand_shapes.get(b, [])
        out_shape = self._get_node_shape(node)

        # Reshape rank-1 inputs to rank-2 (gemm requires 2D operands)
        if a_shape and len(a_shape) == 1:
            tmp = f'operand_{self.operand_counter}'
            self.operand_counter += 1
            stmts.append(f'[{tmp}] = reshape({a}, newShape=[1, {a_shape[0]}]);')
            a = tmp

        if b_shape and len(b_shape) == 1:
            tmp = f'operand_{self.operand_counter}'
            self.operand_counter += 1
            stmts.append(f'[{tmp}] = reshape({b}, newShape=[{b_shape[0]}, 1]);')
            b = tmp

        # If the expected output is not rank-2, route gemm through an intermediate
        # and reshape down (e.g. mat*vec produces a 1D result, not (1, n))
        needs_output_reshape = bool(out_shape) and len(out_shape) != 2
        gemm_out = output
        if needs_output_reshape:
            gemm_out = f'operand_{self.operand_counter}'
            self.operand_counter += 1

        stmts.append(f'[{gemm_out}] = gemm({a}, {b});')

        if needs_output_reshape:
            shape_str = ', '.join(str(d) for d in out_shape)
            stmts.append(f'[{output}] = reshape({gemm_out}, newShape=[{shape_str}]);')

        return '\n    '.join(stmts)

    def _convert_linear(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert linear function to WebNN gemm"""
        # linear(input, weight, bias)
        if len(inputs) >= 2:
            input_tensor = inputs[0]
            weight = inputs[1]
            if weight in self.operand_shapes and len(self.operand_shapes[weight]) >= 2:
                if len(inputs) >= 3:
                    bias = inputs[2]
                    # Use a separate add for the bias rather than gemm's c parameter,
                    # because the WebNN runtime does not reliably broadcast a 1-D c.
                    mm_out = f'operand_{self.operand_counter}'
                    self.operand_counter += 1
                    return (
                        f'[{mm_out}] = gemm({input_tensor}, {weight}, bTranspose=true);\n'
                        f'    [{output}] = add({mm_out}, {bias});'
                    )
                else:
                    return f'[{output}] = gemm({input_tensor}, {weight}, bTranspose=true);'

            # TODO: other cases untested
            raise NotImplementedError("This linear case is untested.")
            input_node = node.args[0] if node.args and isinstance(node.args[0], fx.Node) else None
            input_shape = self._get_node_shape(input_node) if input_node is not None else []
            # Gemm expects rank-2 input. Flatten when needed.
            if input_shape and len(input_shape) != 2:
                batch = int(input_shape[0]) if input_shape else 1
                features = int(math.prod(input_shape[1:])) if len(input_shape) > 1 else 1
                reshaped = f'operand_{self.operand_counter}'
                self.operand_counter += 1
                reshape_stmt = f'[{reshaped}] = reshape({input_tensor}, newShape=[{batch}, {features}]);'
                if len(inputs) >= 3:
                    bias = inputs[2]
                    gemm_stmt = f'[{output}] = gemm({reshaped}, {weight}, bTranspose=true, c={bias});'
                else:
                    gemm_stmt = f'[{output}] = gemm({reshaped}, {weight}, bTranspose=true);'
                return f'{reshape_stmt}\n    {gemm_stmt}'

    def _convert_linear_module(self, node: fx.Node, module: torch.nn.Linear, output: str, inputs: List[str]) -> str:
        """Convert Linear module to WebNN gemm"""
        input_tensor = inputs[0] if inputs else 'unknown'

        # Get weight and bias operands
        weight_name = f'{node.target}.weight'
        weight_operand = self.weight_operands.get(weight_name, 'unknown')

        if module.bias is not None:
            bias_name = f'{node.target}.bias'
            bias_operand = self.weight_operands.get(bias_name, 'unknown')
            return f'[{output}] = gemm({input_tensor}, {weight_operand}, bTranspose=true, c={bias_operand});'
        else:
            return f'[{output}] = gemm({input_tensor}, {weight_operand}, bTranspose=true);'

    def _convert_global_avg_pool(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert global average pooling to WebNN reduceMean"""
        input_tensor = inputs[0] if inputs else 'unknown'
        # Global average pool is reduceMean over spatial dimensions (usually axes 2,3 for NCHW)
        return f'[{output}] = reduceMean({input_tensor}, axes=[2, 3], keepDimensions=true);'

    def _convert_avg_pool2d(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert 2D average pooling to WebNN averagePool2d"""
        input_tensor = inputs[0] if inputs else 'unknown'

        # Extract parameters from args/kwargs
        kernel_size = node.args[1] if len(node.args) > 1 else node.kwargs.get('kernel_size', [2, 2])
        stride = node.args[2] if len(node.args) > 2 else node.kwargs.get('stride', kernel_size)
        padding = node.args[3] if len(node.args) > 3 else node.kwargs.get('padding', [0, 0])

        # Ensure parameters are lists
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size, kernel_size]
        if not isinstance(stride, (list, tuple)):
            stride = [stride, stride]
        if not isinstance(padding, (list, tuple)):
            padding = [padding, padding, padding, padding]
        elif len(padding) == 2:
            padding = [padding[0], padding[0], padding[1], padding[1]]

        params = []
        params.append(f'windowDimensions=[{kernel_size[0]}, {kernel_size[1]}]')
        if stride != [1, 1]:
            params.append(f'strides=[{stride[0]}, {stride[1]}]')
        if padding != [0, 0, 0, 0]:
            params.append(f'padding=[{padding[0]}, {padding[1]}, {padding[2]}, {padding[3]}]')

        params_str = ', '.join(params)
        return f'[{output}] = averagePool2d({input_tensor}, {params_str});'

    def _convert_max_pool2d(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert 2D max pooling to WebNN maxPool2d"""
        input_tensor = inputs[0] if inputs else 'unknown'

        # Extract parameters from args/kwargs
        kernel_size = node.args[1] if len(node.args) > 1 else node.kwargs.get('kernel_size', [2, 2])
        stride = node.args[2] if len(node.args) > 2 else node.kwargs.get('stride', kernel_size)
        padding = node.args[3] if len(node.args) > 3 else node.kwargs.get('padding', [0, 0])

        # Ensure parameters are lists
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size, kernel_size]
        if not isinstance(stride, (list, tuple)):
            stride = [stride, stride]
        if not isinstance(padding, (list, tuple)):
            padding = [padding, padding, padding, padding]
        elif len(padding) == 2:
            padding = [padding[0], padding[0], padding[1], padding[1]]

        params = []
        params.append(f'windowDimensions=[{kernel_size[0]}, {kernel_size[1]}]')
        if stride != [1, 1]:
            params.append(f'strides=[{stride[0]}, {stride[1]}]')
        if padding != [0, 0, 0, 0]:
            params.append(f'padding=[{padding[0]}, {padding[1]}, {padding[2]}, {padding[3]}]')

        params_str = ', '.join(params)
        return f'[{output}] = maxPool2d({input_tensor}, {params_str});'

    def _convert_reduce_mean(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert mean/reduce to WebNN reduceMean"""
        input_tensor = inputs[0] if inputs else 'unknown'

        # Get axes from args or kwargs
        axes = None
        keep_dims = True

        if 'dim' in node.kwargs:
            axes = node.kwargs['dim']
            if not isinstance(axes, (list, tuple)):
                axes = [axes]
        elif 'axis' in node.kwargs:
            axes = node.kwargs['axis']
            if not isinstance(axes, (list, tuple)):
                axes = [axes]
        elif len(node.args) > 1:
            axes = node.args[1]
            if not isinstance(axes, (list, tuple)):
                axes = [axes]

        if 'keepdim' in node.kwargs:
            keep_dims = node.kwargs['keepdim']
        elif 'keepdims' in node.kwargs:
            keep_dims = node.kwargs['keepdims']

        if axes:
            axes_str = ', '.join(map(str, axes))
            keep_str = 'true' if keep_dims else 'false'
            return f'[{output}] = reduceMean({input_tensor}, axes=[{axes_str}], keepDimensions={keep_str});'
        else:
            # Reduce over all axes
            return f'[{output}] = reduceMean({input_tensor});'

    def _convert_transpose(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert transpose/permute to WebNN transpose"""
        input_tensor = inputs[0] if inputs else 'unknown'

        # Get permutation from args or kwargs
        perm = None
        if 'dims' in node.kwargs:
            perm = node.kwargs['dims']
        elif len(node.args) > 1:
            # For permute: permute(input, dim0, dim1, ...)
            # For transpose: transpose(input, dim0, dim1)
            if len(node.args) == 3:  # transpose(input, dim0, dim1)
                dim0, dim1 = node.args[1], node.args[2]
                # Create permutation that swaps dim0 and dim1
                # Need to know rank from metadata
                input_shape = self._get_node_shape(node.args[0]) if isinstance(node.args[0], fx.Node) else []
                rank = len(input_shape)
                perm = list(range(rank))
                if dim0 < rank and dim1 < rank:
                    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
            else:  # permute(input, dim0, dim1, dim2, ...)
                perm = list(node.args[1:])

        if perm:
            perm_str = ', '.join(map(str, perm))
            return f'[{output}] = transpose({input_tensor}, permutation=[{perm_str}]);'
        else:
            # Default transpose (reverse all dimensions)
            return f'[{output}] = transpose({input_tensor});'

    def _convert_concat(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert concatenation to WebNN concat"""
        # Get axis/dim
        axis = node.kwargs.get('dim', 0)
        if 'axis' in node.kwargs:
            axis = node.kwargs['axis']
        elif len(node.args) > 1 and not isinstance(node.args[1], fx.Node):
            axis = node.args[1]

        # Collect input tensors
        # torch.cat takes a list/tuple as first argument
        if len(inputs) > 0:
            inputs_str = ', '.join(inputs)
            return f'[{output}] = concat([{inputs_str}], axis={axis});'
        return f'// Invalid concat operation'

    def _convert_stack(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """
        Convert stack to WebNN operations.

        Stack creates a new dimension and concatenates tensors along it.
        For example: stack([a, b, c], dim=0) where a.shape=[2, 3]
        results in output.shape=[3, 2, 3]

        Implementation: unsqueeze each input at the stack dimension, then concat
        """
        # Get dimension to stack along
        dim = node.kwargs.get('dim', 0)
        if 'axis' in node.kwargs:
            dim = node.kwargs['axis']

        if not inputs:
            return f'// Invalid stack: no inputs'

        # Get shape of first input to understand dimensions
        if len(node.args) > 0 and isinstance(node.args[0], (list, tuple)):
            # Stack takes a list of tensors as first argument
            first_tensor = node.args[0][0] if node.args[0] else None
            if first_tensor and isinstance(first_tensor, fx.Node):
                input_shape = self._get_node_shape(first_tensor)
            else:
                input_shape = []
        else:
            input_shape = []

        # Unsqueeze each input at the stack dimension
        unsqueezed_operands = []
        steps = []

        for i, inp in enumerate(inputs):
            unsqueezed = f'operand_{self.operand_counter}'
            self.operand_counter += 1
            unsqueezed_operands.append(unsqueezed)

            if input_shape:
                # Calculate new shape with unsqueezed dimension
                new_shape = list(input_shape)
                # Insert 1 at the stack dimension
                # Handle negative indexing
                if dim < 0:
                    insert_pos = len(new_shape) + 1 + dim
                else:
                    insert_pos = dim
                new_shape.insert(insert_pos, 1)

                shape_str = ', '.join(map(str, new_shape))
                steps.append(f'[{unsqueezed}] = reshape({inp}, newShape=[{shape_str}]);')
            else:
                # Without shape info, still try to unsqueeze
                # This might not work perfectly without knowing the actual shape
                steps.append(f'[{unsqueezed}] = reshape({inp}, newShape=[..., 1, ...]);')

        # Concatenate all unsqueezed tensors along the stack dimension
        unsqueezed_str = ', '.join(unsqueezed_operands)
        # Handle negative indexing
        if dim < 0 and input_shape:
            concat_dim = len(input_shape) + 1 + dim
        else:
            concat_dim = dim

        steps.append(f'[{output}] = concat([{unsqueezed_str}], axis={concat_dim});')

        return '\n    '.join(steps)

    def _convert_split(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert split to WebNN split"""
        input_tensor = inputs[0] if inputs else 'unknown'

        # Get split sizes or number of splits
        split_size_or_sections = node.args[1] if len(node.args) > 1 else None
        dim = node.kwargs.get('dim', 0)
        if len(node.args) > 2:
            dim = node.args[2]

        if isinstance(split_size_or_sections, (list, tuple)):
            # Split into specific sizes
            splits_str = ', '.join(map(str, split_size_or_sections))
            return f'[{output}] = split({input_tensor}, splits=[{splits_str}], axis={dim});'
        else:
            # Split into equal sections
            return f'[{output}] = split({input_tensor}, splits={split_size_or_sections}, axis={dim});'

    def _convert_slice(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert slice to WebNN slice"""
        input_tensor = inputs[0] if inputs else 'unknown'

        # torch.slice or tensor slicing
        # Get parameters from args
        dim = node.args[1] if len(node.args) > 1 else 0
        start = node.args[2] if len(node.args) > 2 else 0
        end = node.args[3] if len(node.args) > 3 else None
        step = node.args[4] if len(node.args) > 4 else 1

        # Get input shape to calculate sizes
        input_shape = self._get_node_shape(node.args[0]) if node.args and isinstance(node.args[0], fx.Node) else []

        if end is None and input_shape and dim < len(input_shape):
            end = input_shape[dim]

        if end is not None:
            size = (end - start) // step
            # WebNN slice takes starts and sizes
            # Build starts and sizes for all dimensions
            starts = [0] * len(input_shape)
            sizes = list(input_shape)
            starts[dim] = start
            sizes[dim] = size

            starts_str = ', '.join(map(str, starts))
            sizes_str = ', '.join(map(str, sizes))
            return f'[{output}] = slice({input_tensor}, starts=[{starts_str}], sizes=[{sizes_str}]);'

        return f'// Slice with unknown dimensions'

    def _convert_expand(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert expand to WebNN expand (broadcast)"""
        input_tensor = inputs[0] if inputs else 'unknown'

        # Get target shape from args
        target_shape = node.args[1] if len(node.args) > 1 else None

        if target_shape:
            if isinstance(target_shape, (list, tuple)):
                shape_str = ', '.join(map(str, target_shape))
            else:
                shape_str = str(target_shape)
            return f'[{output}] = expand({input_tensor}, newShape=[{shape_str}]);'

        return f'// Expand with unknown shape'

    def _convert_pad(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert pad to WebNN pad"""
        input_tensor = inputs[0] if inputs else 'unknown'

        # Get padding from args
        padding = node.args[1] if len(node.args) > 1 else [0, 0, 0, 0]
        mode = node.kwargs.get('mode', 'constant')
        value = node.kwargs.get('value', 0)

        # Convert padding format
        if isinstance(padding, (list, tuple)):
            # PyTorch padding is usually [left, right, top, bottom] for 2D
            # WebNN expects [begin_0, end_0, begin_1, end_1, ...]
            padding_str = ', '.join(map(str, padding))

            if mode == 'constant':
                return f'[{output}] = pad({input_tensor}, padding=[{padding_str}], mode="constant", value={value});'
            else:
                return f'[{output}] = pad({input_tensor}, padding=[{padding_str}], mode="{mode}");'

        return f'// Pad with unknown parameters'

    def _convert_tile(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert tile/repeat to WebNN tile"""
        input_tensor = inputs[0] if inputs else 'unknown'

        # Get repetitions from args
        reps = node.args[1] if len(node.args) > 1 else None

        if reps:
            if isinstance(reps, (list, tuple)):
                reps_str = ', '.join(map(str, reps))
            else:
                reps_str = str(reps)
            return f'[{output}] = tile({input_tensor}, repetitions=[{reps_str}]);'

        return f'// Tile with unknown repetitions'

    def _convert_reshape(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """Convert reshape/view to WebNN reshape"""
        input_tensor = inputs[0] if inputs else 'unknown'

        # Prefer static output shape from FX metadata when available.
        meta_shape = self._get_node_shape(node)
        if meta_shape:
            shape_str = ', '.join(map(str, meta_shape))
            return f'[{output}] = reshape({input_tensor}, newShape=[{shape_str}]);'

        # Handle flatten(input, start_dim, end_dim) style arguments.
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
                        shape_str = ', '.join(map(str, new_shape))
                        return f'[{output}] = reshape({input_tensor}, newShape=[{shape_str}]);'

        # Extract new shape from args
        if len(node.args) > 1:
            new_shape = node.args[1]
            if isinstance(new_shape, (list, tuple)):
                shape_str = ', '.join(map(str, new_shape))
                return f'[{output}] = reshape({input_tensor}, newShape=[{shape_str}]);'

        # Last-resort: preserve rank or flatten using input metadata when available.
        if node.args and isinstance(node.args[0], fx.Node):
            in_shape = self._get_node_shape(node.args[0])
            if in_shape:
                if len(in_shape) > 2:
                    batch = int(in_shape[0])
                    features = int(math.prod(in_shape[1:]))
                    return f'[{output}] = reshape({input_tensor}, newShape=[{batch}, {features}]);'
                shape_str = ', '.join(map(str, in_shape))
                return f'[{output}] = reshape({input_tensor}, newShape=[{shape_str}]);'

        return f'[{output}] = add({input_tensor}, {input_tensor});'

    def _convert_identity(self, node: fx.Node, output: str, inputs: List[str]) -> str:
        """
        Emit a shape-preserving identity via reshape to keep the graph executable
        when an op is not mapped yet.
        """
        if len(inputs) == 1:
            return f'[{output}] = identity({inputs[0]});'
        return '// Invalid identity operation'

    def _create_inline_constant(self, value) -> str:
        """
        Create an inline constant operand for a scalar value.
        These constants are embedded directly in the .webnn file, not stored in safetensors.
        """
        # Check if we already have this constant
        for const_name, const_value in self.inline_constants.items():
            value_type = type(value)
            if isinstance(value, torch.Tensor) and isinstance(const_value, torch.Tensor):
                if torch.allclose(value, const_value, rtol=1e-5, atol=1e-8):
                    return const_name
            elif isinstance(value, value_type) and isinstance(const_value, value_type):
                return const_name

        # Create a new constant operand
        const_name = f'const_scalar_{self.operand_counter}'
        self.operand_counter += 1

        # Store the constant value
        self.inline_constants[const_name] = value

        return const_name

    def _extract_inline_constants(self) -> str:
        """Extract inline scalar constants that are embedded in the .webnn file"""
        consts = []

        for name, value in self.inline_constants.items():
            # Determine the type based on the value
            if isinstance(value, torch.Tensor):
                dtype = self._get_webnn_dtype(value.dtype)
                shape = list(value.shape)
                shape_str = ', '.join(map(str, shape))
                raw = value.cpu().numpy().tobytes()
                byte_list = ', '.join(str(b) for b in raw)
                consts.append(f'    {name}: {dtype}[{shape_str}] @bytes([{byte_list}]);')
            else:
                if isinstance(value, float):
                    dtype = 'f32'
                elif isinstance(value, int):
                    dtype = 'i32'
                else:
                    dtype = 'f32'  # default

                consts.append(f'    {name}: {dtype}[] @scalar({value});')

        return '\n'.join(consts) + '\n' if consts else ''

    def _get_node_shape(self, node: fx.Node) -> List[int]:
        """Best-effort extraction of static shape from FX node metadata."""
        if not hasattr(node, 'meta'):
            return []
        meta = node.meta
        val = meta.get('val')
        if val is not None and hasattr(val, 'shape'):
            return [int(d) for d in val.shape]
        tensor_meta = meta.get('tensor_meta')
        if tensor_meta is not None and hasattr(tensor_meta, 'shape'):
            return [int(d) for d in tensor_meta.shape]
        return []

    def _extract_outputs(self, gm: fx.GraphModule) -> str:
        """Extract output declarations"""
        outputs = []
        for node in gm.graph.nodes:
            if node.op == 'output':
                # Output node contains the return value
                if isinstance(node.args[0], (list, tuple)):
                    for arg in node.args[0]:
                        if isinstance(arg, fx.Node):
                            outputs.append(self._get_input_operand(arg))
                elif isinstance(node.args[0], fx.Node):
                    outputs.append(self._get_input_operand(node.args[0]))

        return '; '.join(outputs) + ';' if outputs else ''

    def _get_operand_name(self, node: fx.Node) -> str:
        """Get or create operand name for a node"""
        if node.name not in self.node_to_operand:
            self.node_to_operand[node.name] = f'operand_{self.operand_counter}'
            self.operand_counter += 1
        return self.node_to_operand[node.name]

    def _get_input_operand(self, node) -> str:
        """Get operand name for an input node"""
        if isinstance(node, fx.Node):
            # Map FX placeholders for parameters/buffers back to state_dict keys so
            # ops reference declared const operands.
            if node.op == 'placeholder':
                key = self._placeholder_to_state_key(node.name)
                if key in self.weight_operands:
                    return self.weight_operands[key]
            if node.name in self.node_to_operand:
                return self.node_to_operand[node.name]
            else:
                return self._get_operand_name(node)
        return str(node)

    def _placeholder_to_state_key(self, name: str) -> str:
        """
        Convert Dynamo placeholder names like:
          l_self_modules_features_modules_0_parameters_weight_
        to state_dict keys like:
          features.0.weight
        """
        key = name
        if key.startswith('l_'):
            key = key[2:]
        hierarchy = key.split('_')
        if hierarchy[0] == 'self':
            hierarchy.pop(0)
        if hierarchy[-1] == '':
            hierarchy.pop(-1)

        hierarchy = filter(lambda x: x not in IGNORED_PLACEHOLDER_TOKENS, hierarchy)
        return '.'.join(hierarchy)

    def _get_webnn_dtype(self, dtype: torch.dtype) -> str:
        """Convert PyTorch dtype to WebNN dtype"""
        dtype_map = {
            torch.float32: 'f32',
            torch.float16: 'f16',
            torch.int32: 'i32',
            torch.int64: 'i64',
            torch.int8: 'i8',
            torch.uint8: 'u8',
        }
        return dtype_map.get(dtype, 'f32')

    def _get_module(self, gm: fx.GraphModule, target: str):
        """Get module from GraphModule by target path"""
        atoms = target.split('.')
        mod = gm
        for atom in atoms:
            if not hasattr(mod, atom):
                return None
            mod = getattr(mod, atom)
        return mod

"""
Custom PyTorch Exporter using torch.compile and Dynamo
Demonstrates how to build a custom export backend for PyTorch models
"""

import torch
from torch._dynamo.backends.common import aot_autograd
from torch.fx.passes.shape_prop import ShapeProp
from typing import Callable, List, Optional, Union, Tuple
import json
import struct
from safetensors.torch import save_file, load_file
from .webnn_generator import WebNNGraphGenerator


class CustomExporter:
    """
    Custom exporter that captures the FX graph from Dynamo and converts it
    to a custom format. This is a minimal example to understand the flow.
    """

    def __init__(self, debug=True):
        self.debug = debug
        self.exported_graphs = []
        self.model = None  # Store reference to the original model
        self.fx_graph_module = None  # Store FX GraphModule for WebNN export
        self.webnn_generator = WebNNGraphGenerator()

    def export_graph(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        """
        This function receives the FX graph from Dynamo and can process it.

        Args:
            gm: The torch.fx.GraphModule representing the traced computation
            example_inputs: Example inputs used for tracing
        """
        if self.debug:
            print("\n" + "="*80)
            print("DYNAMO EXPORT CALLBACK TRIGGERED")
            print("="*80)
            print("\nGraph Module:")
            print(gm.graph)
            print("\nCode:")
            print(gm.code)
            print("\n" + "="*80)

        # Propagate shapes through the graph to add metadata
        # This ensures that all nodes have shape information available
        try:
            ShapeProp(gm).propagate(*example_inputs)
            if self.debug:
                print("\n" + "="*80)
                print("Shape propagation successful")
                print("="*80)
        except Exception as e:
            if self.debug:
                print(f"\nWarning: Shape propagation failed: {e}")
                print("Some operations may not have complete shape information")

        # Convert FX graph to custom format
        graph_repr = self._convert_fx_to_custom_format(gm)
        self.exported_graphs.append(graph_repr)

        # Store FX graph module for WebNN export
        self.fx_graph_module = gm

        # Return the original graph module so it can still be executed
        # Note: This needs to return the FX graph for Dynamo to work correctly.
        # To get a WebNN executor instead, use export_model_with_weights(..., return_executor=True)
        return gm

    def _convert_fx_to_custom_format(self, gm: torch.fx.GraphModule) -> dict:
        """
        Convert FX graph to a custom format.
        This is where you'd implement your actual export logic.
        """
        nodes = []

        for node in gm.graph.nodes:
            node_info = {
                'name': node.name,
                'op': node.op,
                'target': str(node.target),
                'args': [str(arg) for arg in node.args],
                'kwargs': {k: str(v) for k, v in node.kwargs.items()},
            }

            # Add type-specific information
            if node.op == 'call_function':
                node_info['function'] = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
            elif node.op == 'call_method':
                node_info['method'] = node.target
            elif node.op == 'call_module':
                node_info['module'] = node.target

            nodes.append(node_info)

            if self.debug:
                print(f"\nNode: {node.name}")
                print(f"  Op: {node.op}")
                print(f"  Target: {node.target}")
                print(f"  Args: {node.args}")
                print(f"  Kwargs: {node.kwargs}")
                if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                    print(f"  Tensor Meta: {node.meta['tensor_meta']}")

        return {
            'nodes': nodes,
            'graph_str': str(gm.graph),
            'code': gm.code
        }

    def save_to_file(self, filepath: str):
        """Save exported graphs to a JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.exported_graphs, f, indent=2)
        print(f"\nExported graphs saved to {filepath}")

    def save_weights(self, model: torch.nn.Module, filepath: str):
        """
        Save model weights to a safetensors file

        Args:
            model: The PyTorch model whose weights to save
            filepath: Path to save the safetensors file
        """
        state_dict = model.state_dict()

        # Add generated constants (like arange) to the state dict
        if hasattr(self, 'webnn_generator') and hasattr(self.webnn_generator, 'generated_constants'):
            for name, tensor in self.webnn_generator.generated_constants.items():
                # Use special prefix to distinguish generated constants
                key = f'_generated.{name}'
                state_dict[key] = tensor

        save_file(state_dict, filepath)

    def set_model(self, model: torch.nn.Module):
        """Store reference to the original model"""
        self.model = model

    def save_to_webnn(self, filepath: str, graph_name: str = "model"):
        """
        Save model as WebNN graph format

        Args:
            filepath: Path to save the .webnn file
            graph_name: Name for the WebNN graph
        """
        if self.fx_graph_module is None:
            raise ValueError("No FX graph available. Run export_model first.")
        if self.model is None:
            raise ValueError("No model reference. Run export_model first.")

        webnn_graph = self.webnn_generator.generate(
            self.fx_graph_module,
            self.model,
            graph_name=graph_name
        )

        with open(filepath, 'w') as f:
            f.write(webnn_graph)

# Global exporter instance
_exporter = None


def get_custom_backend(debug=True):
    """
    Factory function that returns a Dynamo backend using our custom exporter.
    This is what you pass to torch.compile(backend=...)
    """
    global _exporter
    _exporter = CustomExporter(debug=debug)

    def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        return _exporter.export_graph(gm, example_inputs)

    return custom_backend


def get_exporter():
    """Get the global exporter instance to access exported graphs"""
    return _exporter


def export_model(
    model: torch.nn.Module,
    example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    debug=True
):
    """
    High-level API to export a model using our custom backend.

    Args:
        model: The PyTorch model to export
        example_input: Example input tensor(s) for tracing. Can be:
            - A single torch.Tensor
            - A tuple of torch.Tensors for models with multiple inputs
        debug: Whether to print debug information

    Returns:
        Compiled model and exporter instance
    """
    backend = get_custom_backend(debug=debug)
    compiled_model = torch.compile(model, backend=backend)

    # Store model reference in exporter
    if _exporter:
        _exporter.set_model(model)

    # Run once to trigger export
    with torch.no_grad():
        if isinstance(example_input, tuple):
            compiled_model(*example_input)
        else:
            compiled_model(example_input)

    return compiled_model, get_exporter()


def export_model_with_weights(
    model: torch.nn.Module,
    example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    webnn_path: str,
    weights_path: str,
    debug=True,
    graph_name: str = "model",
    return_executor: bool = False
):
    """
    Export model to WebNN format with weights.

    Args:
        model: The PyTorch model to export
        example_input: Example input tensor(s) for tracing. Can be:
            - A single torch.Tensor
            - A tuple of torch.Tensors for models with multiple inputs
        webnn_path: Path to save the WebNN graph file
        weights_path: Path to save the weights safetensors file
        debug: Whether to print debug information
        graph_name: Name for the WebNN graph (default: "model")
        return_executor: If True, returns a WebNNExecutor instead of compiled_model.
                        The executor wraps the WebNN graph and provides a PyTorch-like
                        interface with automatic tensor conversion. (default: False)

    Returns:
        If return_executor=False: Tuple of (compiled_model, exporter)
        If return_executor=True: Tuple of (webnn_executor, exporter)
    """
    # Export the model graph
    compiled_model, exporter = export_model(model, example_input, debug=debug)

    # Save WebNN format
    exporter.save_to_webnn(webnn_path, graph_name=graph_name)

    # Save weights to safetensors
    exporter.save_weights(model, weights_path)

    if return_executor:
        # Import executor here to avoid import errors if webnn not available
        from .executor import WebNNExecutor

        try:
            # Create and return WebNN executor
            executor = WebNNExecutor(webnn_path, weights_path, example_input)
            return executor, exporter
        except ImportError as e:
            print(f"Warning: Could not create WebNNExecutor: {e}")
            print("Returning compiled PyTorch model instead.")
            return compiled_model, exporter

    return compiled_model, exporter


def load_weights_from_safetensors(model: torch.nn.Module, filepath: str, strict=True):
    """
    Load weights from a safetensors file into a model.

    Args:
        model: The PyTorch model to load weights into
        filepath: Path to the safetensors file
        strict: Whether to strictly enforce that the keys in state_dict match

    Returns:
        The model with loaded weights
    """
    state_dict = load_file(filepath)
    model.load_state_dict(state_dict, strict=strict)
    print(f"\nWeights loaded from {filepath}")

    # Print loading statistics
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameters loaded: {total_params:,}")

    return model



if __name__ == '__main__':
    main()

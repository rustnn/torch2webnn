import re
import torch
import numpy as np
from typing import Union, Tuple, Dict, Any, List

try:
    import webnn
    WEBNN_AVAILABLE = True
except ImportError:
    WEBNN_AVAILABLE = False


class WebNNExecutor(torch.nn.Module):
    """
    Executor that wraps a WebNN graph and provides a PyTorch-like interface.

    This allows using exported WebNN models as drop-in replacements for PyTorch models.
    Handles automatic conversion of PyTorch tensors to/from numpy arrays.
    """

    def __init__(self, webnn_path: str, weights_path: str, example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
        super().__init__()

        if not WEBNN_AVAILABLE:
            raise ImportError(
                "WebNN is not available. This executor requires the WebNN runtime to be installed. "
                "The model has been exported to WebNN format, but cannot be executed without the runtime."
            )

        self.webnn_path = webnn_path
        self.weights_path = weights_path
        self.example_input = example_input

        # Create WebNN context and load graph
        self.context = webnn.ML().create_context(device_type="cpu")
        self.webnn_graph = webnn.MLGraph.load(
            webnn_path,
            weights_path=weights_path
        )

        # Get input/output names
        self.input_names = self.webnn_graph.get_input_names()
        self.output_names = self.webnn_graph.get_output_names()

        # Parse declaration order from the file — the runtime's get_input_names()
        # may return names in alphabetical order rather than declaration order,
        # so we parse the inputs {} section to get the correct positional mapping.
        self.ordered_input_names = self._parse_input_order(webnn_path)

        # Determine if we have multiple inputs
        self.multi_input = isinstance(example_input, tuple)

    def _parse_input_order(self, webnn_path: str) -> List[str]:
        """Parse the WebNN file's inputs {} section to get names in declaration order."""
        try:
            with open(webnn_path, 'r') as f:
                content = f.read()
            match = re.search(r'inputs\s*\{([^}]*)\}', content)
            if match:
                names = re.findall(r'(\w+)\s*:', match.group(1))
                if names:
                    return names
        except Exception:
            pass
        return self.input_names  # fallback to runtime order

    def forward(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Execute the WebNN graph with PyTorch tensor inputs.

        Args:
            *args: Input tensors (single tensor or multiple tensors)
            **kwargs: Named input tensors

        Returns:
            Output tensor(s) as PyTorch tensors
        """
        # Prepare inputs dictionary
        inputs_dict = {}

        if self.multi_input:
            # Multiple inputs — use declaration order so positional args are mapped correctly
            if len(args) != len(self.ordered_input_names):
                raise ValueError(
                    f"Expected {len(self.ordered_input_names)} inputs, got {len(args)}. "
                    f"Input names: {self.ordered_input_names}"
                )

            for input_name, input_tensor in zip(self.ordered_input_names, args):
                # Convert to numpy array on CPU with float32 dtype
                np_input = input_tensor.detach().cpu().numpy().astype(np.float32)
                inputs_dict[input_name] = np_input
        else:
            # Single input
            if len(args) == 0:
                raise ValueError("No input provided")

            input_tensor = args[0]
            np_input = input_tensor.detach().cpu().numpy().astype(np.float32)
            inputs_dict[self.input_names[0]] = np_input

        # Execute WebNN graph
        outputs = self.context.compute(self.webnn_graph, inputs_dict)

        # Convert outputs back to PyTorch tensors
        if len(self.output_names) == 1:
            # Single output
            output_array = outputs[self.output_names[0]]
            return torch.from_numpy(output_array)
        else:
            # Multiple outputs
            output_tensors = []
            for output_name in self.output_names:
                output_array = outputs[output_name]
                output_tensors.append(torch.from_numpy(output_array))
            return tuple(output_tensors)

    def __repr__(self):
        return f"WebNNExecutor(graph={self.webnn_path}, weights={self.weights_path})"


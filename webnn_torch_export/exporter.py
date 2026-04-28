"""
PyTorch → WebNN exporter using torch.export (ATen IR).

Public API:
    export_model(model, example_input)              → (compiled_model, exporter)
    export_model_with_weights(model, example_input, webnn_path, weights_path)
"""

import torch
import torch.export
from typing import Union, Tuple
from safetensors.torch import save_file, load_file

from .webnn_generator import WebNNGraphGenerator


class CustomExporter:
    """
    Wraps a torch.export.ExportedProgram and converts it to the WebNN graph format.
    """

    def __init__(self, ep: torch.export.ExportedProgram, debug: bool = False):
        self.ep = ep
        self.debug = debug
        self.webnn_generator = WebNNGraphGenerator()
        # Compatibility: expose graph nodes in the same dict format as the old Dynamo backend
        self.exported_graphs = [self._graph_to_dict(ep.graph_module)]

        if debug:
            print("\n" + "=" * 80)
            print("EXPORTED PROGRAM")
            print("=" * 80)
            ep.graph.print_tabular()

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _graph_to_dict(gm: torch.fx.GraphModule) -> dict:
        """Return graph nodes in the same dict format as the old Dynamo-based exporter."""
        nodes = []
        for node in gm.graph.nodes:
            nodes.append({
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "args": [str(a) for a in node.args],
                "kwargs": {k: str(v) for k, v in node.kwargs.items()},
            })
        return {"nodes": nodes, "graph_str": str(gm.graph)}

    # ------------------------------------------------------------------
    # WebNN / weights serialisation
    # ------------------------------------------------------------------

    def save_to_webnn(self, filepath: str, graph_name: str = "model") -> None:
        """Write the WebNN graph text file."""
        webnn_graph = self.webnn_generator.generate(self.ep, graph_name=graph_name)
        with open(filepath, "w") as f:
            f.write(webnn_graph)

    def save_weights(self, filepath: str) -> None:
        """
        Save model parameters and buffers to a safetensors file.

        Keys match the state_dict paths referenced in @weights(...) inside the
        .webnn file (e.g. "conv1.weight", "bn.running_mean").
        """
        state_dict = {
            **dict(self.ep.named_parameters()),
            **dict(self.ep.named_buffers()),
        }
        save_file(state_dict, filepath)


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def export_model(
    model: torch.nn.Module,
    example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    debug: bool = False,
) -> Tuple["CustomExporter", torch.export.ExportedProgram]:
    """
    Export a model to ATen IR using torch.export.

    Args:
        model: The PyTorch model.
        example_input: A single tensor or tuple of tensors.
        debug: Print the exported graph.

    Returns:
        (exporter, exported_program)
    """
    if not isinstance(example_input, tuple):
        example_input = (example_input,)

    with torch.no_grad():
        ep = torch.export.export(model, example_input)

    exporter = CustomExporter(ep, debug=debug)
    return exporter, ep


def export_model_with_weights(
    model: torch.nn.Module,
    example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    webnn_path: str,
    weights_path: str,
    debug: bool = False,
    graph_name: str = "model",
    return_executor: bool = False,
):
    """
    Export model to WebNN format and save weights.

    Args:
        model: The PyTorch model.
        example_input: Example input tensor(s).
        webnn_path: Path to write the .webnn graph file.
        weights_path: Path to write the .safetensors weights file.
        debug: Print the exported graph.
        graph_name: Name embedded in the .webnn file header.
        return_executor: If True, returns a WebNNExecutor.

    Returns:
        If return_executor=False: (exporter, exported_program)
        If return_executor=True:  (webnn_executor, exported_program)
    """
    exporter, ep = export_model(model, example_input, debug=debug)
    exporter.save_to_webnn(webnn_path, graph_name=graph_name)
    exporter.save_weights(weights_path)

    if return_executor:
        from .executor import WebNNExecutor
        try:
            executor = WebNNExecutor(webnn_path, weights_path, example_input)
            return executor, ep
        except ImportError as e:
            print(f"Warning: WebNN runtime not available: {e}")
            return exporter, ep

    return exporter, ep


def load_weights_from_safetensors(
    model: torch.nn.Module, filepath: str, strict: bool = True
) -> torch.nn.Module:
    """Load weights from a safetensors file into a model."""
    state_dict = load_file(filepath)
    model.load_state_dict(state_dict, strict=strict)
    total = sum(p.numel() for p in state_dict.values())
    print(f"Loaded {total:,} parameters from {filepath}")
    return model
"""Export fixed-size PyTorch models directly into pywebnn networks."""

from typing import Tuple, Union
from pathlib import Path
import shutil

import torch
import torch.export
from safetensors.torch import load_file, save_file

from .webnn_generator import WebNNGraphGenerator


class CustomExporter:
    """An exported ATen program and its shape-checked, in-memory WebNN graph."""

    def __init__(
        self, ep: torch.export.ExportedProgram, debug: bool = False, context=None
    ):
        self.ep = ep
        self.debug = debug
        self.webnn_generator = WebNNGraphGenerator(context=context)
        self.context = self.webnn_generator.context
        self.exported_graphs = [self._graph_to_dict(ep.graph_module)]
        if debug:
            ep.graph.print_tabular()
        self.webnn_graph = self.webnn_generator.generate(ep)

    @staticmethod
    def _graph_to_dict(gm: torch.fx.GraphModule) -> dict:
        return {
            "nodes": [
                {
                    "name": n.name,
                    "op": n.op,
                    "target": str(n.target),
                    "args": [str(a) for a in n.args],
                    "kwargs": {k: str(v) for k, v in n.kwargs.items()},
                }
                for n in gm.graph.nodes
            ],
            "graph_str": str(gm.graph),
        }

    def to_executor(self):
        """Wrap the constructed network in a PyTorch-like callable."""
        from .executor import WebNNExecutor

        return WebNNExecutor(self)

    def to_webnn_text(self) -> str:
        """Return RustNN's debug text for the constructed graph (inline constants)."""
        generator = self.webnn_generator
        return generator.builder.rustnn_webnn_text_for_outputs(generator.outputs)

    def save_to_webnn(self, filepath: str, graph_name: str = "model") -> None:
        """Delegate text and sibling .safetensors serialization to RustNN.

        graph_name is accepted for source compatibility; RustNN controls the header.
        """
        generator = self.webnn_generator
        generator.builder.rustnn_save_webnn(generator.outputs, filepath)

    def save_weights(self, filepath: str) -> None:
        """Optionally save a separate copy of the original parameters and buffers."""
        state_dict = {
            **dict(self.ep.named_parameters()),
            **dict(self.ep.named_buffers()),
        }
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in state_dict.items()},
            str(filepath),
        )


def export_model(
    model: torch.nn.Module,
    example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    debug: bool = False,
    *,
    context=None,
) -> Tuple[CustomExporter, torch.export.ExportedProgram]:
    """Build a static WebNN network and check every operand's PyTorch shape.

    Returns (exporter, exported_program). Use exporter.to_executor() for inference.
    Construction needs no runtime. An optional pywebnn context selects a backend/device.
    The model's training/evaluation mode is respected; use model.eval() for inference.
    """
    if not isinstance(example_input, tuple):
        example_input = (example_input,)
    with torch.no_grad():
        ep = torch.export.export(model, example_input)
    return CustomExporter(ep, debug=debug, context=context), ep


def export_model_with_weights(
    model: torch.nn.Module,
    example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    webnn_path: str,
    weights_path: str,
    debug: bool = False,
    graph_name: str = "model",
    return_executor: bool = False,
    *,
    context=None,
):
    """Build directly, then save RustNN text and its external constant weights.

    This compatibility helper uses the RustNN saver exposed by pywebnn. Execution uses the
    already-built network, with no file parsing or graph reload.
    """
    exporter, ep = export_model(model, example_input, debug=debug, context=context)
    exporter.save_to_webnn(webnn_path, graph_name=graph_name)
    sidecar = Path(webnn_path).with_suffix(".safetensors")
    if sidecar.resolve() != Path(weights_path).resolve():
        shutil.copyfile(sidecar, weights_path)
    return (exporter.to_executor() if return_executor else exporter), ep


def load_weights_from_safetensors(
    model: torch.nn.Module, filepath: str, strict: bool = True
):
    """Load a weights snapshot into a PyTorch model."""
    model.load_state_dict(load_file(filepath), strict=strict)
    return model


def main():
    """Build a pywebnn network from a saved torch.export program (.pt2)."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="Fixed-size torch.export program (.pt2)")
    parser.add_argument(
        "output", help="Destination for pywebnn native graph serialization"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    CustomExporter(torch.export.load(args.program), debug=args.debug).save_to_webnn(
        args.output
    )

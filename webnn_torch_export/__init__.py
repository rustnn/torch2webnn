"""Build and execute shape-checked WebNN networks from fixed-size PyTorch models."""

from .exporter import (
    CustomExporter,
    export_model,
    export_model_with_weights,
    load_weights_from_safetensors,
)
from .executor import WebNNExecutor
from .webnn_generator import ShapeMismatchError

__version__ = "0.1.0"
__all__ = [
    "CustomExporter",
    "export_model",
    "export_model_with_weights",
    "load_weights_from_safetensors",
    "WebNNExecutor",
    "ShapeMismatchError",
]

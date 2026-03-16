"""
WebNN PyTorch Exporter - Custom export backend using Dynamo
"""

from .exporter import (
    CustomExporter,
    export_model,
    export_model_with_weights,
    load_weights_from_safetensors,
    get_custom_backend,
    get_exporter,
)

# Optional import - WebNNExecutor requires webnn runtime
try:
    from .executor import WebNNExecutor
    _executor_available = True
except ImportError:
    _executor_available = False
    WebNNExecutor = None

__version__ = "0.1.0"

__all__ = [
    "CustomExporter",
    "export_model",
    "export_model_with_weights",
    "load_weights_from_safetensors",
    "get_custom_backend",
    "get_exporter",
    "WebNNExecutor",  # May be None if webnn not available
]

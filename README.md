# WebNN PyTorch Exporter

Build and execute fixed-size WebNN networks directly with `pywebnn` from
`torch.export` ATen graphs.

## Installation

```bash
pip install -e ".[dev]"
pytest
```

`pywebnn` is a required dependency. Use the updated local pywebnn bindings that
expose `MLGraphBuilder.new_uncompiled`, `rustnn_webnn_text_for_outputs`, and
`rustnn_save_webnn`. Construction and text export need no runtime backend. For
inference, the selected runtime must be available to the dynamic loader. For ONNX Runtime, put the directory containing
`libonnxruntime.so` in `LD_LIBRARY_PATH`; if multiple runtimes are installed,
`WEBNN_RUNTIME_LIBRARY` selects the exact library before importing `webnn`.

## Export and execute

```python
import torch
from webnn_torch_export import export_model

model = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.ReLU()).eval()
x = torch.randn(2, 10)

exporter, exported_program = export_model(model, x)
executor = exporter.to_executor()
y = executor(x)
torch.testing.assert_close(y, model(x), rtol=1e-4, atol=1e-4)
```

`export_model` constructs `exporter.webnn_graph` immediately with an uncompiled
pywebnn builder. Parameters, buffers,
lifted tensor constants, and generated constants become builder constants. No
text graph, external weight lookup, file parsing, or temporary files are needed
for inference. `exporter.to_executor()` creates a CPU context when needed. Pass
`context=your_webnn_context` to select a backend/device.

Every operand is checked immediately against a shape independently inferred by
PyTorch, including inputs, constants, and decomposition intermediates. Each FX
node result is also checked against `torch.export` tensor metadata. A mismatch
raises `ShapeMismatchError` with the node, operation, and both shapes; unsupported
operators raise `NotImplementedError` during construction.

Networks have fixed input shapes and dtypes. Symbolic dimensions, empty tensor
operands, input/buffer mutation, and data-dependent shapes are unsupported.
Use `model.eval()` for inference. Tuple/list/dictionary tensor inputs and outputs
keep their exported structure and order. Outputs are CPU PyTorch tensors.
Some operator options, such as training dropout, pooling indices, arbitrary
einsum equations, and bilinear `align_corners=True`, are unsupported and raise
an error if consumed or requested.

## Optional persistence

```python
text = exporter.to_webnn_text()  # RustNN text with inline constants
exporter.save_to_webnn("model.webnn")  # RustNN text + model.safetensors
exporter.save_weights("torch_weights.safetensors")  # optional PyTorch state snapshot
```

`export_model_with_weights` remains a compatibility helper for saving the RustNN
text and external constant weights; `return_executor=True` wraps the already-built
network. The RustNN saver writes `<stem>.safetensors` beside the text file and
uses `@weights` references. If a different `weights_path` is requested, the helper
also copies that sidecar to the requested path. The exporter constructs operands
through pywebnn and delegates all text formatting and sidecar writing to RustNN.

The `webnn-export program.pt2 model.webnn` command builds a network from a fixed-size
program previously saved with `torch.export.save`.

## License

Apache License (2.0), see LICENSE.

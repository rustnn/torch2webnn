# WebNN PyTorch Exporter

Export PyTorch models to WebNN format using torch dynamo IR.

---

## [WARNING] EXPERIMENTAL - DO NOT USE IN PRODUCTION

This is an early-stage experimental implementation for research and exploration. Many features are incomplete, untested, or may change significantly.

---

## Installation

### For Development

```bash
# Clone your forked repository
git clone https://github.com/<yourusername>/webnn_torch_export.git
cd webnn_torch_export

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
# Optional: Run pytest
pytest
```

### For Use

```bash
pip install webnn_torch_export
```

### Use the Exporter in Your Code

**Export graph only:**

```python
from webnn_torch_export import export_model
import torch
import torch.nn as nn

# Create your model
model = nn.Conv2d(3, 16, kernel_size=3)
input_tensor = torch.randn(1, 3, 28, 28)

# Export with debug output
compiled_model, exporter = export_model(model, input_tensor, debug=True)

# Save exported graph
exporter.save_to_file('my_export.json')

# Access exported graphs programmatically
for graph in exporter.exported_graphs:
    print(graph['nodes'])
```

**Export graph + weights:**

```python
from webnn_torch_export import export_model_with_weights, load_weights_from_safetensors
import torch
import torch.nn as nn

# Create and export your model
model = nn.Sequential(
    nn.Conv2d(3, 16, 3),
    nn.ReLU(),
    nn.Linear(16, 10)
)
input_tensor = torch.randn(1, 3, 28, 28)

# Export both graph and weights
compiled_model, exporter = export_model_with_weights(
    model=model,
    example_input=input_tensor,
    graph_path="model_graph.json",
    weights_path="model_weights.safetensors",
    debug=False
)

# Later: load weights into a fresh model
new_model = nn.Sequential(
    nn.Conv2d(3, 16, 3),
    nn.ReLU(),
    nn.Linear(16, 10)
)
load_weights_from_safetensors(new_model, "model_weights.safetensors")
```

## License

Apache License (2.0) (see LICENSE file)

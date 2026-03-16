# WebNN PyTorch Exporter

Export PyTorch models to WebNN format using torch dynamo IR.

---

## [WARNING] EXPERIMENTAL - DO NOT USE IN PRODUCTION

This is an early-stage experimental implementation for research and exploration. Many features are incomplete, untested, or may change significantly.

---

## Installation

### For Development

```bash
# Clone the repository
git clone https://github.com/yourusername/webnn_torch_export.git
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

### Run Basic Example

```bash
# Using the installed command
webnn-export

# Or run directly
python -m webnn_torch_export.exporter
```

## Key Components

### CustomExporter

The `CustomExporter` class is a Dynamo backend that:
1. Receives FX graphs from PyTorch's compilation process
2. Converts them to a custom format (JSON)
3. Provides debug output to understand graph structure
4. Maintains execution compatibility

**Key methods:**
- `export_graph()`: Main callback that receives FX graphs
- `_convert_fx_to_custom_format()`: Converts FX graph to JSON
- `save_to_file()`: Exports graphs to JSON files

### Test Infrastructure

**Single Operator Tests** (`tests/test_single_ops.py`):
- `test_conv2d_export()`: Tests Conv2d export
- `test_matmul_export()`: Tests matmul export
- `test_linear_export()`: Tests Linear layer export
- `test_conv_with_different_configs()`: Parametrized tests for various Conv2d configurations
- `test_exported_graph_structure()`: Validates exported graph structure

**Integration Tests** (`tests/test_mnist_integration.py`):
- `SimplerMNISTClassifier`: Conv + ReLU + Linear
- `MNISTClassifier`: Full classifier with 2 conv blocks
- `test_simple_mnist_export()`: Exports simple model
- `test_full_mnist_export()`: Exports full model
- `test_mnist_inference_consistency()`: Tests consistency across multiple runs
- `test_mnist_batch_size_invariance()`: Tests with different batch sizes

## How It Works

### 1. Dynamo Backend Registration

```python
def custom_backend(gm: torch.fx.GraphModule, example_inputs):
    # Your export logic here
    return gm

compiled_model = torch.compile(model, backend=custom_backend)
```

### 2. FX Graph Structure

When Dynamo compiles a model, it produces an FX graph with nodes representing:
- **placeholder**: Input tensors
- **call_function**: Function calls (e.g., `torch.relu`, `torch.matmul`)
- **call_module**: Module invocations (e.g., `conv1`, `fc1`)
- **call_method**: Tensor method calls (e.g., `x.flatten()`)
- **output**: Return values

### 3. Export Flow

```
PyTorch Model → torch.compile() → Dynamo → FX Graph → Custom Backend → Export Format
                                                              ↓
                                                    Your Export Logic
```

## Debug Output

With `debug=True`, the exporter prints:
- Complete FX graph representation
- Generated Python code
- Individual node details:
  - Node name and operation type
  - Target function/module
  - Arguments and keyword arguments
  - Tensor metadata (shapes, dtypes)

## Example Output

```
================================================================================
DYNAMO EXPORT CALLBACK TRIGGERED
================================================================================

Graph Module:
graph():
    %x : [num_users=1] = placeholder[target=x]
    %conv1 : [num_users=1] = call_module[target=conv1](args = (%x,), kwargs = {})
    %relu : [num_users=1] = call_function[target=torch.nn.functional.relu](args = (%conv1,), kwargs = {})
    return (relu,)

Node: x
  Op: placeholder
  Target: x
  ...
```

## Exported JSON Format

```json
{
  "nodes": [
    {
      "name": "x",
      "op": "placeholder",
      "target": "x",
      "args": [],
      "kwargs": {}
    },
    {
      "name": "conv1",
      "op": "call_module",
      "target": "conv1",
      "module": "conv1",
      "args": ["x"],
      "kwargs": {}
    }
  ],
  "graph_str": "graph(): ...",
  "code": "def forward(self, x): ..."
}
```

## Extending the Exporter

### Adding New Operator Support

When you export a model with unsupported operations, you'll get a **clear error message** showing exactly what's missing:

```
================================================================================
UNSUPPORTED OPERATION
================================================================================
Operation: layer_norm
Node: layer_norm_output
Target: <function layer_norm at 0x...>
Schema: aten::layer_norm(Tensor input, int[] normalized_shape, ...)
Args: ['input_tensor', '[3072]', 'weight', 'bias', '1e-5']
Kwargs: {}
================================================================================

This operation is not yet supported in WebNN export.
To add support, update webnn_op_mappings.py with a mapping for this operation.
```

This makes it easy to **incrementally add support** for operations as you need them.

**Quick Steps:**

1. **Run your export** - get the error showing the unsupported operation
2. **Add mapping** in `webnn_torch_export/webnn_op_mappings.py`:
   ```python
   TARGET_CONTAINS_TO_CONVERTER: Dict[str, ConverterFn] = {
       # ... existing mappings ...
       "layer_norm": lambda gen, node, output, inputs: gen._convert_layer_norm(node, output, inputs),
   }
   ```
3. **Implement converter** in `webnn_torch_export/webnn_generator.py`:
   ```python
   def _convert_layer_norm(self, node: fx.Node, output: str, inputs: List[str]) -> str:
       """Convert LayerNorm to WebNN"""
       input_tensor = inputs[0] if inputs else 'unknown'
       # ... conversion logic ...
       return f'[{output}] = layerNormalization({input_tensor}, ...);'
   ```
4. **Test** - run export again, repeat for next unsupported operation

**For detailed guidance, see [ADDING_OPS.md](ADDING_OPS.md)** - a comprehensive guide covering:
- How to map PyTorch operations to WebNN
- Common patterns (activations, normalization, matrix ops)
- Step-by-step walkthrough with examples
- WebNN operation reference
- Debugging tips

### Custom Export Format

Modify `_convert_fx_to_custom_format()` to output your desired format:
```python
def _convert_fx_to_custom_format(self, gm):
    # Convert to your format (protobuf, flatbuffer, etc.)
    my_format = convert_to_my_format(gm.graph)
    return my_format
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=webnn_torch_export --cov-report=html

# Run specific markers
pytest -m "not slow"
pytest -m integration
```

### Building the Package

```bash
# Build distribution
python -m build

# Install locally
pip install -e ".[dev]"
```

## Requirements

- PyTorch 2.0+ (for `torch.compile` support)
- Python 3.8+

## Tips for Debugging

1. **Start with `debug=True`** to see full graph output
2. **Use single operator tests** to understand individual operations
3. **Check node metadata** for tensor shapes and types
4. **Verify correctness** by comparing original vs compiled outputs
5. **Examine exported JSON** to understand graph structure

## Next Steps

- Add support for more operators (pooling, normalization, etc.)
- Implement graph optimization passes
- Add serialization to binary formats (protobuf, flatbuffer)
- Handle dynamic shapes
- Support quantized models
- Add execution validation tests

## License

Apache License (2.0) (see LICENSE file)

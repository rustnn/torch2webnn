# Adding ATen Ops

The exporter uses `torch.export.export` which produces an ATen IR graph. Every
`call_function` node has a target like `aten.relu.default`. The dispatch table in
`webnn_op_mappings.py` maps `str(node.target)` → a method name on
`WebNNGraphGenerator`. If the target isn't in the table the node is emitted as a
comment and the `.webnn` file will fail to parse.

## Quick reference: adding an op

**1. Add an entry to `ATEN_OP_TABLE` in `webnn_op_mappings.py`:**

```python
"aten.some_op.overload": "_convert_some_op",
```

**2. Implement `_convert_some_op` in `webnn_generator.py`:**

```python
def _convert_some_op(self, node: fx.Node, output: str, inputs: List[str]) -> str:
    x = inputs[0]
    # node.args holds positional args (first is always the input tensor node)
    # node.kwargs holds keyword args
    return f"[{output}] = webnnOp({x});"
```

The method receives:
- `output` — pre-allocated operand name for this node's result
- `inputs` — operand names for every `fx.Node` in `node.args`, in order
- `node.args` / `node.kwargs` — raw args (scalars, lists, bools) from the ATen schema

Return `None` instead of a string to suppress the node entry entirely (use this for
ops that bake a constant into `inline_constants` and override `node_to_operand`, like
`_convert_arange`).

**3. To find the exact target string for an unfamiliar op:**

```python
ep = torch.export.export(model, (x,))
for n in ep.graph.nodes:
    if n.op == "call_function":
        print(str(n.target), n.args, n.kwargs)
```

---

## Example patterns

### No-op
```python
# webnn_op_mappings.py
"aten.contiguous.default": "_convert_identity",
```

### Direct WebNN equivalent
```python
# webnn_op_mappings.py
"aten.bmm.default": "_convert_matmul",
```
`_convert_matmul` already exists; nothing else needed.

### Parameterised activation (e.g. relu6)
```python
# webnn_op_mappings.py
"aten.relu6.default": "_convert_relu6",

# webnn_generator.py
def _convert_relu6(self, node: fx.Node, output: str, inputs: List[str]) -> str:
    return f"[{output}] = clamp({inputs[0]}, minValue=0.0, maxValue=6.0);"
```

### Scalar-first subtract (`1 - x`)
```python
# webnn_op_mappings.py
"aten.rsub.Scalar": "_convert_rsub_scalar",

# webnn_generator.py
def _convert_rsub_scalar(self, node: fx.Node, output: str, inputs: List[str]) -> str:
    # rsub(x, scalar) = scalar - x
    scalar = node.args[1]
    c = self._create_inline_constant(float(scalar))
    return f"[{output}] = sub({c}, {inputs[0]});"
```

### Op that produces a constant (no graph node needed)
```python
# webnn_op_mappings.py
"aten.full.default": "_convert_full",

# webnn_generator.py — return None to skip nodes {} entry
def _convert_full(self, node: fx.Node, output: str, inputs: List[str]) -> Optional[str]:
    size = list(node.args[0])
    fill = node.args[1]
    dtype = node.kwargs.get("dtype", torch.float32) or torch.float32
    values = torch.full(size, fill, dtype=dtype)
    name = f"const_full_{self.operand_counter}"; self.operand_counter += 1
    self.inline_constants[name] = values
    self.operand_shapes[name] = list(values.shape)
    self.node_to_operand[node.name] = name   # redirect downstream refs
    return None
```

### Multi-step decomposition
```python
def _convert_select_int(self, node: fx.Node, output: str, inputs: List[str]) -> str:
    # aten.select.int(input, dim, index) → slice dim then squeeze
    x = inputs[0]
    in_shape = self._get_node_shape(node.args[0])
    dim = int(node.args[1]) % len(in_shape)
    idx = int(node.args[2])
    starts = [0] * len(in_shape);  starts[dim] = idx
    sizes  = list(in_shape);       sizes[dim]  = 1
    out_shape = in_shape[:dim] + in_shape[dim+1:]
    sliced = f"operand_{self.operand_counter}"; self.operand_counter += 1
    s_str = ", ".join(map(str, starts))
    sz_str = ", ".join(map(str, sizes))
    os_str = ", ".join(map(str, out_shape))
    return (f"[{sliced}] = slice({x}, starts=[{s_str}], sizes=[{sz_str}]);\n"
            f"\t[{output}] = reshape({sliced}, newShape=[{os_str}]);")
```
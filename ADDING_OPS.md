# Adding ATen operations

`torch.export.export` produces ATen `call_function` nodes. The dispatch table in
`webnn_torch_export/webnn_op_mappings.py` maps `str(node.target)` to a method on
`WebNNGraphGenerator`. Unsupported targets fail immediately.

Add the target to `ATEN_OP_TABLE` and implement its converter. Converters receive
`(node, args, kwargs)` with FX references already resolved into `CheckedOperand`
objects. Scalars and nested argument structures are preserved.

```python
# webnn_op_mappings.py
"aten.some_op.default": "_some_op",

# webnn_generator.py

def _some_op(self, node, args, kwargs):
    x = args[0]
    return self._emit("some_webnn_op", torch.some_op, x)
```

Always create intermediate operands through `_emit`, `_constant`, or existing
checked helpers. `_emit` evaluates the reference function on PyTorch meta tensors,
calls the named pywebnn builder method, and compares the resulting operand shape
immediately. Builder options are keyword arguments; capture the corresponding
options in the reference function:

```python
return self._emit(
    "softmax", lambda t: torch.softmax(t, axis), x, axis=axis
)
```

The reference function receives meta tensors in place of operand arguments.
`_emit`'s keyword options are passed only to the WebNN builder. The FX result is
then checked against `node.meta["val"]` before downstream nodes are processed.
Return tuples/lists for multi-output operations; every represented output is
checked. Unsupported auxiliary outputs may be `None` only if unused; consuming
one must raise an error.

For a decomposition, check every step:

```python
sigmoid = self._emit("sigmoid", torch.sigmoid, x)
return self._binary("mul", x, sigmoid)
```

Add an execution test comparing the built network with the original PyTorch
model across relevant shapes/options. Do not compare PyTorch only with its own
exported ATen program. Include a negative test when adding an unsupported option.

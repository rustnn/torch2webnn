"""Execute directly constructed pywebnn graphs with PyTorch tensors."""

import torch
from torch.utils import _pytree


class WebNNExecutor(torch.nn.Module):
    """A PyTorch-like callable sharing the exporter's in-memory network/context."""

    def __init__(self, exporter):
        super().__init__()
        if exporter.context is None:
            import webnn

            exporter.context = webnn.ML().create_context()
        self.context = exporter.context
        self.webnn_graph = exporter.webnn_graph
        generator = exporter.webnn_generator
        self.input_names = list(generator.input_names)
        self.output_names = list(generator.output_names)
        self.input_specs = dict(generator.input_specs)
        self.output_specs = dict(generator.output_specs)
        self.call_spec = exporter.ep.call_spec
        self.scalar_inputs = dict(generator.scalar_inputs)
        from .webnn_generator import webnn_dtype

        self._inputs = {
            name: self.context.create_tensor(
                list(shape), webnn_dtype(dtype), readable=True, writable=True
            )
            for name, (shape, dtype) in self.input_specs.items()
        }
        self._outputs = {
            name: self.context.create_tensor(
                list(shape), webnn_dtype(dtype), readable=True, writable=True
            )
            for name, (shape, dtype) in self.output_specs.items()
        }

    def forward(self, *args, **kwargs):
        user_args, spec = _pytree.tree_flatten((args, kwargs))
        if spec != self.call_spec.in_spec:
            raise ValueError(
                f"Input structure does not match the exported inputs: {spec}"
            )
        for index, expected in self.scalar_inputs.items():
            value = user_args[index]
            if type(value) is not type(expected) or value != expected:
                raise ValueError(
                    f"Input {index} was specialized to {expected!r}, got {value!r}"
                )
        tensors = [value for value in user_args if isinstance(value, torch.Tensor)]
        if len(tensors) != len(self.input_names):
            raise ValueError(
                f"Expected {len(self.input_names)} tensor inputs, got {len(tensors)}"
            )
        for name, tensor in zip(self.input_names, tensors):
            shape, dtype = self.input_specs[name]
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"Input {name}: expected fixed shape {shape}, got {tuple(tensor.shape)}"
                )
            if tensor.dtype != dtype:
                raise ValueError(
                    f"Input {name}: expected dtype {dtype}, got {tensor.dtype}"
                )
            self.context.write_tensor(
                self._inputs[name], tensor.detach().cpu().contiguous().numpy()
            )
        # Explicit fixed-size tensors also support scalar outputs, which some
        # pywebnn versions cannot infer in the convenience compute() path.
        self.context.dispatch(self.webnn_graph, self._inputs, self._outputs)
        leaves = []
        for name in self.output_names:
            shape, dtype = self.output_specs[name]
            value = torch.from_numpy(self.context.read_tensor(self._outputs[name]))
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"Output {name}: expected shape {shape}, got {tuple(value.shape)}"
                )
            # pywebnn may construct NumPy outputs from Python lists, which widen
            # float32/float16 to float64. Restore the graph descriptor's dtype.
            leaves.append(value.to(dtype))
        return _pytree.tree_unflatten(leaves, self.call_spec.out_spec)

    def __repr__(self):
        return f"WebNNExecutor(inputs={self.input_names}, outputs={self.output_names})"

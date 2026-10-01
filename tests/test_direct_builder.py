"""Regressions for direct graph construction and immediate operand validation."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import webnn

from webnn_torch_export import (
    CustomExporter,
    ShapeMismatchError,
    export_model,
    export_model_with_weights,
)
from .conftest import assert_export_matches
from .models import SingleGroupNorm, SingleGELU, CastToFloat16AndBack


@pytest.mark.parametrize(
    "model,shape",
    [
        (SingleGroupNorm(), (2, 4, 3, 5)),
        (SingleGELU(), (2, 5)),
        (CastToFloat16AndBack(), (2, 5)),
    ],
)
def test_decompositions_and_casts(model, shape):
    assert_export_matches(model, torch.randn(shape), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "operation", ["input", "constant", "transpose", "matmul", "add"]
)
def test_mismatch_fails_at_operand_creation(operation):
    context = webnn.ML().create_context(device_type="cpu")
    calls = []

    class BuilderProxy:
        def __init__(self):
            self.builder = context.create_graph_builder()

        def __getattr__(self, name):
            method = getattr(self.builder, name)

            def call(*args, **kwargs):
                calls.append(name)
                value = method(*args, **kwargs)
                if name == operation:
                    return SimpleNamespace(shape=[999], data_type=value.data_type)
                return value

            return call

    proxy = SimpleNamespace(create_graph_builder=BuilderProxy)
    with pytest.raises(
        ShapeMismatchError, match=r"PyTorch shape .* != pywebnn shape \(999,\)"
    ):
        export_model(torch.nn.Linear(3, 4).eval(), torch.randn(2, 3), context=proxy)
    assert calls[-1] == operation
    assert "build" not in calls


def test_fx_metadata_mismatch_fails_before_build():
    ep = torch.export.export(torch.nn.ReLU(), (torch.randn(2, 3),))
    relu = next(n for n in ep.graph.nodes if n.op == "call_function")
    relu.meta["val"] = torch.empty(2, 4, device="meta")
    with pytest.raises(ShapeMismatchError, match="aten.relu.default"):
        CustomExporter(ep)


def test_dynamic_shapes_rejected():
    ep = torch.export.export(
        torch.nn.ReLU(),
        (torch.randn(3, 4),),
        dynamic_shapes={"input": {0: torch.export.Dim("batch", min=1, max=8)}},
    )
    with pytest.raises(ValueError, match="Dynamic shapes"):
        CustomExporter(ep)


def test_fixed_input_shape_dtype_and_structure():
    exporter, _ = export_model(torch.nn.ReLU(), torch.randn(2, 3))
    executor = exporter.to_executor()
    with pytest.raises(ValueError, match="fixed shape"):
        executor(torch.randn(4, 3))
    with pytest.raises(ValueError, match="dtype"):
        executor(torch.randn(2, 3, dtype=torch.float16))
    with pytest.raises(ValueError, match="structure"):
        executor()


def test_input_order_and_nested_duplicate_outputs():
    class Model(torch.nn.Module):
        def forward(self, z, a):
            return {"difference": z - a, "other": [z, (a, z)]}

    _, exporter = assert_export_matches(
        Model(), (torch.full((2, 3), 7.0), torch.full((2, 3), 2.0))
    )
    assert exporter.webnn_generator.input_names == ["z", "a"]


def test_nested_inputs_and_specialized_scalar():
    class Model(torch.nn.Module):
        def forward(self, pair, factor):
            return pair["z"] + pair["a"] * factor

    inputs = ({"z": torch.randn(2, 3), "a": torch.randn(2, 3)}, 2.0)
    executor, _ = assert_export_matches(Model(), inputs)
    with pytest.raises(ValueError, match="specialized"):
        executor(inputs[0], 3.0)


def test_integer_inputs_preserve_dtype():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 2

    assert_export_matches(Model(), torch.arange(6, dtype=torch.int32).reshape(2, 3))


def test_vector_dot_scalar_shape():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x @ y

    assert_export_matches(Model(), (torch.randn(4), torch.randn(4)))


@pytest.mark.parametrize("causal", [False, True])
def test_attention_masks(causal):
    class Model(torch.nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=0.25)

    assert_export_matches(
        Model(), tuple(torch.randn(1, 2, 4, 8) for _ in range(3)), rtol=1e-4, atol=1e-4
    )


def test_strided_negative_slice():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x[:, -5:-1:2]

    assert_export_matches(Model(), torch.randn(3, 8))


def test_unknown_operator_rejected():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.sort(x).values

    with pytest.raises(NotImplementedError, match="aten.sort"):
        export_model(Model(), torch.randn(2, 3))


def test_unused_pool_indices_supported_consumed_indices_rejected():
    class Model(torch.nn.Module):
        def forward(self, x):
            return F.max_pool2d(x, 2, return_indices=True)

    with pytest.raises(NotImplementedError, match="auxiliary output"):
        export_model(Model(), torch.randn(1, 2, 4, 4))


def test_optional_serialization_uses_built_graph(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Graph must not be reloaded to execute")

    monkeypatch.setattr(webnn.MLGraph, "load", forbidden)
    model = torch.nn.Linear(3, 4).eval()
    x = torch.randn(2, 3)
    path = tmp_path / "model.webnn"
    executor, _ = export_model_with_weights(
        model, x, path, tmp_path / "weights.safetensors", return_executor=True
    )
    assert "webnn_graph" in path.read_text()
    assert path.with_suffix(".safetensors").is_file()
    torch.testing.assert_close(executor(x), model(x), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
def test_noninteger_resize_scale(mode):
    class Model(torch.nn.Module):
        def forward(self, x):
            options = {"align_corners": False} if mode == "bilinear" else {}
            return F.interpolate(x, scale_factor=(1.7, 2.3), mode=mode, **options)

    assert_export_matches(Model(), torch.randn(1, 2, 5, 7), rtol=1e-4, atol=1e-4)


def test_cli_builds_saved_program(tmp_path, monkeypatch):
    from webnn_torch_export.exporter import main

    ep = torch.export.export(torch.nn.ReLU(), (torch.randn(2, 3),))
    source, destination = tmp_path / "program.pt2", tmp_path / "model.webnn"
    torch.export.save(ep, source)
    monkeypatch.setattr("sys.argv", ["webnn-export", str(source), str(destination)])
    main()
    assert "webnn_graph" in destination.read_text()
    assert destination.with_suffix(".safetensors").is_file()


def test_lifted_tensor_constant():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x + torch.tensor([1.0, 2.0, 3.0])

    assert_export_matches(Model(), torch.randn(2, 3))


def test_empty_tensor_rejected():
    with pytest.raises(ValueError, match="positive tensor dimensions"):
        export_model(torch.nn.ReLU(), torch.empty(0, 3))


def test_native_text_and_sidecar_roundtrip(tmp_path):
    model = torch.nn.Linear(3, 4).eval()
    x = torch.randn(2, 3)
    exporter, _ = export_model(model, x)
    text = exporter.to_webnn_text()
    assert "webnn_graph" in text and "matmul(" in text
    assert exporter.context is None
    path = tmp_path / "model.webnn"
    exporter.save_to_webnn(path)
    assert "@weights(" in path.read_text()
    loaded = webnn.MLGraph.load(str(path))
    context = webnn.ML().create_context(device_type="cpu", backend="onnx")
    outputs = context.compute(
        loaded, {exporter.webnn_generator.input_names[0]: x.numpy()}
    )
    actual = torch.from_numpy(outputs["output_0"]).to(torch.float32)
    torch.testing.assert_close(actual, model(x), rtol=1e-4, atol=1e-4)


def test_export_needs_no_runtime(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Export must not construct a runtime context")

    monkeypatch.setattr(webnn.ML, "create_context", forbidden)
    exporter, _ = export_model(torch.nn.Linear(3, 4).eval(), torch.randn(2, 3))
    assert exporter.context is None
    assert "matmul(" in exporter.to_webnn_text()

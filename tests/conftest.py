"""
Pytest configuration, fixtures, and shared helpers for webnn_torch_export tests.
"""

import pytest
import torch
import tempfile
import os
from typing import Union, Tuple


@pytest.fixture
def seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(42)
    return 42


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def assert_export_matches(
    model: torch.nn.Module,
    example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    rtol: float = 1e-5,
    atol: float = 1e-5,
    debug: bool = False,
):
    """
    Export model via torch.compile backend and assert outputs match original PyTorch.

    Args:
        model: PyTorch model to export
        example_input: Input tensor or tuple of tensors
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        debug: Whether to print debug information

    Returns:
        Tuple of (compiled_model, exporter)
    """
    from webnn_torch_export import export_model

    model.eval()
    with torch.no_grad():
        expected = model(*example_input) if isinstance(example_input, tuple) else model(example_input)

    exporter, ep = export_model(model, example_input, debug=debug)

    # ep.module() runs the exported ATen graph — should match the original model
    exported_callable = ep.module()
    with torch.no_grad():
        actual = exported_callable(*example_input) if isinstance(example_input, tuple) else exported_callable(example_input)

    if not torch.allclose(expected, actual, rtol=rtol, atol=atol):
        max_diff = torch.max(torch.abs(expected - actual)).item()
        raise AssertionError(
            f"Exported output doesn't match PyTorch\n"
            f"  Max diff: {max_diff:.2e}  (rtol={rtol}, atol={atol})"
        )

    return exported_callable, exporter


def assert_generates_webnn(
    model: torch.nn.Module,
    example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    debug: bool = False,
) -> str:
    """
    Export a model and generate its WebNN graph text, returning the text.

    Does NOT require the WebNN runtime — only exercises the generator.
    Useful for asserting that the generator produces valid output for a given op.
    """
    from webnn_torch_export import export_model
    import tempfile
    import os

    if not isinstance(example_input, tuple):
        example_input = (example_input,)

    model.eval()
    exporter, _ = export_model(model, example_input, debug=debug)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".webnn", delete=False) as f:
        path = f.name
    try:
        exporter.save_to_webnn(path)
        with open(path) as f:
            return f.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def validate_webnn_execution(
    model: torch.nn.Module,
    example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    debug: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    """
    Helper function to validate WebNN export and execution.

    This function:
    1. Exports the model to WebNN format
    2. Creates a WebNNExecutor
    3. Runs inference through WebNN runtime
    4. Compares outputs with PyTorch
    5. Cleans up temporary files

    Args:
        model: PyTorch model to export and validate
        example_input: Example input tensor(s)
        debug: Whether to print debug information
        rtol: Relative tolerance for output comparison
        atol: Absolute tolerance for output comparison

    Returns:
        Tuple of (webnn_executor, exporter) if successful

    Raises:
        AssertionError: If WebNN output doesn't match PyTorch output
        ImportError: If WebNN runtime is not available (test will be skipped)
    """
    from webnn_torch_export import export_model_with_weights, WebNNExecutor

    # Check if WebNN is available
    if WebNNExecutor is None:
        pytest.skip("WebNN runtime not available")

    # Create temporary files for WebNN graph and weights
    with tempfile.NamedTemporaryFile(mode='w', suffix='.webnn', delete=False) as webnn_file:
        webnn_path = webnn_file.name

    with tempfile.NamedTemporaryFile(mode='wb', suffix='.safetensors', delete=False) as weights_file:
        weights_path = weights_file.name

    try:
        # Get expected PyTorch output
        model.eval()
        with torch.no_grad():
            if isinstance(example_input, tuple):
                expected_output = model(*example_input)
            else:
                expected_output = model(example_input)

        # Export to WebNN with executor
        result, ep_or_exporter = export_model_with_weights(
            model,
            example_input,
            webnn_path=webnn_path,
            weights_path=weights_path,
            debug=debug,
            return_executor=True
        )
        executor = result
        exporter = ep_or_exporter

        # Verify executor was created
        assert executor is not None, "WebNNExecutor was not created"
        assert hasattr(executor, 'forward'), "WebNNExecutor missing forward method"

        # Run inference through WebNN
        with torch.no_grad():
            if isinstance(example_input, tuple):
                webnn_output = executor(*example_input)
            else:
                webnn_output = executor(example_input)

        # Compare outputs
        if not isinstance(expected_output, tuple):
            webnn_output = (webnn_output,)
            expected_output = (expected_output,)

        assert len(expected_output) == len(webnn_output), "Output count mismatch"
        for i, (exp, actual) in enumerate(zip(expected_output, webnn_output)):
            if exp.dtype != actual.dtype:
                actual = actual.to(exp.dtype)
            if not torch.allclose(exp, actual, rtol=rtol, atol=atol):
                max_diff = torch.max(torch.abs(exp - actual)).item()
                mean_diff = torch.mean(torch.abs(exp - actual)).item()
                raise AssertionError(
                    f"WebNN output {i} doesn't match PyTorch output\n"
                    f"  Max difference: {max_diff:.2e}\n"
                    f"  Mean difference: {mean_diff:.2e}\n"
                    f"  Tolerance: rtol={rtol}, atol={atol}\n"
                    f"  Expected shape: {exp.shape}, dtype: {exp.dtype}\n"
                    f"  Actual shape: {actual.shape}, dtype: {actual.dtype}"
                )

        return executor, exporter
    except Exception as e:
        print(f"WebNN validation failed: {e}")
        webnn_graph = open(webnn_path, 'rb').read().decode('utf-8')
        print(f"WebNN graph:\n{webnn_graph}")
        raise e
    finally:
        # Clean up temporary files
        if os.path.exists(webnn_path):
            os.unlink(webnn_path)
        if os.path.exists(weights_path):
            os.unlink(weights_path)

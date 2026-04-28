"""
Export the Flux Klein 4B flow model to WebNN.

Usage:
    python export_flow.py
"""

import os
import torch
import numpy as np
from webnn_torch_export import export_model_with_weights
from flux2.util import load_flow_model
from _common import hf_login, OUTPUT_DIR


def main():
    hf_login()
    torch.manual_seed(42)
    device = "cpu"

    print("=" * 60)
    print("Flux Klein 4B — Flow Model Export")
    print("=" * 60)

    print("\nLoading flow model...")
    try:
        flow_model = load_flow_model("flux.2-klein-4b", debug_mode=True, device=device)
        flow_model.eval()
    except SystemExit:
        print("Could not load weights from HuggingFace — using placeholder model.")
        from flux2.model import Flux2, Klein4BParams
        with torch.device(device):
            flow_model = Flux2(Klein4BParams()).to(torch.bfloat16)
        flow_model.eval()

    total_params = sum(p.numel() for p in flow_model.parameters())
    print(f"\nFlow Model Statistics:")
    print(f"  Total parameters : {total_params:,}")
    print(f"  Model size (fp32): ~{total_params * 4 / 1024**2:.1f} MB")

    # Sample inputs
    batch_size    = 1
    img_seq_len   = 64
    txt_seq_len   = 32
    inf_dtype     = torch.bfloat16

    sample_x         = torch.randn(batch_size, img_seq_len, 128, dtype=inf_dtype)
    sample_x_ids     = torch.zeros(batch_size, img_seq_len, 4, dtype=torch.long)
    sample_timesteps = torch.tensor([0.5], dtype=inf_dtype)
    sample_ctx       = torch.randn(batch_size, txt_seq_len, 7680, dtype=inf_dtype)
    sample_ctx_ids   = torch.zeros(batch_size, txt_seq_len, 4, dtype=torch.long)
    sample_guidance  = torch.tensor([1.0], dtype=inf_dtype)

    print("\nInput shapes:")
    for name, t in [("x", sample_x), ("x_ids", sample_x_ids),
                    ("timesteps", sample_timesteps), ("ctx", sample_ctx),
                    ("ctx_ids", sample_ctx_ids), ("guidance", sample_guidance)]:
        print(f"  {name}: {list(t.shape)}")

    print("\nRunning forward pass...")
    with torch.no_grad():
        output = flow_model(
            x=sample_x, x_ids=sample_x_ids, timesteps=sample_timesteps,
            ctx=sample_ctx, ctx_ids=sample_ctx_ids, guidance=sample_guidance
        )
    print(f"Output shape: {output.shape}")

    webnn_path   = os.path.join(OUTPUT_DIR, "flux_klein_flow.webnn")
    weights_path = os.path.join(OUTPUT_DIR, "flux_klein_flow_weights.safetensors")

    print("\nExporting to WebNN...")
    try:
        export_model_with_weights(
            model=flow_model,
            example_input=(sample_x, sample_x_ids, sample_timesteps,
                           sample_ctx, sample_ctx_ids, sample_guidance),
            webnn_path=webnn_path,
            weights_path=weights_path,
            graph_name="flux_klein_4b_flow",
            debug=False,
        )
        print(f"Export successful!")
        print(f"  Graph  : {webnn_path}  ({os.path.getsize(webnn_path) / 1024:.1f} KB)")
        print(f"  Weights: {weights_path}  ({os.path.getsize(weights_path) / 1024:.1f} KB)")

        # Validate against WebNN runtime if available
        try:
            import webnn
            print("\nValidating against WebNN runtime...")
            context = webnn.ML().create_context(device_type="auto")
            webnn_graph = webnn.MLGraph.load(webnn_path, weights_path=weights_path)

            # Build input dict — convert bfloat16 to float32 for WebNN
            inputs = {
                "x":         sample_x.float().detach().cpu().numpy(),
                "x_ids":     sample_x_ids.detach().cpu().numpy(),
                "timesteps": sample_timesteps.float().detach().cpu().numpy(),
                "ctx":       sample_ctx.float().detach().cpu().numpy(),
                "ctx_ids":   sample_ctx_ids.detach().cpu().numpy(),
                "guidance":  sample_guidance.float().detach().cpu().numpy(),
            }

            webnn_output = context.compute(webnn_graph, inputs)
            output_name  = webnn_graph.get_output_names()[0]
            torch_output = output.float().detach().cpu().numpy()

            mae = float(np.mean(np.abs(webnn_output[output_name] - torch_output)))
            print(f"WebNN vs Torch MAE: {mae:.6f}")
        except ImportError:
            print("webnn package not available — skipping runtime validation.")

    except NotImplementedError as e:
        print(f"Unsupported operation:\n{e}")
    except Exception as e:
        import traceback
        print(f"Export error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""
Export the Flux Klein AutoEncoder decoder to WebNN.

Usage:
    python export_decoder.py
"""

import os
import torch
import numpy as np
from webnn_torch_export import export_model_with_weights
from flux2.util import load_ae
from _common import hf_login, OUTPUT_DIR


class DecoderOnly(torch.nn.Module):
    """Wraps the AutoEncoder so only the decode path is exported."""

    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, z):
        return self.ae.decode(z)


def main():
    hf_login()
    torch.manual_seed(42)
    device = "cpu"

    print("=" * 60)
    print("Flux Klein — AutoEncoder Decoder Export")
    print("=" * 60)

    print("\nLoading autoencoder...")
    try:
        autoencoder = load_ae("flux.2-klein-4b", device=device)
        autoencoder.eval()
    except SystemExit:
        print("Could not load weights from HuggingFace — using placeholder model.")
        from flux2.autoencoder import AutoEncoder, AutoEncoderParams
        with torch.device(device):
            autoencoder = AutoEncoder(AutoEncoderParams())
        autoencoder.eval()

    ae_params = sum(p.numel() for p in autoencoder.parameters())
    print(f"\nAutoEncoder Statistics:")
    print(f"  Total parameters : {ae_params:,}")
    print(f"  Model size (fp32): ~{ae_params * 4 / 1024**2:.1f} MB")

    sample_latent = torch.randn(1, 128, 32, 32)
    print(f"\nInput latent shape: {list(sample_latent.shape)}")

    with torch.no_grad():
        decoded_image = autoencoder.decode(sample_latent)
    print(f"Decoded image shape: {list(decoded_image.shape)}")

    decoder_model = DecoderOnly(autoencoder)

    webnn_path   = os.path.join(OUTPUT_DIR, "flux_klein_decoder.webnn")
    weights_path = os.path.join(OUTPUT_DIR, "flux_klein_decoder_weights.safetensors")

    print("\nExporting decoder to WebNN...")
    try:
        export_model_with_weights(
            model=decoder_model,
            example_input=sample_latent,
            webnn_path=webnn_path,
            weights_path=weights_path,
            graph_name="flux_klein_decoder",
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
            input_name  = webnn_graph.get_input_names()[0]
            output_name = webnn_graph.get_output_names()[0]
            webnn_output = context.compute(
                webnn_graph,
                {input_name: sample_latent.detach().cpu().numpy().astype(np.float32)},
            )[output_name]
            torch_output = decoded_image.detach().cpu().numpy()
            mae = float(np.mean(np.abs(webnn_output - torch_output)))
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
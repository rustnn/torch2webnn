"""
Export the Flux Klein AutoEncoder (encoder and/or decoder) to WebNN.

The autoencoder is the entry and exit point for image-to-image workflows:

    img2img pipeline
    ─────────────────────────────────────────────────────────────
    source image  ──[ Encoder ]──► latent
                                      │
                               add noise, blend
                                      │
                                  ◄──────────────────────────────
                                  ▲   flow model (4-50 steps)   │
                                  └──────────────────────────────┘
                                      │
                              denoised latent
                                      │
                              ──[ Decoder ]──► output image
    ─────────────────────────────────────────────────────────────

For text-to-image the encoder is not used; a random latent is denoised instead.

Usage:
    python export_autoencoder.py                  # export both
    python export_autoencoder.py --mode encoder
    python export_autoencoder.py --mode decoder
"""

import argparse
import os

import numpy as np
import torch

from webnn_torch_export import export_model_with_weights
from flux2.util import load_ae
from _common import hf_login, OUTPUT_DIR


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------

class EncoderOnly(torch.nn.Module):
    """Wraps the AutoEncoder so only the encode path is exported.

    Input:  RGB image  [batch, 3, H, W]   (H = W = 512 for Flux Klein)
    Output: Latent     [batch, 128, H/16, W/16]
    """

    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        return self.ae.encode(x)


class DecoderOnly(torch.nn.Module):
    """Wraps the AutoEncoder so only the decode path is exported.

    Input:  Latent     [batch, 128, H/16, W/16]
    Output: RGB image  [batch, 3, H, W]
    """

    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, z):
        return self.ae.decode(z)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _file_size_kb(path: str) -> float:
    return os.path.getsize(path) / 1024


def _validate_webnn(webnn_path: str, weights_path: str, sample_input: torch.Tensor,
                    torch_output: torch.Tensor) -> None:
    """Run the exported graph through the WebNN runtime and report MAE."""
    try:
        import webnn
        context = webnn.ML().create_context(device_type="auto")
        graph = webnn.MLGraph.load(webnn_path, weights_path=weights_path)
        input_name = graph.get_input_names()[0]
        output_name = graph.get_output_names()[0]
        webnn_out = context.compute(
            graph,
            {input_name: sample_input.detach().cpu().numpy().astype(np.float32)},
        )[output_name]
        mae = float(np.mean(np.abs(webnn_out - torch_output.detach().cpu().numpy())))
        print(f"    WebNN vs Torch MAE: {mae:.6f}")
    except ImportError:
        print("    webnn package not available — skipping runtime validation.")


def export_encoder(autoencoder, sample_image: torch.Tensor) -> None:
    print("\n── Encoder ──────────────────────────────────────────────")
    print(f"  Input  (image):  {list(sample_image.shape)}")

    with torch.no_grad():
        torch_latent = autoencoder.encode(sample_image)
    print(f"  Output (latent): {list(torch_latent.shape)}")

    encoder_model = EncoderOnly(autoencoder)

    webnn_path   = os.path.join(OUTPUT_DIR, "flux_klein_encoder.webnn")
    weights_path = os.path.join(OUTPUT_DIR, "flux_klein_encoder_weights.safetensors")

    try:
        export_model_with_weights(
            model=encoder_model,
            example_input=sample_image,
            webnn_path=webnn_path,
            weights_path=weights_path,
            graph_name="flux_klein_encoder",
        )
        print(f"  Graph  : {webnn_path}  ({_file_size_kb(webnn_path):.1f} KB)")
        print(f"  Weights: {weights_path}  ({_file_size_kb(weights_path):.1f} KB)")
        _validate_webnn(webnn_path, weights_path, sample_image, torch_latent)
    except NotImplementedError as e:
        print(f"  Unsupported operation: {e}")
    except Exception as e:
        import traceback
        print(f"  Export error: {e}")
        traceback.print_exc()


def export_decoder(autoencoder, sample_latent: torch.Tensor) -> None:
    print("\n── Decoder ──────────────────────────────────────────────")
    print(f"  Input  (latent): {list(sample_latent.shape)}")

    with torch.no_grad():
        torch_image = autoencoder.decode(sample_latent)
    print(f"  Output (image):  {list(torch_image.shape)}")

    decoder_model = DecoderOnly(autoencoder)

    webnn_path   = os.path.join(OUTPUT_DIR, "flux_klein_decoder.webnn")
    weights_path = os.path.join(OUTPUT_DIR, "flux_klein_decoder_weights.safetensors")

    try:
        export_model_with_weights(
            model=decoder_model,
            example_input=sample_latent,
            webnn_path=webnn_path,
            weights_path=weights_path,
            graph_name="flux_klein_decoder",
        )
        print(f"  Graph  : {webnn_path}  ({_file_size_kb(webnn_path):.1f} KB)")
        print(f"  Weights: {weights_path}  ({_file_size_kb(weights_path):.1f} KB)")
        _validate_webnn(webnn_path, weights_path, sample_latent, torch_image)
    except NotImplementedError as e:
        print(f"  Unsupported operation: {e}")
    except Exception as e:
        import traceback
        print(f"  Export error: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(mode: str = "both") -> None:
    hf_login()
    torch.manual_seed(42)

    print("=" * 60)
    print("Flux Klein — AutoEncoder Export")
    print(f"Mode: {mode}")
    print("=" * 60)

    print("\nLoading autoencoder...")
    try:
        autoencoder = load_ae("flux.2-klein-4b", device="cpu")
        autoencoder.eval()
    except SystemExit:
        print("Could not load weights from HuggingFace — using placeholder model.")
        from flux2.autoencoder import AutoEncoder, AutoEncoderParams
        with torch.device("cpu"):
            autoencoder = AutoEncoder(AutoEncoderParams())
        autoencoder.eval()

    n_params = sum(p.numel() for p in autoencoder.parameters())
    print(f"  Parameters : {n_params:,}")
    print(f"  Size (fp32): ~{n_params * 4 / 1024**2:.1f} MB")

    # Canonical shapes from ARCHITECTURE.md
    sample_image  = torch.randn(1, 3, 512, 512)   # RGB image
    sample_latent = torch.randn(1, 128, 32, 32)   # 16× compressed latent

    if mode in ("encoder", "both"):
        export_encoder(autoencoder, sample_image)

    if mode in ("decoder", "both"):
        export_decoder(autoencoder, sample_latent)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Flux Klein AutoEncoder to WebNN")
    parser.add_argument(
        "--mode",
        choices=["encoder", "decoder", "both"],
        default="both",
        help="Which part of the autoencoder to export (default: both)",
    )
    args = parser.parse_args()
    main(mode=args.mode)

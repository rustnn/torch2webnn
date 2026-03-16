"""
Flux Klein Export Demo

Demonstrates exporting a more complex multi-component diffusion model:
- Text Encoder (Qwen3-4B)
- Flow Model (Flux Klein 4B)
- AutoEncoder (VAE)

This example shows how to handle models with multiple sub-components,
each of which can be exported separately.
"""

import torch
import os
import numpy as np
from webnn_torch_export import export_model_with_weights
from flux2.util import load_flow_model, load_ae
from flux2.sampling import get_schedule, batched_prc_txt, batched_prc_img, denoise
import webnn

hf_token = os.environ.get("HF_TOKEN")

# Login to HuggingFace if token is provided
if hf_token:
    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print(f"Logged in to HuggingFace with token")
    except Exception as e:
        print(f"Note: Could not login to HuggingFace: {e}")

def main():
    torch.manual_seed(42)
    device = "cpu"  # Using CPU for export compatibility

    print("=" * 60)
    print("FLUX KLEIN 4B - WebNN Export Demo")
    print("=" * 60)
    print("\nThis example demonstrates exporting a complex diffusion model")
    print("with multiple components: text encoder, flow model, and autoencoder.\n")

    # Model selection
    model_name = "flux.2-klein-4b"
    print(f"Loading model: {model_name}")
    print("Note: First run will download ~8GB of model weights from HuggingFace")

    # =========================================================================
    # PART 1: Load Text Encoder
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 1: Text Encoder (Qwen3-4B)")
    print("=" * 60)

    print("\nLoading text encoder...")
    # text_encoder = load_text_encoder(model_name, device=device)
    # text_encoder.eval()

    # Sample text input
    sample_prompt = "a cat wearing sunglasses"
    print(f"\nSample prompt: '{sample_prompt}'")

    # Encode text (simplified - real usage would involve tokenization)
    # For demo purposes, we'll create a dummy encoded text tensor
    # Real text encoding happens inside text_encoder with proper tokenization
    print("\nSkipping text encoder export (uses Qwen3 - very large model)")
    print("In production, you'd export this separately or use a pre-encoded cache")
    output_dir = os.path.dirname(__file__)

    # =========================================================================
    # PART 2: Load Flow Model (Main Diffusion Model)
    # =========================================================================
    export_flow_model = True
    export_ae_model = False
    if export_flow_model:
        print("\n" + "=" * 60)
        print("PART 2: Flow Model (Flux Klein 4B)")
        print("=" * 60)

        print("\nLoading flow model...")
        try:
            flow_model = load_flow_model(model_name, debug_mode=True, device=device)
            flow_model.eval()
        except SystemExit:
            print("\n" + "!" * 60)
            print("ERROR: Could not load Flux Klein model")
            print("!" * 60)
            print("\nThis example requires access to the HuggingFace model repository.")
            print("Please ensure:")
            print("  1. You have a HuggingFace account")
            print("  2. You've accepted the model license at:")
            print("     https://huggingface.co/black-forest-labs/FLUX.2-klein-4B")
            print("  3. You're logged in: huggingface-cli login")
            print("\nAlternatively, set environment variables to use local weights:")
            print("  export KLEIN_4B_MODEL_PATH=/path/to/flux-2-klein-4b.safetensors")
            print("  export AE_MODEL_PATH=/path/to/ae.safetensors")
            print("\nFor now, this demo will continue with a placeholder model...")
            print("!" * 60)

            # Create a minimal placeholder model for demonstration
            from flux2.model import Flux2, Klein4BParams
            with torch.device(device):
                flow_model = Flux2(Klein4BParams()).to(torch.bfloat16)
            flow_model.eval()
            print("Using minimal placeholder model (not the real Flux Klein weights)")

        # Count parameters
        total_params = sum(p.numel() for p in flow_model.parameters())
        trainable_params = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
        print(f"\nFlow Model Statistics:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Model size (fp32): ~{total_params * 4 / 1024**2:.1f} MB")

        # Create sample inputs for the flow model
        # The flow model takes:
        # - x: latent image tensor [batch, seq_len, channels]
        # - x_ids: position IDs for latents
        # - timesteps: diffusion timestep
        # - ctx: text embeddings [batch, text_seq_len, text_dim]
        # - ctx_ids: position IDs for text
        # - guidance: guidance scale

        batch_size = 1
        img_seq_len = 64  # Small for demo (real: ~1024 for 512x512)
        txt_seq_len = 32  # Small for demo (real: ~256)

        # Note: these dimensions match Klein4BParams
        latent_channels = 128
        text_embed_dim = 7680  # context_in_dim for Klein 4B
        inf_dtype = torch.bfloat16
        sample_x = torch.randn(batch_size, img_seq_len, latent_channels, dtype=inf_dtype)
        sample_x_ids = torch.zeros(batch_size, img_seq_len, 4, dtype=torch.long)  # [t, h, w, l]
        sample_timesteps = torch.tensor([0.5], dtype=inf_dtype)  # Mid diffusion timestep
        sample_ctx = torch.randn(batch_size, txt_seq_len, text_embed_dim, dtype=inf_dtype)
        sample_ctx_ids = torch.zeros(batch_size, txt_seq_len, 4, dtype=torch.long)
        sample_guidance = torch.tensor([1.0], dtype=inf_dtype)  # Klein 4B uses guidance=1.0

        print("\nInput shapes:")
        print(f"  - Latent image (x): {list(sample_x.shape)}")
        print(f"  - Image position IDs (x_ids): {list(sample_x_ids.shape)}")
        print(f"  - Timesteps: {list(sample_timesteps.shape)}")
        print(f"  - Text embeddings (ctx): {list(sample_ctx.shape)}")
        print(f"  - Text position IDs (ctx_ids): {list(sample_ctx_ids.shape)}")
        print(f"  - Guidance: {list(sample_guidance.shape)}")

        # Run inference
        print("\nRunning flow model inference...")
        with torch.no_grad():
            output = flow_model(
                x=sample_x,
                x_ids=sample_x_ids,
                timesteps=sample_timesteps,
                ctx=sample_ctx,
                ctx_ids=sample_ctx_ids,
                guidance=sample_guidance
            )
        print(f"Output shape: {output.shape}")

        # Export flow model
        flow_weights_path = os.path.join(output_dir, "flux_klein_flow_weights.safetensors")
        flow_webnn_path = os.path.join(output_dir, "flux_klein_flow.webnn")

        print("\n" + "-" * 60)
        print("Exporting Flow Model to WebNN...")
        print("-" * 60)

        try:
            compiled_model, exporter = export_model_with_weights(
                model=flow_model,
                example_input=(sample_x, sample_x_ids, sample_timesteps,
                              sample_ctx, sample_ctx_ids, sample_guidance),
                webnn_path=flow_webnn_path,
                weights_path=flow_weights_path,
                graph_name="flux_klein_4b_flow",
                debug=False
            )
            print("Flow model export successful!")
        except NotImplementedError as e:
            print("Flow model export hit an unsupported operation:")
            print(str(e))
            print("\nThis is expected for complex models like Flux Klein.")
            print("You can incrementally add support for each operation in webnn_op_mappings.py")
        except Exception as e:
            print(f"Flow model export encountered an error: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # PART 3: Load AutoEncoder
    # =========================================================================
    if export_ae_model:
        print("\n" + "=" * 60)
        print("PART 3: AutoEncoder (VAE)")
        print("=" * 60)

        print("\nLoading autoencoder...")
        try:
            autoencoder = load_ae(model_name, device=device)
            autoencoder.eval()
        except SystemExit:
            print("Could not load autoencoder weights from HuggingFace.")
            print("Using placeholder model without pretrained weights...")
            from flux2.autoencoder import AutoEncoder, AutoEncoderParams
            with torch.device(device):
                autoencoder = AutoEncoder(AutoEncoderParams())
            autoencoder.eval()

        ae_params = sum(p.numel() for p in autoencoder.parameters())
        print(f"\nAutoEncoder Statistics:")
        print(f"  - Total parameters: {ae_params:,}")
        print(f"  - Model size (fp32): ~{ae_params * 4 / 1024**2:.1f} MB")

        # The autoencoder has two main functions:
        # 1. encode: RGB image -> latent
        # 2. decode: latent -> RGB image

        # Test decoder (latent -> image)
        print("\nTesting decoder...")
        sample_latent = torch.randn(1, 128, 32, 32)  # [batch, channels, h, w]
        print(f"Input latent shape: {list(sample_latent.shape)}")

        with torch.no_grad():
            decoded_image = autoencoder.decode(sample_latent)
        print(f"Decoded image shape: {list(decoded_image.shape)}")

        # Export decoder
        decoder_weights_path = os.path.join(output_dir, "flux_klein_decoder_weights.safetensors")
        decoder_webnn_path = os.path.join(output_dir, "flux_klein_decoder.webnn")

        print("\n" + "-" * 60)
        print("Exporting Decoder to WebNN...")
        print("-" * 60)

        try:
            # Create a wrapper for decoder only
            class DecoderOnly(torch.nn.Module):
                def __init__(self, ae):
                    super().__init__()
                    self.ae = ae

                def forward(self, z):
                    return self.ae.decode(z)

            decoder_model = DecoderOnly(autoencoder)

            compiled_decoder, decoder_exporter = export_model_with_weights(
                model=decoder_model,
                example_input=sample_latent,
                webnn_path=decoder_webnn_path,
                weights_path=decoder_weights_path,
                graph_name="flux_klein_decoder",
                debug=False
            )
            print("Decoder export successful!")

            # Test WebNN decoder
            print("\nTesting WebNN decoder...")
            context = webnn.ML().create_context(device_type="auto")
            webnn_graph = webnn.MLGraph.load(
                decoder_webnn_path,
                weights_path=decoder_weights_path
            )
            input_name = webnn_graph.get_input_names()[0]
            output_name = webnn_graph.get_output_names()[0]
            webnn_output = context.compute(
                webnn_graph,
                {input_name: sample_latent.detach().cpu().numpy().astype(np.float32)},
            )[output_name]
            torch_output = decoded_image.detach().cpu().numpy()

            mae = float(np.mean(np.abs(webnn_output - torch_output)))
            print(f"WebNN vs Torch MAE: {mae:.6f}")

        except NotImplementedError as e:
            print("Decoder export hit an unsupported operation:")
            print(str(e))
            print("\nTo add support, implement the converter in webnn_generator.py")
        except Exception as e:
            print(f"Decoder export encountered an error: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)

    print("\nFlux Klein 4B is a complex multi-component model:")
    print("  1. Text Encoder (Qwen3-4B): Encodes text prompts")
    print("  2. Flow Model: Main diffusion model for denoising")
    print("  3. AutoEncoder: Converts between pixel and latent space")

    print("\nExported components:")
    if export_flow_model:
        if os.path.exists(flow_webnn_path):
            flow_size = os.path.getsize(flow_webnn_path) / 1024
            weights_size = os.path.getsize(flow_weights_path) / 1024
            print(f"  - Flow Model WebNN: {flow_webnn_path}")
            print(f"    Size: {flow_size:.1f} KB (graph) + {weights_size:.1f} KB (weights)")
    if export_ae_model:
        if os.path.exists(decoder_webnn_path):
            decoder_size = os.path.getsize(decoder_webnn_path) / 1024
            decoder_weights_size = os.path.getsize(decoder_weights_path) / 1024
            print(f"  - Decoder WebNN: {decoder_webnn_path}")
            print(f"    Size: {decoder_size:.1f} KB (graph) + {decoder_weights_size:.1f} KB (weights)")

    print("\nKey Takeaways:")
    print("  - Complex models can be exported component-by-component")
    print("  - Each component has its own graph and weights")
    print("  - Text encoder can be cached/pre-computed for efficiency")
    print("  - Flow model runs iteratively during generation (4-50 steps)")
    print("  - Decoder runs once at the end to convert latents to images")

    print("\nNote: Some operations in Flux Klein may not be fully supported")
    print("by WebNN yet. This example demonstrates the export process and")
    print("structure for complex multi-component models.")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()

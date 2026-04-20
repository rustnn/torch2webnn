"""
Flux Klein WebNN Export — Entry Point

Runs all export scripts in sequence. You can also run each script individually:

    python export_text_encoder.py          # text encoder (Qwen3-4B)
    python export_flow.py                  # flow / denoising model
    python export_autoencoder.py           # VAE encoder + decoder (both)
    python export_autoencoder.py --mode encoder   # encoder only (img2img)
    python export_autoencoder.py --mode decoder   # decoder only (txt2img)
"""

import export_text_encoder
import export_flow
import export_autoencoder

if __name__ == "__main__":
    # export_text_encoder.main()
    # export_flow.main()
    export_autoencoder.main(mode="both")

"""Shared setup for Flux Klein export scripts."""

import os
import torch

OUTPUT_DIR = os.path.dirname(__file__)


def hf_login():
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            print("Logged in to HuggingFace")
        except Exception as e:
            print(f"Note: Could not login to HuggingFace: {e}")
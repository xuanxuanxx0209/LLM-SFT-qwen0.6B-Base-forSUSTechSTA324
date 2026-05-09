#!/usr/bin/env python3
"""Check environment setup for vLLM lab."""

import sys
import subprocess

def check_env():
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)

    # Python version
    print(f"Python version: {sys.version}")

    # Check torch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
    except ImportError as e:
        print(f"PyTorch not found: {e}")

    # Check transformers
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"Transformers not found: {e}")

    # Check vllm
    try:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
    except ImportError as e:
        print(f"vLLM not found: {e}")

    # Check requests
    try:
        import requests
        print(f"Requests version: {requests.__version__}")
    except ImportError as e:
        print(f"Requests not found: {e}")

    # Check model files
    import os
    model_path = "/home/ubuntu/models/Qwen3-0.6B"
    if os.path.exists(model_path):
        files = os.listdir(model_path)
        print(f"Model found at {model_path}: {len(files)} files")
    else:
        print(f"Model NOT found at {model_path}")

    print("=" * 60)

if __name__ == "__main__":
    check_env()

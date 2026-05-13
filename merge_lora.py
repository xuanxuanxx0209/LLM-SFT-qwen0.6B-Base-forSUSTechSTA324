#!/usr/bin/env python3
"""Merge the LoRA adapter into the base V3 model and produce a standalone
HF model directory containing model.safetensors plus tokenizer/config files.

Output: /home/ubuntu/qwen3-0.6B-mathsft-V3-lora-merged/
  - model.safetensors (the file you submit)
  - config.json, generation_config.json
  - tokenizer files + chat_template.jinja
"""
import shutil
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_PATH    = "/home/ubuntu/qwen3-0.6B-mathsft-V3"
ADAPTER_PATH = "/home/ubuntu/qwen3-0.6B-mathsft-V3-lora-r8-lr2.5e-5"
MERGED_PATH  = "/home/ubuntu/qwen3-0.6B-mathsft-V3-lora-merged-v6"


def main():
    print(f"[merge] Loading base from {BASE_PATH} ...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"[merge] Loading adapter from {ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    print("[merge] Calling merge_and_unload() ...")
    merged = model.merge_and_unload()

    print(f"[merge] Saving merged model to {MERGED_PATH} ...")
    Path(MERGED_PATH).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(MERGED_PATH, safe_serialization=True)

    # Copy tokenizer + chat template + generation config from base
    print("[merge] Copying tokenizer and config files from base ...")
    tok = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
    tok.save_pretrained(MERGED_PATH)
    for fn in ["chat_template.jinja", "generation_config.json"]:
        src = Path(BASE_PATH) / fn
        if src.exists():
            shutil.copy(src, Path(MERGED_PATH) / fn)
            print(f"  - copied {fn}")

    # Sanity check: list merged dir contents
    print(f"\n[done] Files in {MERGED_PATH}:")
    for p in sorted(Path(MERGED_PATH).iterdir()):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name:<35} {size_mb:>10.2f} MB")


if __name__ == "__main__":
    main()

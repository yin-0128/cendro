"""Merge a trained LoRA adapter into the base model, export GGUF, and write an Ollama Modelfile.

Pipeline:
  1. Load base model + LoRA adapter, merge weights, save the merged HF model.
  2. Convert the merged model to GGUF via llama.cpp's convert script (must be available).
  3. Write a Modelfile so `cendro pull --model cendro-7b` can register it with Ollama.

    python scripts/convert_to_gguf.py --model model/output/cendro-7b \
        --outfile model/cendro-7b.gguf

Requires llama.cpp checked out (point --llama-cpp at it, or set LLAMA_CPP_DIR). The actual
GGUF conversion is delegated to llama.cpp's `convert_hf_to_gguf.py`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from api.prompts import SYSTEM_PROMPT

DEFAULT_BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"


def merge_adapter(base_model: str, adapter_dir: str, merged_dir: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Merging adapter {adapter_dir} into {base_model} ...")
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")


def to_gguf(merged_dir: str, outfile: str, llama_cpp_dir: str, quant: str) -> None:
    convert = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise SystemExit(
            f"Could not find {convert}. Clone llama.cpp and pass --llama-cpp <dir> "
            "(or set LLAMA_CPP_DIR)."
        )
    cmd = [sys.executable, str(convert), merged_dir, "--outfile", outfile, "--outtype", quant]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"GGUF written to {outfile}")


def write_modelfile(outfile: str, model_name: str) -> str:
    modelfile = Path(outfile).with_suffix(".Modelfile")
    gguf_abs = Path(outfile).resolve()
    # Embed the FULL review persona (shared with api/prompts.py) so the model behaves like
    # Cendro even when run directly via `ollama run cendro-7b`, without the FastAPI layer.
    # Triple-quoted SYSTEM keeps the multi-line prompt (and its inner quotes) intact.
    modelfile.write_text(
        f"FROM {gguf_abs}\n"
        f'SYSTEM """{SYSTEM_PROMPT.strip()}"""\n'
        f"PARAMETER temperature 0.2\n",
        encoding="utf-8",
    )
    print(f"Modelfile written to {modelfile}")
    print(f"Register it with: ollama create {model_name} -f {modelfile}")
    return str(modelfile)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA + export GGUF for Ollama.")
    parser.add_argument("--model", required=True, help="Trained LoRA adapter directory.")
    parser.add_argument("--outfile", default="model/cendro-7b.gguf")
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--merged-dir", default="model/output/merged")
    parser.add_argument("--name", default="cendro-7b", help="Ollama model name.")
    parser.add_argument("--quant", default="q4_k_m", help="GGUF outtype (e.g. q4_k_m, f16).")
    parser.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP_DIR", "llama.cpp"))
    args = parser.parse_args()

    merge_adapter(args.base_model, args.model, args.merged_dir)
    to_gguf(args.merged_dir, args.outfile, args.llama_cpp, args.quant)
    write_modelfile(args.outfile, args.name)


if __name__ == "__main__":
    main()

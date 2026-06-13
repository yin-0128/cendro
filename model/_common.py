"""Shared helpers for the training scripts (config loading, 4-bit model + LoRA setup).

Kept separate so train.py / dpo_train.py / evaluate.py don't duplicate the VRAM-safe
quantization and LoRA wiring. Heavy ML imports are done lazily inside functions so that
importing this module (e.g. for the config loader in tests) doesn't require torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    raw: dict[str, Any]

    @property
    def base_model(self) -> str:
        return self.raw["base_model"]

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})


def load_config(path: str | Path) -> Config:
    import yaml  # lazy: only needed for training, not for load_jsonl in tests

    with open(path, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a {prompt, chosen, rejected} (or {prompt, completion}) JSONL file."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bnb_config(cfg: Config):
    """Build a 4-bit BitsAndBytesConfig from the `quant` section."""
    import torch
    from transformers import BitsAndBytesConfig

    q = cfg.section("quant")
    compute_dtype = getattr(torch, q.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=q.get("load_in_4bit", True),
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def lora_config(cfg: Config):
    """Build a peft LoraConfig from the `lora` section."""
    from peft import LoraConfig

    lo = cfg.section("lora")
    return LoraConfig(
        r=lo.get("r", 16),
        lora_alpha=lo.get("alpha", 32),
        lora_dropout=lo.get("dropout", 0.05),
        target_modules=lo.get("target_modules"),
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_tokenizer(cfg: Config):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_quantized_model(cfg: Config):
    """Load the base model in 4-bit, ready for QLoRA training."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config(cfg),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    return model

# TRAINING.md — Fine-tuning Cendro

End-to-end guide for producing the `cendro-3b` model with **QLoRA + DPO** on a single consumer
GPU (built and tested against an RTX 4060, 8GB VRAM).

> **Hardware rule:** 4-bit QLoRA only — never full fine-tuning. Keep `per_device_train_batch_size`
> at 1 and use gradient accumulation. Use `bfloat16` compute dtype.

## 0. Install training deps

```bash
pip install -e ".[train]"
```

This pulls `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`, `accelerate`, and `torch`.
Verify your GPU is visible:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
nvidia-smi
```

## 1. Get a dataset

A small hand-written seed set is committed so you can train immediately:

```
dataset/seed_pairs.jsonl     # ~20-30 {prompt, chosen, rejected} examples
```

To generate more, use an LLM judge to turn raw code samples into preference pairs. The default
judge is **free and local** — a model you've already pulled into Ollama (`--provider ollama`),
so no API key and no cloud. The provider is pluggable.

```bash
# Put raw code snippets (one file per sample) under dataset/raw/, then:

# FREE & local (default) — uses Ollama structured outputs to force the {chosen, rejected} shape:
python scripts/generate_preferences.py \
  --input dataset/raw/ \
  --provider ollama --model qwen2.5-coder:3b \
  --output dataset/dpo_pairs.jsonl

# Optional cloud judge — higher-quality pairs, but sends code to the API and costs money:
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/generate_preferences.py --input dataset/raw/ --provider anthropic   # or: openai
```

`chosen` = a specific, opinionated review with a concrete fix. `rejected` = a generic,
low-signal review. The local judge is weaker than a frontier model, so human-review your pairs
(at least ~10%) before training — a bigger local model (e.g. `qwen2.5:7b`) gives better pairs.

## 2. (Optional) QLoRA SFT warm-up

A short supervised pass on `{prompt, chosen}` pairs helps DPO converge. Skip if you only have
preference data.

```bash
python model/train.py --config configs/qlora_3b.yaml --dataset dataset/seed_pairs.jsonl \
  --output model/output/sft
```

## 3. DPO training

```bash
python model/dpo_train.py \
  --config configs/qlora_3b.yaml \
  --dataset dataset/seed_pairs.jsonl \
  --output model/output/cendro-3b
```

Key settings (in `configs/qlora_3b.yaml`):

| Setting | Value | Why |
|---------|-------|-----|
| `load_in_4bit` | `true` (nf4 + double quant) | Fit 3B on 8GB |
| `bnb_4bit_compute_dtype` | `bfloat16` | Stable, fast on Ada |
| LoRA `r` / `alpha` | 16 / 32 | Good capacity/VRAM tradeoff |
| `per_device_train_batch_size` | 1 | VRAM limit |
| `gradient_accumulation_steps` | 8 | Effective batch 8 |
| `gradient_checkpointing` | `true` | Saves VRAM |
| `beta` (DPO) | 0.1 | Standard preference strength |
| `max_length` / `max_prompt_length` | 1024 / 768 | Code reviews are short |

Monitor VRAM in a second terminal: `nvidia-smi` (or `nvitop`). If you hit OOM, lower
`max_length` first, then `gradient_accumulation_steps`.

## 4. Evaluate

```bash
python model/evaluate.py --model model/output/cendro-3b
```

Checks each review is non-empty and references a concrete issue; optional BERTScore against
reference reviews. Log results (date + metrics) in `.claude/MEMORY.md`.

## 5. Export to GGUF + serve

```bash
python scripts/convert_to_gguf.py --model model/output/cendro-3b --outfile model/cendro-3b.gguf
# Registers a Modelfile so Ollama can load it:
cendro pull --model cendro-3b
cendro serve --model cendro-3b
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| CUDA OOM at start | Lower `max_length`; confirm 4-bit is actually on (`load_in_4bit`) |
| OOM mid-training | Reduce `gradient_accumulation_steps`; enable `gradient_checkpointing` |
| Loss not moving | Check pairs aren't reversed; try `beta` 0.05–0.3 |
| `bitsandbytes` import error on Windows | Install a Windows-compatible wheel; verify CUDA version match |

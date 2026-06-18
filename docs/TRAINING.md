# TRAINING.md — Fine-tuning Cendro

End-to-end guide for producing the `cendro` model with **QLoRA + DPO** on a single consumer GPU
(targets an RTX 4060, 8GB VRAM). Two size targets:

- **`configs/qlora_7b.yaml`** — Qwen2.5-Coder-**7B** (best quality; tight on 8GB — short
  sequences, ideally Unsloth). The recommended target.
- **`configs/qlora_3b.yaml`** — Qwen2.5-Coder-**3B** (safe, fast; good for proving the pipeline).

> **Hardware rule:** 4-bit QLoRA only — never full fine-tuning. Keep `per_device_train_batch_size`
> at 1 and use gradient accumulation. Use `bfloat16` compute dtype.

## 0a. Where to run the training

7B QLoRA is **tight on an 8GB GPU** (it can OOM). Two reliable options:

- **Free Google Colab (T4, 16GB) — easiest for 7B.** No local setup; the 16GB GPU fits 7B
  comfortably. Upload `dataset/train_pairs.jsonl`, run the `dpo_train.py` step in a notebook,
  download `model/output/cendro-7b`. Recommended if local setup is a hassle.
- **WSL2 (Ubuntu) on your own GPU.** Fully local/private. On 8GB, use Unsloth (see below).

### WSL2 setup (Windows)

QLoRA on native Windows often fights `bitsandbytes`/Triton. WSL2 (Ubuntu) is the reliable path
and the only supported environment for Unsloth.

```powershell
# In Windows PowerShell (admin), one-time:
wsl --install -d Ubuntu
```

Then inside the Ubuntu shell:

```bash
# NVIDIA drivers on the Windows host already expose the GPU to WSL — no driver install in WSL.
nvidia-smi                                  # should list your RTX 4060
sudo apt update && sudo apt install -y python3-venv git
git clone https://github.com/yin-0128/cendro && cd cendro
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[train]"
```

> **7B on 8GB:** plain TRL/peft can OOM. For reliable 7B QLoRA, also `pip install unsloth` and
> follow Unsloth's loader (it cuts VRAM ~2x). With the stock scripts, keep `max_length: 768` (or
> drop to 512) from `configs/qlora_7b.yaml`. The 3B config trains comfortably without Unsloth.

## 0b. Install training deps

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

Committed pairs let you train immediately (`dataset/train_pairs.jsonl`). To **scale the dataset
up**, there are two steps — synthesize buggy code, then distill review pairs from it.

### 1a. Synthesize buggy snippets (`generate_samples.py`)

```bash
# Free + hosted (strong, recommended): needs a free GROQ_API_KEY (console.groq.com)
export GROQ_API_KEY=gsk_...
python scripts/generate_samples.py --provider groq --output-dir dataset/raw --per-combo 2

# Free + fully local alternative:
python scripts/generate_samples.py --provider ollama --model qwen2.5-coder:7b \
  --output-dir dataset/raw
```
This writes one file per snippet across a bug taxonomy (injection, races, leaks, O(n²), …) ×
languages (py/js/ts/go/rust/java).

### 1b. Distill preference pairs (`generate_preferences.py`)

```bash
# Free + hosted strong judge (Llama-3.3-70B) — the recommended balance:
python scripts/generate_preferences.py --input dataset/raw/ \
  --provider groq --output dataset/distilled_pairs.jsonl

# Free + fully local judge (uses Ollama structured outputs for the {chosen, rejected} shape):
python scripts/generate_preferences.py --input dataset/raw/ \
  --provider ollama --model qwen2.5-coder:14b --output dataset/distilled_pairs.jsonl

# Optional paid cloud judge — highest quality, sends code to the API:
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/generate_preferences.py --input dataset/raw/ --provider anthropic   # or: openai
```

`chosen` = a specific, opinionated review with a concrete fix. `rejected` = a generic,
low-signal review. **Judge quality matters most:** `groq` (free 70B) ≫ a small local model.
Spot-check ~10% of the pairs, then **merge** the distilled set with the hand-authored
`gold_pairs.jsonl` + `seed_pairs.jsonl` into `dataset/train_pairs.jsonl`, and **carve off a
held-out slice** (`dataset/heldout_pairs.jsonl`, ~20–30 pairs not used in training) for Step 4.

## 2. (Optional) QLoRA SFT warm-up

A short supervised pass on `{prompt, chosen}` pairs helps DPO converge. Skip if you only have
preference data.

```bash
# 7B (quality target). Swap to configs/qlora_3b.yaml for the fast/safe run.
python model/train.py --config configs/qlora_7b.yaml --dataset dataset/train_pairs.jsonl \
  --output model/output/sft
```

## 3. DPO training

```bash
python model/dpo_train.py \
  --config configs/qlora_7b.yaml \
  --dataset dataset/train_pairs.jsonl \
  --output model/output/cendro-7b
```

Key settings (in `configs/qlora_7b.yaml` — the 3B config mirrors them with a longer `max_length`):

| Setting | Value | Why |
|---------|-------|-----|
| `load_in_4bit` | `true` (nf4 + double quant) | Fit the model on 8GB |
| `bnb_4bit_compute_dtype` | `bfloat16` | Stable, fast on Ada |
| LoRA `r` / `alpha` | 16 / 32 | Good capacity/VRAM tradeoff |
| `per_device_train_batch_size` | 1 | VRAM limit |
| `gradient_accumulation_steps` | 8 | Effective batch 8 |
| `gradient_checkpointing` | `true` | Saves VRAM |
| `beta` (DPO) | 0.1 | Standard preference strength |
| `max_length` / `max_prompt_length` | 1024 / 768 | Code reviews are short |

Monitor VRAM in a second terminal: `nvidia-smi` (or `nvitop`). If you hit OOM, lower
`max_length` first, then `gradient_accumulation_steps`.

## 4. Evaluate — prove it beats the base model

```bash
# Score the BASE model and your fine-tune on the SAME held-out prompts, side by side.
python model/evaluate.py --model model/output/cendro-7b \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset dataset/heldout_pairs.jsonl --compare-base
```

This prints both specificity scores and the delta (e.g. `base 41% → cendro 78%`). **Use a
held-out dataset that was NOT in training** — otherwise the win is fake. Optional `--references`
adds BERTScore vs the `chosen` reviews. Log results (date + metrics) in `.claude/MEMORY.md`.

> ⚠️ **Gate:** if cendro-7b doesn't clearly beat base, do **not** ship it. Grow/curate more
> data and retrain — a model that doesn't beat stock Qwen isn't worth publishing.

## 5. Export to GGUF, serve, and publish

```bash
python scripts/convert_to_gguf.py --model model/output/cendro-7b \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct --outfile model/cendro-7b.gguf --name cendro-7b
# Writes a Modelfile (with the full review SYSTEM_PROMPT baked in) so Ollama can load it:
ollama create cendro-7b -f model/cendro-7b.Modelfile
cendro serve --model cendro-7b           # serve locally
ollama run cendro-7b                      # or use it directly, no API needed
```

**Publish so anyone can install it** (free Ollama account; the registry hosts the weights —
GitHub does not):

```bash
ollama cp cendro-7b yin-0128/cendro-7b
ollama push yin-0128/cendro-7b            # → users run: ollama pull yin-0128/cendro-7b
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| CUDA OOM at start | Lower `max_length`; confirm 4-bit is actually on (`load_in_4bit`) |
| OOM mid-training | Reduce `gradient_accumulation_steps`; enable `gradient_checkpointing` |
| Loss not moving | Check pairs aren't reversed; try `beta` 0.05–0.3 |
| `bitsandbytes` import error on Windows | Install a Windows-compatible wheel; verify CUDA version match |

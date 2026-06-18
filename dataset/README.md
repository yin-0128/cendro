# Datasets

DPO preference data: each row is `{prompt, chosen, rejected}` where `prompt` is the exact
Cendro review request (see [`api/prompts.py`](../api/prompts.py)), `chosen` is a specific,
opinionated review with a concrete fix, and `rejected` is a generic, low-signal review.

| File | Pairs | Source | Committed? |
|------|------:|--------|:----------:|
| `seed_pairs.jsonl` | 12 | Hand-written | ✅ |
| `gold_pairs.jsonl` | 32 | **Hand-authored expert reviews** for every file in `samples/`, grounded in OWASP / language docs, built by [`scripts/build_gold_dataset.py`](../scripts/build_gold_dataset.py) | ✅ |
| `distilled_pairs.jsonl` | 289 | **Distilled** from a strong judge over synthesized buggy code (6 languages), via [`scripts/generate_samples.py`](../scripts/generate_samples.py) → [`scripts/generate_preferences.py`](../scripts/generate_preferences.py) | ✅ |
| `train_pairs.jsonl` | 44 → merge | `seed` + `gold` (+ curated `distilled`) merged — **the training input**. Re-merge after distillation. | ✅ |
| `samples/` | 32 files | Reproducible curated buggy-code corpus (8 languages) — quality anchors | ✅ |
| `raw/` | 289 files | Synthesized buggy snippets (input to distillation) | ❌ (git-ignored) |

> **Held-out set:** before training, carve ~20–30 pairs into `heldout_pairs.jsonl` (kept OUT of
> `train_pairs.jsonl`) so `model/evaluate.py --compare-base` can fairly measure base vs fine-tune.

## Why the committed set is hand-authored

The committed pairs are written by hand (expert reviews that name the *real* bug + impact + fix)
rather than produced by a local model. A local judge is convenient but misses subtle bugs
(loop-variable capture, unquoted shell vars, …) and would teach the model those gaps. High-quality
`chosen` examples are what make DPO actually improve the model.

## Grow the data — three paths

**Distillation (recommended — how `distilled_pairs.jsonl` was built):** synthesize buggy code,
then have a strong judge write the review pairs. **Free** with the Groq 70B judge.
```bash
export GROQ_API_KEY=gsk_...   # free: console.groq.com
python scripts/generate_samples.py --provider groq --output-dir dataset/raw --per-combo 2
python scripts/generate_preferences.py --input dataset/raw/ \
  --provider groq --output dataset/distilled_pairs.jsonl
```
Spot-check ~10% of the pairs, then merge the good ones into `train_pairs.jsonl`. Quality tracks
the judge: `groq` (free 70B) ≫ a small local model. `--provider anthropic`/`openai` are paid.

**Free + local:** same `generate_preferences.py` step with `--provider ollama --model
qwen2.5-coder:14b` — fully offline, weaker judge, so review the `chosen` reviews more carefully.

**Hand-authored (highest quality, slowest):** add reviews to `scripts/build_gold_dataset.py`
(keyed by sample filename) and re-run it. This is how `gold_pairs.jsonl` is produced.

## Train on it

```bash
python model/dpo_train.py --config configs/qlora_7b.yaml \
  --dataset dataset/train_pairs.jsonl --output model/output/cendro-7b
```

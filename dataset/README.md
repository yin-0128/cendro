# Datasets

DPO preference data: each row is `{prompt, chosen, rejected}` where `prompt` is the exact
Cendro review request (see [`api/prompts.py`](../api/prompts.py)), `chosen` is a specific,
opinionated review with a concrete fix, and `rejected` is a generic, low-signal review.

| File | Pairs | Source | Committed? |
|------|------:|--------|:----------:|
| `seed_pairs.jsonl` | 12 | Hand-written | ✅ |
| `gold_pairs.jsonl` | 32 | **Hand-authored expert reviews** for every file in `samples/`, grounded in OWASP / language docs, built by [`scripts/build_gold_dataset.py`](../scripts/build_gold_dataset.py) | ✅ |
| `train_pairs.jsonl` | 44 | `seed_pairs` + `gold_pairs` merged — **the recommended training input** | ✅ |
| `dpo_pairs.jsonl` | — | Output of the local LLM-judge generator | ❌ (git-ignored) |
| `samples/` | 32 files | Reproducible buggy-code corpus (8 languages) | ✅ |
| `raw/` | — | Your own private code to generate from | ❌ (git-ignored) |

## Why the committed set is hand-authored

The committed pairs are written by hand (expert reviews that name the *real* bug + impact + fix)
rather than produced by a local model. A local judge is convenient but misses subtle bugs
(loop-variable capture, unquoted shell vars, …) and would teach the model those gaps. High-quality
`chosen` examples are what make DPO actually improve the model.

## Grow the data — two paths

**Free + local (lower quality, fast):** let a local model judge raw code.
```bash
python scripts/generate_preferences.py --input dataset/samples/ \
  --provider ollama --model qwen2.5-coder:14b --output dataset/dpo_pairs.jsonl
```
Then **skim and fix** the `chosen` reviews before training — the local judge often misses the
real bug. A bigger judge (`:32b`) helps. `--provider anthropic`/`openai` give the best pairs but
cost money and send code to the cloud.

**Hand-authored (highest quality):** add reviews to `scripts/build_gold_dataset.py` (keyed by
sample filename) and re-run it. This is how `gold_pairs.jsonl` is produced.

## Train on it

```bash
python model/dpo_train.py --config configs/qlora_7b.yaml \
  --dataset dataset/train_pairs.jsonl --output model/output/cendro-7b
```

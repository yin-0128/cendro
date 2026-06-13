<div align="center">

# 🔍 Cendro
### Local AI Code Reviewer — Private, Fast, Opinionated

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: pre-release](https://img.shields.io/badge/status-pre--release-orange.svg)](#️-status)
[![Stars](https://img.shields.io/github/stars/yin-0128/cendro?style=social)](https://github.com/yin-0128/cendro)

**Your code never leaves your machine.**

[Status](#️-status) · [Quick Start](#-quick-start) · [How it works](#-how-it-works) · [Train your own](#️-train-your-own-model) · [Roadmap](#️-roadmap)

</div>

---

## 🚧 Status

> **Cendro is in early development (pre-release).** The pieces below are honest about what
> works today versus what is planned. Nothing here is published to PyPI or the VS Code
> Marketplace yet — install from source.

| Component | State |
|-----------|-------|
| FastAPI inference server (`/review`, `/health`, `/models`) | ✅ Working (via Ollama) |
| `cendro` CLI (`serve` / `review` / `pull`) | ✅ Working |
| VS Code extension | 🧪 Minimal, runs from source (F5) |
| QLoRA + DPO training pipeline | ✅ Scripts ready, not yet trained on real data |
| Fine-tuned `cendro-3b` model | ⛔ Not trained yet — MVP uses off-the-shelf `qwen2.5-coder:3b` |
| GitHub Action (PR review) | 🧪 Built, not yet battle-tested — Claude API or self-hosted Ollama backend |
| Published packages (PyPI / Marketplace) | ⛔ Planned |

---

## ✨ Why Cendro?

Most AI code tools send your code to the cloud. Cendro is designed to run **entirely on your
machine** — no data leaves, no subscriptions, no rate limits.

The goal is not just another ChatGPT wrapper. The plan is to fine-tune the model with **Direct
Preference Optimization (DPO)** so it gives *opinionated, specific* reviews — not generic
suggestions:

```diff
- "Consider optimizing this loop."          ← Generic AI
+ "This O(n²) loop processes 10k items.      ← Cendro (target behavior)
+  At scale this will block your thread.
+  Use a dict lookup instead — O(1).
+  Here's the fix: [code]"
```

> Today the MVP serves an off-the-shelf `qwen2.5-coder:3b` model through the same prompt and
> API. The DPO fine-tune that produces the opinionated `cendro-3b` model is the next milestone.

---

## 🚀 Quick Start

**Requires:** [Ollama](https://ollama.ai) installed, 8GB+ VRAM or 16GB RAM, Python 3.10+

```bash
# 1. Clone and install from source
git clone https://github.com/yin-0128/cendro
cd cendro
pip install -e .

# 2. Pull the base model into Ollama
cendro pull            # pulls qwen2.5-coder:3b

# 3. Start the local server
cendro serve           # serves on http://localhost:8000

# 4. Review code from the CLI
cendro review path/to/file.py
```

Or review a snippet directly:

```bash
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{"code": "def f():\n  return [i for i in range(1000000)]", "language": "python"}'
```

---

## 🧠 How It Works

```
Your Code (stays local)
       ↓
FastAPI Server (localhost:8000)
       ↓
Model via Ollama (qwen2.5-coder:3b today → fine-tuned cendro-3b next)
       ↓
Detailed Code Review
       ↑
CLI / VS Code Extension / GitHub Action (planned)
```

The plan is a **3B-parameter model fine-tuned with QLoRA + DPO** on good-vs-bad code review
pairs — small enough to run locally, tuned to actually help. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

---

## 🔌 VS Code Extension

The extension is minimal and runs from source today (not yet on the Marketplace):

```bash
cd extension
npm install
# Press F5 in VS Code → opens an Extension Development Host
```

Then: highlight code → right-click → **"Review with Cendro"** (requires `cendro serve` running).

---

## ⚙️ GitHub Action

Cendro reviews changed files on every pull request and posts a summary comment. Two backends:
the **Claude API** (works on GitHub-hosted runners) or a **self-hosted Cendro/Ollama server**
(keeps code on your own infra — use a self-hosted runner). See [github-action/README.md](github-action/README.md).

```yaml
# .github/workflows/code-review.yml
name: Cendro Code Review
on: [pull_request]

permissions:
  contents: read
  pull-requests: write

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: yin-0128/cendro/github-action@main
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          backend: anthropic                       # or: ollama (self-hosted)
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

> 🧪 Built and self-reviewed, but not yet exercised on a live PR — try it and open an issue
> with anything rough.

---

## 🏋️ Train Your Own Model

Fine-tune on your own codebase style or rules. **Requires:** NVIDIA GPU with 8GB+ VRAM.

```bash
# Install training deps
pip install -e ".[train]"

# (Optional) generate preference data with an LLM judge (default: Claude)
python scripts/generate_preferences.py \
  --input dataset/raw/ \
  --output dataset/dpo_pairs.jsonl

# Train with QLoRA + DPO (a committed seed dataset works out of the box)
python model/dpo_train.py \
  --config configs/qlora_3b.yaml \
  --dataset dataset/seed_pairs.jsonl \
  --output model/output/cendro-3b

# Export to GGUF and serve your model
python scripts/convert_to_gguf.py --model model/output/cendro-3b --outfile model/cendro-3b.gguf
cendro serve --model cendro-3b
```

Full guide: [docs/TRAINING.md](docs/TRAINING.md)

---

## 📊 Benchmarks (target — not yet measured)

These are the **goals** we will evaluate against once `cendro-3b` is trained. No numbers are
claimed yet.

| Metric | Target |
|--------|--------|
| Specificity (mentions concrete issue + fix) | High |
| False-positive rate | Low |
| VRAM at inference (3B, 4-bit) | ~4 GB |
| Local latency (RTX 4060) | Interactive |

Evaluation harness: `model/evaluate.py` (see [docs/TRAINING.md](docs/TRAINING.md)).

---

## 🗺️ Roadmap

- [x] FastAPI inference server (Ollama-backed)
- [x] `cendro` CLI
- [x] QLoRA + DPO training scripts
- [x] GitHub Action (PR review) — Claude API / self-hosted backends
- [ ] Trained `cendro-3b` model on a real DPO dataset
- [ ] VS Code extension (Marketplace release)
- [ ] Streaming responses
- [ ] Language-specific model variants (Python, TypeScript, Go)
- [ ] Hosted cloud tier (bring your own key or subscribe)

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

**Good first issues:**
- Add hand-written DPO pairs to `dataset/seed_pairs.jsonl`
- Improve review output formatting
- Add support for a new programming language

---

## 📄 License

MIT — free to use, modify, and distribute. See [LICENSE](LICENSE).

---

<div align="center">

Built by [@yin-0128](https://github.com/yin-0128) · Star ⭐ if this saves you time

</div>

---

## 🤖 Development

This project uses AI-assisted development with Claude. Architecture and technical decisions are
documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); contributor workflows in
[docs/WORKFLOWS.md](docs/WORKFLOWS.md).

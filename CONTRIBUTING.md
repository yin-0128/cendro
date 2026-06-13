# Contributing to Cendro

Thanks for your interest! Cendro is a privacy-first, local AI code reviewer. This guide covers
how to get set up and the conventions we follow.

## Development setup

```bash
git clone https://github.com/yin-0128/cendro
cd cendro
python -m venv .venv && . .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Optional (for training work): `pip install -e ".[train]"` — needs an NVIDIA GPU with 8GB+ VRAM.

## Running the server

```bash
cendro pull        # pull qwen2.5-coder:3b into Ollama
cendro serve       # http://localhost:8000
```

## Before you open a PR

```bash
ruff check .       # lint
ruff format .      # format
pytest -v          # tests (Ollama is mocked; no GPU needed)
```

CI runs the same checks. The test suite must not require a live model or GPU.

## Project layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system design and
[docs/WORKFLOWS.md](docs/WORKFLOWS.md) for common task recipes.

| Area | Where |
|------|-------|
| API server | `api/` |
| Inference wrapper | `model/inference.py` |
| Training pipeline | `model/train.py`, `model/dpo_train.py`, `configs/`, `scripts/` |
| Datasets | `dataset/` (`seed_pairs.jsonl` is committed) |
| VS Code extension | `extension/` |

## Good first issues

- Add hand-written, high-quality DPO pairs to `dataset/seed_pairs.jsonl` (`{prompt, chosen, rejected}`).
- Improve the review system prompt in `api/prompts.py`.
- Add support for a new programming language in the prompt/format.

## Conventions

- **Hardware-aware:** training defaults must fit 8GB VRAM. Never propose full fine-tuning — QLoRA only.
- **Small, testable steps** over large rewrites.
- **Conventional commits:** `feat:`, `fix:`, `docs:`, `chore:`, `test:`.
- Keep the API transport-only; prompt/model logic lives in `api/prompts.py` and `model/inference.py`.

## License

By contributing, you agree your contributions are licensed under the [MIT License](LICENSE).

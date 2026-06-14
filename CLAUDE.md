# CLAUDE.md — Project Intelligence File
> This file is the **single source of truth** for Claude operating in this project.
> It MUST be updated after every significant session, task, or decision.
> Claude should read this file at the start of EVERY session before doing anything.

---

## 🧠 Project Overview

**Project Name:** `Cendro`
**Type:** Fine-tuned local code review AI + VS Code Extension + GitHub Action
**Goal:** Privacy-first, self-hosted AI code reviewer — open source core, paid hosted tier
**Primary Language:** Python (backend), TypeScript (VS Code extension)
**Model Base:** `Qwen/Qwen2.5-Coder-7B-Instruct` (3B available for faster/safer training)
**Training Method:** QLoRA + DPO (Direct Preference Optimization)

---

## 🗂️ Project Structure

```
project-root/
├── CLAUDE.md               ← YOU ARE HERE (public source of truth; update every session)
├── README.md               ← Public-facing, professional
├── LICENSE                 ← MIT
├── CONTRIBUTING.md         ← Contributor guide
├── pyproject.toml          ← Packaging + `cendro` CLI entry point
├── .claude/                ← AI working files, git-ignored (local only)
│   ├── CLAUDE.md           ← Local working copy of this file
│   ├── MEMORY.md           ← Decisions, context, history
│   ├── SKILLS.md / HOOKS.md / ORCHESTRATION.md / MCP.md / TOOLS.md / AGENTS.md
├── docs/                   ← Public contributor docs
│   ├── ARCHITECTURE.md     ← System design decisions
│   ├── WORKFLOWS.md        ← How to do common tasks
│   └── TRAINING.md         ← End-to-end fine-tuning guide
├── api/                    ← FastAPI inference server
│   ├── main.py             ← Endpoints: /review /health /models
│   └── prompts.py          ← Shared code-review system prompt
├── model/
│   ├── inference.py        ← Ollama inference wrapper (used by api/)
│   ├── train.py            ← QLoRA SFT training script
│   ├── dpo_train.py        ← DPO preference training
│   └── evaluate.py         ← Eval harness
├── configs/                ← Training YAML configs (qlora_7b.yaml, qlora_3b.yaml)
├── dataset/                ← Training data (seed_pairs.jsonl committed)
├── scripts/                ← generate_preferences.py, convert_to_gguf.py
├── extension/              ← VS Code extension (TypeScript)
└── .github/workflows/      ← CI + GitHub Action
```

> **Note:** AI working files live in `.claude/` (git-ignored). Public `docs/`
> holds only `ARCHITECTURE.md`, `WORKFLOWS.md`, and `TRAINING.md`.

---

## ⚙️ Hardware Constraints (NEVER FORGET)

| Component | Spec |
|-----------|------|
| GPU | RTX 4060 8GB VRAM |
| RAM | 32GB |
| CPU | Ryzen 5600 |
| OS | Windows 11 |

**Rules from hardware:**
- Always use **4-bit QLoRA** — never full fine-tune
- Max model size: **7B parameters** (7B is the quality target; 3B for faster/safer training)
- Batch size: keep low (1–2) unless tested
- Use `bfloat16` or `float16`, never `float32`
- Always check VRAM before suggesting training configs

---

## 🔄 Session Update Protocol

**At the END of every session, Claude MUST update:**
1. `.claude/MEMORY.md` — what was done, decided, or discovered
2. `CLAUDE.md` — current status, last task, next task
3. Relevant doc file if workflow/tool/skill changed

**Current Status:** `7B serving verified; hand-authored 44-pair gold DPO dataset committed`
**Last Completed Task:** `Built dataset/gold_pairs.jsonl (32 hand-authored expert reviews) + train_pairs.jsonl (44); dropped weak 14b set`
**Next Task:** `In WSL2: run QLoRA+DPO (configs/qlora_7b.yaml on dataset/train_pairs.jsonl); GGUF-export; test Action on a live PR`
**Blockers:** `None`

---

## 🚫 Claude Behavior Rules

- **Never hallucinate library versions** — check PyPI or docs if unsure
- **Never suggest full fine-tuning** — always QLoRA on this hardware
- **Never skip hardware check** — always think about 8GB VRAM limit
- **Always update memory** at end of session
- **Prefer small, testable steps** over large rewrites
- **Ask before refactoring** existing working code
- If uncertain about project context → read `.claude/MEMORY.md` first

---

## 🧩 Key Dependencies

```
# Update this as packages are added
transformers>=4.40
peft>=0.10
trl>=0.8          # For DPO training
bitsandbytes>=0.43
datasets
fastapi
uvicorn
torch>=2.2
```

---

## 📌 Important Decisions Log
> Move detailed entries to .claude/MEMORY.md. Keep only the latest 3 here.

- `2026-06-13` — Upgraded base model to **Qwen2.5-Coder-7B-Instruct** (best quality that fits 8GB for inference; QLoRA 7B training is tight — train in WSL2, `configs/qlora_7b.yaml`). 3B remains the fast/safe fallback.
- `2026-06-13` — Committed DPO data is **hand-authored** (`dataset/gold_pairs.jsonl` + `seed_pairs.jsonl` → `train_pairs.jsonl`, 44 pairs); the local-judge generator stays for growing data but its output must be human-reviewed.
- `2026-06-13` — DPO preference data generation defaults to a **free, local Ollama judge** (privacy-first); `--provider anthropic`/`openai` are optional higher-quality cloud judges.

---

## 🤖 Agent Instructions

When operating as an agent in this project:
1. Read this file first
2. Read `.claude/MEMORY.md` for full context
3. Check current status above before starting
4. Use the smallest model/approach that works
5. Update memory before ending session

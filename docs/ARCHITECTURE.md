# ARCHITECTURE.md — System Design
> The authoritative reference for how the system is built and why.
> Claude must check this before making structural changes.

---

## 🏗️ High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    USER INTERFACES                        │
│  ┌─────────────────┐      ┌──────────────────────────┐  │
│  │  VS Code Plugin  │      │     GitHub Action Bot    │  │
│  │  (TypeScript)    │      │     (.github/workflows)  │  │
│  └────────┬─────────┘      └────────────┬─────────────┘  │
└───────────┼────────────────────────────┼────────────────┘
            │  HTTP/REST                  │  HTTP/REST
            ▼                            ▼
┌──────────────────────────────────────────────────────────┐
│                    FastAPI Server                         │
│              api/main.py  :8000                           │
│    - /review     POST code → review response             │
│    - /health     GET  status check                        │
│    - /models     GET  available models                    │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              Inference Layer                              │
│         Ollama (local) OR llama.cpp                       │
│         Loads fine-tuned GGUF model                       │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              Fine-tuned Model                             │
│   Base: Qwen2.5-Coder-7B (chosen target)             │
│   Training: QLoRA → merged → GGUF quantized              │
│   Method: DPO on code review preference pairs            │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 Key Files and Their Roles

| File | Role | Change frequency |
|------|------|-----------------|
| `api/main.py` | FastAPI server entry point | Low |
| `model/train.py` | QLoRA training script | Medium |
| `model/dpo_train.py` | DPO preference training | Medium |
| `model/inference.py` | Inference wrapper | Low |
| `extension/src/extension.ts` | VS Code plugin main | Medium |
| `github-action/action.yml` | GitHub Action definition | Low |
| `scripts/generate_samples.py` | Synthesize buggy-code corpus (distillation input) | Medium |
| `scripts/generate_preferences.py` | DPO data generation (ollama/groq/anthropic/openai judge) | Medium |
| `scripts/convert_to_gguf.py` | Merge LoRA → GGUF + Ollama Modelfile (bakes in SYSTEM_PROMPT) | Low |

---

## 🔑 Design Decisions

### Why QLoRA + DPO?
- QLoRA: Only way to fine-tune 3B+ model on 8GB VRAM
- DPO: Teaches model *preference* (good vs bad review) without RL complexity
- Together: Production-quality fine-tune on consumer hardware

### Why Ollama for inference?
- Dead simple install (`curl | sh`)
- Handles GGUF loading, memory management, API server
- Users don't need to install Python to run inference
- Fallback: llama.cpp directly if more control needed

### Why FastAPI?
- Async = handles multiple extension requests
- Auto-generates OpenAPI docs
- Easy to deploy on any cloud later for hosted tier

### Why VS Code over JetBrains?
- 17M users vs ~5M
- TypeScript ecosystem is simpler for solo dev
- Can add JetBrains later via same API

---

## 🚧 Known Limitations

- 7B is the chosen target (quality + community reach). 7B QLoRA is tight on the 8GB 4060 →
  train via Unsloth (WSL2) or a free Colab T4 (16GB). End-user hardware is not a constraint.
- DPO data generation is free (local Ollama judge, or the free Groq 70B judge, or the
  hand-authored gold set); cloud judges (Claude/OpenAI) are optional paid upgrades
- GitHub Action requires user to self-host or use cloud API
- No streaming response yet (planned)

---

## 🗺️ Future Architecture (v2)

```
[Current] Local only
[v1.1]    Optional cloud fallback (user's API key)
[v2.0]    Hosted SaaS tier (our GPU server)
[v2.5]    Team dashboard + multi-repo analytics
```

---

## 📁 Repository Structure (Public vs Private)

```
project-root/
├── .claude/          ← AI working files, git-ignored (local only)
├── .gitignore
├── README.md         ← Public
├── docs/
│   ├── ARCHITECTURE.md   ← Public (contributor reference)
│   └── WORKFLOWS.md      ← Public (contributor guide)
├── model/            ← Training scripts (public, weights git-ignored)
├── extension/        ← VS Code extension source
├── github-action/    ← GitHub Action source
├── api/              ← FastAPI inference server
└── scripts/          ← Utility scripts
```

**Note:** `.claude/` folder contains AI session memory, agent instructions, and dev tooling. It is git-ignored and never pushed to GitHub.

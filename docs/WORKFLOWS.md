# WORKFLOWS.md — Standard Operating Procedures
> Exact steps for recurring tasks. Claude follows these precisely.
> Update when a workflow changes or a better way is found.

---

## 🔁 Workflow 1: Start a New Session

```
1. Read CLAUDE.md → check current status + next task
2. Read docs/MEMORY.md → restore full context
3. Confirm hardware constraints are respected in plan
4. State what you're going to do before doing it
5. Begin task
```

---

## 🔁 Workflow 2: End a Session

```
1. Summarize what was completed
2. Append session log entry to docs/MEMORY.md
3. Update CLAUDE.md:
   - Current Status
   - Last Completed Task
   - Next Task
   - Any new blockers
4. Commit message suggestion: "feat/fix/docs: [what changed]"
```

---

## 🔁 Workflow 3: Add a New Feature

```
1. Check docs/ARCHITECTURE.md — does it fit current design?
2. Write the plan in plain English first (no code yet)
3. Identify files that will change
4. Check VRAM impact if ML-related
5. Implement in small testable steps
6. Update ARCHITECTURE.md if design changed
7. Log decision in MEMORY.md
```

---

## 🔁 Workflow 4: Training a New Model Version

```
1. Check dataset/README.md for current data state
2. Verify QLoRA config matches hardware (8GB VRAM)
3. Run: python model/train.py --config configs/[config].yaml
4. Monitor VRAM with: watch -n 1 nvidia-smi
5. Evaluate with: python model/evaluate.py
6. Log results in MEMORY.md with date + metrics
7. If better → update model/CURRENT_MODEL.md
```

---

## 🔁 Workflow 5: Generating DPO Training Data

```
1. Collect raw code snippets (dataset/samples/ committed, or dataset/raw/ for your own)
2. Run: python scripts/generate_preferences.py --provider ollama --model qwen2.5-coder:14b
   → FREE + local by default (local Ollama judge). --provider anthropic/openai optional (cloud, paid)
   → Or hand-author reviews via scripts/build_gold_dataset.py for top quality
3. Human-review the pairs (the local judge can miss subtle bugs) — fix weak `chosen` reviews
4. Store in dataset/dpo_pairs.jsonl; curate the good ones into the committed set
5. Log dataset size + date in MEMORY.md
```

---

## 🔁 Workflow 6: Testing the VS Code Extension

```
1. cd extension/
2. npm install
3. Press F5 in VS Code → opens Extension Development Host
4. Open a Python file → highlight code → right click
5. Verify "Review with AI" option appears
6. Check output in Developer Console
```

---

## 🔁 Workflow 7: Debugging Inference Issues

```
1. Check model is loaded: curl http://localhost:8000/health
2. Check VRAM: nvidia-smi
3. Try with smaller input first
4. Check logs: tail -f logs/inference.log
5. If OOM → reduce max_new_tokens or use smaller model
```

---

## 🔁 Workflow 8: Publishing a New Release

```
1. Update CHANGELOG.md
2. Bump version in pyproject.toml + package.json
3. Run tests: pytest + npm test
4. Build extension: npm run package
5. Tag release: git tag v[x.x.x]
6. Push: git push --tags
7. GitHub Actions handles the rest
8. Post on: Reddit r/LocalLLaMA, r/MachineLearning, HN Show HN
```

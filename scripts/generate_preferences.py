"""Generate DPO preference pairs from raw code samples using an LLM judge.

For each code file under --input, this produces one {prompt, chosen, rejected} record:
  - prompt   : the exact Cendro review request (from api/prompts.py) — what we train on.
  - chosen   : a specific, opinionated review WITH a concrete fix (from the judge).
  - rejected : a deliberately generic, low-signal review (also from the judge).

The judge defaults to Claude (claude-opus-4-8) via the Anthropic SDK. The provider is
pluggable with --provider {anthropic,openai}. Requires the matching API key in the env
(ANTHROPIC_API_KEY / OPENAI_API_KEY).

    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_preferences.py --input dataset/raw/ --output dataset/dpo_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from api.prompts import build_review_messages

# Asks the judge for both sides in one call, returned as strict JSON.
_JUDGE_INSTRUCTION = """You are building training data for a code-review model.

Given the code below, produce TWO reviews as JSON with keys "chosen" and "rejected":
- "chosen": a senior-engineer review. Specific and opinionated. Name the concrete
  problem, explain the impact, and give a fix (a short code snippet where it helps).
- "rejected": a generic, low-signal review of the SAME code — vague suggestions with
  no specifics or fix (the kind of unhelpful output we want to train AWAY from).

Return ONLY the JSON object, no prose.

Language: {language}
Code:
```{language}
{code}
```"""

_EXT_LANG = {
    ".py": "python", ".ts": "typescript", ".js": "javascript", ".go": "go",
    ".rs": "rust", ".java": "java", ".rb": "ruby", ".c": "c", ".cpp": "cpp",
}


def guess_language(path: Path) -> str:
    return _EXT_LANG.get(path.suffix.lower(), "text")


def _parse_pair(text: str) -> dict[str, str]:
    """Extract {chosen, rejected} from the judge's JSON response (tolerates code fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1].lstrip("json").strip()
    data = json.loads(text)
    return {"chosen": data["chosen"].strip(), "rejected": data["rejected"].strip()}


def judge_anthropic(code: str, language: str) -> dict[str, str]:
    from anthropic import Anthropic

    prompt = _JUDGE_INSTRUCTION.format(language=language, code=code)
    client = Anthropic()
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=2000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return _parse_pair(text)


def judge_openai(code: str, language: str) -> dict[str, str]:
    from openai import OpenAI

    prompt = _JUDGE_INSTRUCTION.format(language=language, code=code)
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_pair(resp.choices[0].message.content)


JUDGES = {"anthropic": judge_anthropic, "openai": judge_openai}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs.")
    parser.add_argument("--input", required=True, help="Directory of raw code files.")
    parser.add_argument("--output", default="dataset/dpo_pairs.jsonl")
    parser.add_argument("--provider", choices=list(JUDGES), default="anthropic")
    args = parser.parse_args()

    judge = JUDGES[args.provider]
    files = sorted(p for p in Path(args.input).rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"No files found under {args.input}")

    os.makedirs(Path(args.output).parent, exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for path in files:
            code = path.read_text(encoding="utf-8", errors="ignore")
            if not code.strip():
                continue
            language = guess_language(path)
            try:
                pair = judge(code, language)
            except Exception as exc:  # one bad sample shouldn't kill the run
                print(f"  skip {path}: {exc}")
                continue
            # `prompt` is the user turn of the real review request — train/serve stay aligned.
            prompt = build_review_messages(code, language=language)[1]["content"]
            out.write(json.dumps({"prompt": prompt, **pair}) + "\n")
            written += 1
            print(f"  ok   {path}")

    print(f"Wrote {written} pairs to {args.output}")


if __name__ == "__main__":
    main()

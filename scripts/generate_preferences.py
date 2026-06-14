"""Generate DPO preference pairs from raw code samples using an LLM judge.

For each code file under --input, this produces one {prompt, chosen, rejected} record:
  - prompt   : the exact Cendro review request (from api/prompts.py) — what we train on.
  - chosen   : a specific, opinionated review WITH a concrete fix (from the judge).
  - rejected : a deliberately generic, low-signal review (also from the judge).

The judge is pluggable via --provider:
  - ollama (default): FREE and fully local — uses a model already pulled into Ollama.
    No API key, no cloud, code never leaves your machine. Lower quality than a frontier
    cloud model, but completely free.
  - anthropic / openai: optional cloud judges — higher-quality pairs, but they send code
    to the respective API and need ANTHROPIC_API_KEY / OPENAI_API_KEY (and cost money).

    # Free + local (default): generate with a model you've already pulled
    python scripts/generate_preferences.py --input dataset/raw/ \\
        --provider ollama --model qwen2.5-coder:3b

    # Optional cloud judge (better pairs, costs money)
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_preferences.py --input dataset/raw/ --provider anthropic
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


def _coerce(value) -> str:
    """Small local models sometimes nest the review in an object — flatten to text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        # Join any string-ish leaves (e.g. {"issue": "...", "fix": "..."}).
        parts = [str(v).strip() for v in value.values() if isinstance(v, (str, int, float))]
        return "\n".join(parts) if parts else json.dumps(value)
    if isinstance(value, list):
        return "\n".join(_coerce(v) for v in value)
    return str(value).strip()


def _parse_pair(text: str) -> dict[str, str]:
    """Extract {chosen, rejected} from the judge's JSON response (tolerates code fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1].lstrip("json").strip()
    data = json.loads(text)
    chosen, rejected = _coerce(data.get("chosen", "")), _coerce(data.get("rejected", ""))
    if not chosen or not rejected:
        raise ValueError("judge response missing 'chosen'/'rejected'")
    return {"chosen": chosen, "rejected": rejected}


def judge_ollama(code: str, language: str, *, model: str, server_url: str) -> dict[str, str]:
    """Free, fully local judge. Uses a model already pulled into Ollama (no API key, no cloud)."""
    import urllib.request

    prompt = _JUDGE_INSTRUCTION.format(language=language, code=code)
    # Pass a JSON *schema* (Ollama structured outputs) so even small models return exactly
    # {chosen, rejected} as strings — far more reliable than format="json" alone.
    schema = {
        "type": "object",
        "properties": {"chosen": {"type": "string"}, "rejected": {"type": "string"}},
        "required": ["chosen", "rejected"],
    }
    payload = json.dumps(
        {
            "model": model or "qwen2.5-coder:3b",
            "stream": False,
            "format": schema,
            "options": {"temperature": 0.3},
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
    return _parse_pair(body["message"]["content"])


def judge_anthropic(code: str, language: str, *, model: str, server_url: str) -> dict[str, str]:
    """Optional cloud judge (Claude). Higher-quality pairs, but sends code to the Claude API."""
    from anthropic import Anthropic

    prompt = _JUDGE_INSTRUCTION.format(language=language, code=code)
    client = Anthropic()
    resp = client.messages.create(
        model=model or "claude-opus-4-8",
        max_tokens=2000,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return _parse_pair(text)


def judge_openai(code: str, language: str, *, model: str, server_url: str) -> dict[str, str]:
    """Optional cloud judge (OpenAI). Sends code to the OpenAI API."""
    from openai import OpenAI

    prompt = _JUDGE_INSTRUCTION.format(language=language, code=code)
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model or "gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_pair(resp.choices[0].message.content)


# `ollama` is the default: free, local, and on-brand for a privacy-first tool.
JUDGES = {"ollama": judge_ollama, "anthropic": judge_anthropic, "openai": judge_openai}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs.")
    parser.add_argument("--input", required=True, help="Directory of raw code files.")
    parser.add_argument("--output", default="dataset/dpo_pairs.jsonl")
    parser.add_argument(
        "--provider", choices=list(JUDGES), default="ollama",
        help="Judge backend. 'ollama' is free/local (default); 'anthropic'/'openai' are cloud.",
    )
    parser.add_argument("--model", default="", help="Judge model id (backend-specific default).")
    parser.add_argument(
        "--server-url",
        default="http://localhost:11434",
        help="Ollama server URL (ollama provider).",
    )
    args = parser.parse_args()

    def judge(code: str, language: str) -> dict[str, str]:
        return JUDGES[args.provider](
            code, language, model=args.model, server_url=args.server_url
        )

    files = sorted(p for p in Path(args.input).rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"No files found under {args.input}")

    os.makedirs(Path(args.output).parent, exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for path in files:
            language = guess_language(path)
            if language == "text":  # skip non-code files (README, docs, etc.)
                print(f"  skip {path}: not a recognized code file")
                continue
            code = path.read_text(encoding="utf-8", errors="ignore")
            if not code.strip():
                continue
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

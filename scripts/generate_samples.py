"""Generate a diverse corpus of buggy code snippets to feed DPO data generation.

The committed `dataset/samples/` corpus (~32 snippets) is too small to train on. This script
asks a strong model to synthesize many *new* realistic-but-flawed snippets across a bug
taxonomy x languages, writing one file per snippet. Point `generate_preferences.py` at the
output directory to distill {chosen, rejected} review pairs from them.

Providers mirror generate_preferences.py:
  - groq (default here): FREE hosted, strong 70B-class model, no local GPU. Needs GROQ_API_KEY.
  - ollama: FREE + fully local, uses a model already pulled into Ollama.

    # Free + hosted (recommended for variety/quality):
    export GROQ_API_KEY=gsk_...
    python scripts/generate_samples.py --output-dir dataset/raw --per-combo 2

    # Free + local:
    python scripts/generate_samples.py --provider ollama --model qwen2.5-coder:7b \\
        --output-dir dataset/raw

The output is intentionally written to `dataset/raw/` (git-ignored scratch). Spot-check the
snippets, then run generate_preferences.py and curate the resulting pairs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# Bug categories mirror dataset/samples/README.md coverage so the distribution stays balanced.
CATEGORIES = [
    "SQL injection from string-built queries",
    "OS command injection from unsanitized input",
    "eval / dynamic execution of user input",
    "unsafe deserialization of untrusted data",
    "missing authorization / ownership check before a sensitive action",
    "weak crypto or insecure randomness for secrets/tokens",
    "path traversal from unsanitized filenames",
    "off-by-one or boundary error",
    "null / None / nil dereference",
    "identity vs equality comparison bug",
    "dict/list/map mutated while iterating",
    "naive datetime / timezone handling",
    "division by zero or unchecked denominator",
    "unsynchronized shared state / race condition",
    "O(n^2) or accidentally quadratic loop",
    "string concatenation in a loop / repeated recompilation",
    "N+1 query pattern",
    "swallowed exception / bare except / blind infinite retry",
    "unclosed file/socket/resource leak",
    "missing network timeout",
    "sequential awaits or floating/unawaited promise",
    "float used for money",
    "unquoted shell variable / dangerous rm",
]

# language -> (file extension, label used in the prompt)
LANGUAGES = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "go": ".go",
    "rust": ".rs",
    "java": ".java",
}

_GEN_INSTRUCTION = """You are creating training material for a code-review model.

Write a SHORT, realistic {language} code snippet (a function or small module, ~8-25 lines)
that contains this flaw: {category}.

Rules:
- The bug must be genuine and reviewable, not labeled. Do NOT add comments pointing at the bug.
- Make it look like ordinary production code someone might actually write.
- No prose, no explanation, no markdown fences. Output ONLY the raw {language} code.
"""

# Not every flaw maps to every language; skip obvious mismatches to avoid junk.
_SKIP = {
    ("unquoted shell variable / dangerous rm", "go"),
    ("unquoted shell variable / dangerous rm", "rust"),
    ("unquoted shell variable / dangerous rm", "java"),
    ("unquoted shell variable / dangerous rm", "typescript"),
    ("N+1 query pattern", "rust"),
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


def _strip_fences(text: str) -> str:
    """Models sometimes wrap code in ``` fences despite instructions — strip them."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text.rstrip())
    return text.strip()


def gen_groq(prompt: str, *, model: str, server_url: str) -> str:
    import urllib.request

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com and "
            "`export GROQ_API_KEY=gsk_...`."
        )
    payload = json.dumps(
        {
            "model": model or "llama-3.3-70b-versatile",
            "temperature": 0.8,  # higher temp -> more variety across snippets
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # Cloudflare in front of Groq blocks urllib's default UA with 403 (error 1010).
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Cendro/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    return _strip_fences(body["choices"][0]["message"]["content"])


def gen_ollama(prompt: str, *, model: str, server_url: str) -> str:
    import urllib.request

    payload = json.dumps(
        {
            "model": model or "qwen2.5-coder:7b",
            "stream": False,
            "options": {"temperature": 0.8},
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
    return _strip_fences(body["message"]["content"])


GENERATORS = {"groq": gen_groq, "ollama": gen_ollama}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate buggy code snippets for DPO distillation."
    )
    parser.add_argument("--output-dir", default="dataset/raw")
    parser.add_argument("--provider", choices=list(GENERATORS), default="groq")
    parser.add_argument(
        "--model", default="", help="Generator model id (provider-specific default)."
    )
    parser.add_argument("--server-url", default="http://localhost:11434", help="Ollama server URL.")
    parser.add_argument(
        "--per-combo", type=int, default=1, help="Snippets per (category, language)."
    )
    parser.add_argument(
        "--languages", default=",".join(LANGUAGES),
        help="Comma-separated subset of languages to generate.",
    )
    args = parser.parse_args()

    languages = [lng.strip() for lng in args.languages.split(",") if lng.strip() in LANGUAGES]
    generate = GENERATORS[args.provider]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for category in CATEGORIES:
        for language in languages:
            if (category, language) in _SKIP:
                continue
            for i in range(args.per_combo):
                prompt = _GEN_INSTRUCTION.format(language=language, category=category)
                try:
                    code = generate(prompt, model=args.model, server_url=args.server_url)
                except Exception as exc:  # one failure shouldn't kill the run
                    print(f"  skip {language}/{_slug(category)}: {exc}")
                    continue
                if not code.strip():
                    continue
                suffix = f"_{i}" if args.per_combo > 1 else ""
                name = f"{language}_{_slug(category)}{suffix}{LANGUAGES[language]}"
                (out_dir / name).write_text(code + "\n", encoding="utf-8")
                written += 1
                print(f"  ok   {name}")

    print(f"Wrote {written} snippets to {out_dir}")
    print("Next: python scripts/generate_preferences.py --input", out_dir, "--provider groq")


if __name__ == "__main__":
    main()

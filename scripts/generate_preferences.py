"""Generate DPO preference pairs from raw code samples using an LLM judge.

For each code file under --input, this produces one {prompt, chosen, rejected} record:
  - prompt   : the exact Cendro review request (from api/prompts.py) — what we train on.
  - chosen   : a specific, opinionated review WITH a correct concrete fix (from the judge).
  - rejected : a HARD negative — looks just as confident and detailed as `chosen` and is of
               comparable length, but is subtly WRONG (misses the bug / proposes a bad fix /
               nitpicks style). This forces DPO to learn substance, not "pick the longer one"
               (see scripts/check_pairs.py to confirm the length gap stays ~1x).

The judge is pluggable via --provider:
  - ollama (default): FREE and fully local — uses a model already pulled into Ollama.
    No API key, no cloud, code never leaves your machine. Lower quality than a frontier
    cloud model, but completely free.
  - groq: FREE (generous free tier) hosted judge — a strong 70B-class model with no local
    hardware needed. Needs a free GROQ_API_KEY (console.groq.com). Sends code to Groq.
  - anthropic / openai: optional cloud judges — higher-quality pairs, but they send code
    to the respective API and need ANTHROPIC_API_KEY / OPENAI_API_KEY (and cost money).

    # Free + local (default): generate with a model you've already pulled
    python scripts/generate_preferences.py --input dataset/raw/ \\
        --provider ollama --model qwen2.5-coder:3b

    # Free + hosted (strong judge, no local GPU): distill from Groq's free tier
    export GROQ_API_KEY=gsk_...
    python scripts/generate_preferences.py --input dataset/raw/ --provider groq

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
#
# `rejected` is a HARD negative, NOT a short/vague review. If `rejected` were obviously
# generic and short, DPO learns the trivial shortcut "prefer the longer text" instead of
# review quality (the chosen/rejected length gap was 4-7x and the model hit 100% reward
# accuracy in half an epoch). A hard negative looks just as confident and detailed as
# `chosen` but is subtly WRONG, so the only way to win the preference is real substance.
_JUDGE_INSTRUCTION = """You are building training data for a code-review model.

Given the code below, produce TWO reviews as JSON with keys "chosen" and "rejected":
- "chosen": a senior-engineer review. Specific and opinionated. Name the single most
  important concrete problem (correctness/security first), explain the impact, and give a
  correct fix (a short code snippet where it helps).
- "rejected": a HARD negative. It must look just as confident, detailed, and well-formatted
  as "chosen" -- SIMILAR LENGTH and tone -- but be subtly WRONG in substance. Do at least
  one of: (a) miss the real bug while confidently flagging a non-issue, (b) propose a
  plausible-sounding fix that is actually incorrect or introduces a new bug, or (c) focus on
  style/nits while ignoring the real correctness or security problem. Do NOT make it short,
  vague, or obviously low-effort -- it should be tempting but mistaken.

Both reviews must be of comparable length so length alone never distinguishes them.

Return ONLY the JSON object, no prose.

Language: {language}
Code:
```{language}
{code}
```"""

# If the API asks us to wait longer than this (seconds), treat it as the daily cap and stop
# cleanly rather than sleeping. Overridable with --max-wait. Set in main().
MAX_WAIT = 90


class RateLimited(RuntimeError):
    """Raised when the daily quota is (likely) exhausted, so the run stops and resumes later."""


_EXT_LANG = {
    ".py": "python", ".ts": "typescript", ".js": "javascript", ".go": "go",
    ".rs": "rust", ".java": "java", ".rb": "ruby", ".c": "c", ".cpp": "cpp",
    ".sh": "bash", ".sql": "sql", ".cs": "csharp", ".kt": "kotlin", ".php": "php",
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


def _parse_pair_xml(text: str) -> dict[str, str]:
    """Fallback parser for XML-delimited responses when JSON mode fails."""
    import re
    chosen_match = re.search(r"<chosen>(.*?)</chosen>", text, re.DOTALL)
    rejected_match = re.search(r"<rejected>(.*?)</rejected>", text, re.DOTALL)
    if not chosen_match or not rejected_match:
        raise ValueError("fallback XML parse failed: missing <chosen> or <rejected>")
    return {
        "chosen": chosen_match.group(1).strip(),
        "rejected": rejected_match.group(1).strip(),
    }


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


_JUDGE_INSTRUCTION_XML = """You are building training data for a code-review model.

Given the code below, produce TWO reviews wrapped in XML tags:
- <chosen>: a senior-engineer review. Specific and opinionated. Name the single most
  important concrete problem (correctness/security first), explain the impact, and give a
  correct fix (a short code snippet where it helps).
- <rejected>: a HARD negative. It must look just as confident, detailed, and well-formatted
  as <chosen> -- SIMILAR LENGTH and tone -- but be subtly WRONG in substance. Do at least
  one of: (a) miss the real bug while confidently flagging a non-issue, (b) propose a
  plausible-sounding fix that is actually incorrect or introduces a new bug, or (c) focus on
  style/nits while ignoring the real correctness or security problem. Do NOT make it short,
  vague, or obviously low-effort -- it should be tempting but mistaken.

Both reviews must be of comparable length so length alone never distinguishes them.

Format your response exactly as:
<chosen>
...review text here...
</chosen>
<rejected>
...review text here...
</rejected>

Language: {language}
Code:
```{language}
{code}
```"""


def _judge_groq_xml_fallback(
    code: str, language: str, *, model: str, api_key: str
) -> dict[str, str]:
    """Fallback for when Groq's json_validate_failed: use XML delimiters without JSON mode."""
    import urllib.request  # noqa: PLC0415 — lazy import mirrors judge_groq pattern

    prompt = _JUDGE_INSTRUCTION_XML.format(language=language, code=code)
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0.3,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Cendro/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    text = body["choices"][0]["message"]["content"]
    return _parse_pair_xml(text)


def judge_groq(code: str, language: str, *, model: str, server_url: str) -> dict[str, str]:
    """Free hosted judge (Groq, OpenAI-compatible). Strong 70B-class model, no local GPU.

    Uses a free GROQ_API_KEY (console.groq.com). Code is sent to the Groq API. Implemented
    with stdlib urllib so distillation needs no extra dependency.
    """
    import time
    import urllib.error
    import urllib.request

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com and "
            "`export GROQ_API_KEY=gsk_...`."
        )
    prompt = _JUDGE_INSTRUCTION.format(language=language, code=code)
    payload = json.dumps(
        {
            "model": model or "llama-3.3-70b-versatile",
            "temperature": 0.3,
            # OpenAI-compatible JSON mode — return exactly {chosen, rejected}.
            "response_format": {"type": "json_object"},
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
    # Free tier is 12k tokens/min; firing all samples back-to-back hits 429. Wait and retry
    # (honoring Retry-After) so a long run rides under the limit instead of failing.
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
            return _parse_pair(body["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:500]
            if e.code == 429 and attempt < 7:
                # Retry-After header (seconds) if given, else exponential backoff capped at 30s.
                wait = float(e.headers.get("Retry-After") or 0) or min(2 ** attempt, 30)
                if wait > MAX_WAIT:
                    # A wait this long means the per-DAY cap is hit, not just per-minute. Stop
                    # cleanly instead of sleeping for ages — the run resumes next time.
                    raise RateLimited(
                        f"Groq wants a {wait:.0f}s wait (> {MAX_WAIT}s) — you've likely hit the "
                        f"free daily token cap."
                    ) from e
                print(f"  rate limited, waiting {wait:.0f}s (retry {attempt + 1}/7)...")
                time.sleep(max(wait, 2))
                continue
            if e.code == 400 and "json_validate_failed" in detail:
                # Groq's constrained JSON generation failed for this input. Retry once without
                # JSON mode, using XML-style delimiters that are easier for the model to emit.
                return _judge_groq_xml_fallback(
                    code, language, model=model or "llama-3.3-70b-versatile", api_key=api_key
                )
            # Surface Groq's real message (model decommissioned, etc.) for a debuggable failure.
            raise RuntimeError(f"Groq API {e.code}: {detail}") from e
    raise RuntimeError("Groq API: still rate limited after 8 attempts")


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
# `groq` is free + hosted (strong judge, no local GPU); anthropic/openai are paid cloud.
JUDGES = {
    "ollama": judge_ollama,
    "groq": judge_groq,
    "anthropic": judge_anthropic,
    "openai": judge_openai,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs.")
    parser.add_argument("--input", required=True, help="Directory of raw code files.")
    parser.add_argument("--output", default="dataset/dpo_pairs.jsonl")
    parser.add_argument(
        "--provider", choices=list(JUDGES), default="ollama",
        help="Judge backend. 'ollama' is free/local (default); 'groq' is free/hosted; "
        "'anthropic'/'openai' are paid cloud.",
    )
    parser.add_argument("--model", default="", help="Judge model id (backend-specific default).")
    parser.add_argument(
        "--server-url",
        default="http://localhost:11434",
        help="Ollama server URL (ollama provider).",
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="Ignore any existing output and regenerate from scratch (e.g. after changing the "
        "judge prompt). Default behavior RESUMES, skipping pairs already in --output.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Stop after generating this many NEW pairs this run (0 = no limit). Handy for "
        "chipping away under a rate limit across several runs.",
    )
    parser.add_argument(
        "--max-wait", type=int, default=90,
        help="If the API asks to wait longer than this (seconds), stop cleanly (daily cap "
        "likely hit) instead of sleeping. Re-run later to resume.",
    )
    args = parser.parse_args()

    global MAX_WAIT
    MAX_WAIT = args.max_wait

    def judge(code: str, language: str) -> dict[str, str]:
        return JUDGES[args.provider](
            code, language, model=args.model, server_url=args.server_url
        )

    files = sorted(p for p in Path(args.input).rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"No files found under {args.input}")

    out_path = Path(args.output)
    os.makedirs(out_path.parent, exist_ok=True)
    if args.restart and out_path.exists():
        out_path.unlink()

    # RESUME by default: load prompts already in the output and skip those files, so a run
    # interrupted by a rate limit (or Ctrl+C) continues instead of restarting. Each new pair is
    # flushed immediately, so progress is durable even if you stop mid-run.
    done_prompts: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                done_prompts.add(json.loads(line)["prompt"])
        if done_prompts:
            print(f"Resuming: {len(done_prompts)} pairs already in {out_path} — skipping those.")

    written = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for path in files:
            language = guess_language(path)
            if language == "text":  # skip non-code files (README, docs, etc.)
                print(f"  skip {path}: not a recognized code file")
                continue
            code = path.read_text(encoding="utf-8", errors="ignore")
            if not code.strip():
                continue
            # `prompt` is the user turn of the real review request — train/serve stay aligned.
            prompt = build_review_messages(code, language=language)[1]["content"]
            if prompt in done_prompts:
                continue  # already generated in an earlier run
            try:
                pair = judge(code, language)
            except RateLimited as exc:
                print(f"\n{exc}")
                print("Stopping — progress is saved. Re-run the same command later to resume.")
                break
            except Exception as exc:  # one bad sample shouldn't kill the run
                print(f"  skip {path}: {exc}")
                continue
            # Tag the judge so a mixed file is traceable/filterable later (downstream ignores it).
            out.write(json.dumps({"prompt": prompt, **pair, "judge": args.provider}) + "\n")
            out.flush()  # persist now so Ctrl+C / a crash keeps everything generated so far
            done_prompts.add(prompt)
            written += 1
            print(f"  ok   {path}")
            if args.limit and written >= args.limit:
                print(f"  reached --limit {args.limit}; stopping (re-run to continue).")
                break

    print(f"Wrote {written} new pairs to {out_path} ({len(done_prompts)} total).")


if __name__ == "__main__":
    main()

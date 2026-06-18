"""Regenerate rejected responses for gold pairs where chosen/rejected length ratio > 4x.

Gold pairs have excellent hand-authored 'chosen' reviews but very short 'rejected'
responses. This replaces those with proper hard negatives of comparable length using
the Groq judge, then re-runs the merge so train_pairs.jsonl picks them up.

    python scripts/fix_gold_rejected.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

GOLD_PATH = Path("dataset/gold_pairs.jsonl")
MAX_RATIO = 4.0
MODEL = "llama-3.3-70b-versatile"

_REGEN_PROMPT = """You are building training data for a code-review model.

A senior engineer wrote this reference review of the code below:
<chosen_review>
{chosen}
</chosen_review>

Write a HARD NEGATIVE rejected review that is ~{target_len} characters long (similar
length to the chosen review above). It must look just as confident, detailed, and
well-formatted, but be subtly WRONG in substance. Do at least one of:
(a) miss the real bug while confidently flagging a non-issue
(b) propose a plausible-sounding fix that is actually incorrect or introduces a new bug
(c) focus on style/nits while ignoring the real correctness or security problem

Do NOT make it short, vague, or obviously low-effort — it must be tempting but mistaken.
Return ONLY the rejected review text, no labels or extra prose.

Code under review:
{prompt}"""


def _call_groq(content: str, api_key: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "temperature": 0.4,
        "messages": [{"role": "user", "content": content}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Cendro/0.1",
        },
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
            return body["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")
            if e.code == 429 and attempt < 4:
                wait = float(e.headers.get("Retry-After") or 0) or min(2 ** attempt, 30)
                print(f"  rate limited, waiting {wait:.0f}s...")
                time.sleep(max(wait, 2))
                continue
            raise RuntimeError(f"Groq API {e.code}: {detail}") from e
    raise RuntimeError("Still rate limited after 5 attempts")


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set.")

    rows = [json.loads(l) for l in GOLD_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    fixed = 0
    for i, row in enumerate(rows):
        chosen = row.get("chosen", "")
        rejected = row.get("rejected", "")
        if not chosen or not rejected:
            continue
        ratio = max(len(chosen), len(rejected)) / min(len(chosen), len(rejected))
        if ratio <= MAX_RATIO:
            continue

        print(f"  row {i}: ratio={ratio:.1f}x  chosen={len(chosen)}  rejected={len(rejected)}")
        prompt = _REGEN_PROMPT.format(chosen=chosen, target_len=len(chosen), prompt=row["prompt"])
        new_rejected = _call_groq(prompt, api_key)
        new_ratio = max(len(chosen), len(new_rejected)) / min(len(chosen), len(new_rejected))
        print(f"    -> new rejected={len(new_rejected)}  ratio={new_ratio:.1f}x")
        row["rejected"] = new_rejected
        fixed += 1

    GOLD_PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"\nFixed {fixed}/{len(rows)} pairs in {GOLD_PATH}")


if __name__ == "__main__":
    main()

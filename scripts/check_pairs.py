"""Audit DPO preference pairs for the "length shortcut" + basic hygiene.

If `chosen` is consistently much longer than `rejected`, DPO can win the objective just by
preferring the longer text instead of learning review quality (this is exactly what tanked
the first cendro-7b run: a 4-7x length gap and 100% reward accuracy in half an epoch). Keep
the median ratio near ~1x after switching to hard negatives.

    python scripts/check_pairs.py dataset/train_pairs.jsonl
    python scripts/check_pairs.py dataset/*.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def audit(path: Path) -> bool:
    """Print stats for one JSONL file. Returns False if a problem worth flagging is found."""
    chosen_lens: list[int] = []
    rejected_lens: list[int] = []
    dupes = 0
    malformed = 0
    seen_prompts: set[str] = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            prompt, chosen, rejected = row["prompt"], row["chosen"], row["rejected"]
        except (json.JSONDecodeError, KeyError, TypeError):
            malformed += 1
            continue
        if not chosen.strip() or not rejected.strip():
            malformed += 1
            continue
        if prompt in seen_prompts:
            dupes += 1
        seen_prompts.add(prompt)
        chosen_lens.append(len(chosen))
        rejected_lens.append(len(rejected))

    n = len(chosen_lens)
    if n == 0:
        print(f"{path}: no valid pairs ({malformed} malformed)")
        return False

    mc = statistics.median(chosen_lens)
    mr = statistics.median(rejected_lens)
    ratio = mc / mr if mr else float("inf")
    # rejected longer than chosen on a row = length can't be the deciding signal there.
    rejected_longer = sum(r >= c for c, r in zip(chosen_lens, rejected_lens, strict=False))

    print(f"{path}  ({n} pairs)")
    print(f"  median chars  chosen={mc:.0f}  rejected={mr:.0f}  ratio={ratio:.2f}x")
    print(f"  rejected >= chosen length: {rejected_longer}/{n} ({100 * rejected_longer / n:.0f}%)")
    if dupes:
        print(f"  WARN duplicate prompts: {dupes}")
    if malformed:
        print(f"  WARN malformed/empty rows skipped: {malformed}")

    ok = True
    if ratio > 1.5 or ratio < 0.67:
        print(f"  FLAG length ratio {ratio:.2f}x is far from ~1x -> DPO may learn a length "
              f"shortcut. Use hard negatives (scripts/generate_preferences.py).")
        ok = False
    return ok and dupes == 0 and malformed == 0


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or [Path("dataset/train_pairs.jsonl")]
    all_ok = True
    for path in paths:
        if not path.exists():
            print(f"{path}: not found")
            all_ok = False
            continue
        all_ok &= audit(path)
        print()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

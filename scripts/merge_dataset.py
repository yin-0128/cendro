"""Merge preference sources into train + held-out splits.

Combines the distilled, gold, and seed pairs, de-dups on (prompt, chosen),
shuffles deterministically, and carves off a held-out slice that is NOT used
in training (so evaluate.py --compare-base reports an honest win).

    python scripts/merge_dataset.py --heldout 25
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SOURCES = [
    "dataset/distilled_pairs.jsonl",
    "dataset/gold_pairs.jsonl",
    "dataset/seed_pairs.jsonl",
]


def load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge preference pairs into train/heldout splits."
    )
    parser.add_argument("--heldout", type=int, default=25, help="Held-out pairs reserved for eval.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs: list[dict] = []
    for src in SOURCES:
        rows = load(src)
        print(f"  {src:40s} {len(rows)}")
        pairs += rows

    seen: set[tuple[str, str]] = set()
    uniq: list[dict] = []
    for r in pairs:
        key = (r["prompt"], r["chosen"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    random.seed(args.seed)
    random.shuffle(uniq)

    heldout = uniq[: args.heldout]
    train = uniq[args.heldout :]

    write("dataset/train_pairs.jsonl", train)
    write("dataset/heldout_pairs.jsonl", heldout)
    print(
        f"\ntrain={len(train)}  heldout={len(heldout)}  "
        f"(from {len(uniq)} unique of {len(pairs)} total)"
    )


if __name__ == "__main__":
    main()

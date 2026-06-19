"""Deterministically carve a validation slice off a training JSONL.

DPO with no eval split is flying blind — you can't see the train-loss-down /
eval-loss-up overfitting that tanked the first run. This writes two disjoint files
(no leakage) so dpo_train.py can track eval loss and keep the best checkpoint.

NOTE: split from `train_pairs.jsonl`, NOT from `heldout_pairs.jsonl` — heldout is the
final gate and must never be seen during training OR validation.

    python scripts/split_dataset.py --input dataset/train_pairs.jsonl \
        --train dataset/train_split.jsonl --val dataset/val_pairs.jsonl --val-size 24
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a JSONL into disjoint train/val files.")
    parser.add_argument("--input", default="dataset/train_pairs.jsonl")
    parser.add_argument("--train", default="dataset/train_split.jsonl")
    parser.add_argument("--val", default="dataset/val_pairs.jsonl")
    parser.add_argument(
        "--val-size", type=int, default=24, help="Number of rows held out for eval."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.val_size >= len(rows):
        raise SystemExit(f"--val-size {args.val_size} >= dataset size {len(rows)}")

    random.Random(args.seed).shuffle(rows)
    val, train = rows[: args.val_size], rows[args.val_size :]

    for path, part in ((args.train, train), (args.val, val)):
        with open(path, "w", encoding="utf-8") as fh:
            for row in part:
                fh.write(json.dumps(row) + "\n")

    print(f"train: {len(train)} -> {args.train}")
    print(f"val:   {len(val)} -> {args.val}")


if __name__ == "__main__":
    main()

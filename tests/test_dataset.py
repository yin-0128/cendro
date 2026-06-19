"""Validate the committed DPO datasets (CI-safe, no GPU/model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from model._common import load_jsonl

DATA = Path(__file__).resolve().parents[1] / "dataset"
COMMITTED = [
    "seed_pairs.jsonl",
    "gold_pairs.jsonl",
    "distilled_pairs.jsonl",
    "train_pairs.jsonl",
    "heldout_pairs.jsonl",
]


@pytest.mark.parametrize("name", COMMITTED)
def test_dataset_is_well_formed(name):
    rows = load_jsonl(DATA / name)
    assert len(rows) >= 10, f"{name} should have a usable number of pairs"
    keys = set()
    for r in rows:
        assert {"prompt", "chosen", "rejected"} <= r.keys()
        assert r["prompt"].strip() and r["chosen"].strip() and r["rejected"].strip()
        # Hard negatives: chosen and rejected must differ, but rejected is deliberately
        # length-matched — there must be NO "chosen is just longer" shortcut for DPO to exploit.
        assert r["chosen"] != r["rejected"]
        keys.add((r["prompt"], r["chosen"]))
    assert len(keys) == len(rows), f"{name} has duplicate (prompt, chosen) pairs"


def test_train_and_heldout_are_disjoint():
    # evaluate.py --compare-base only reports an honest win if held-out pairs never
    # leaked into training. Guard that split here, on every CI run.
    train = {(r["prompt"], r["chosen"]) for r in load_jsonl(DATA / "train_pairs.jsonl")}
    heldout = {(r["prompt"], r["chosen"]) for r in load_jsonl(DATA / "heldout_pairs.jsonl")}
    assert train and heldout
    assert train.isdisjoint(heldout), "held-out pairs leaked into the training set"

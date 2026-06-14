"""Validate the committed DPO datasets (CI-safe, no GPU/model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from model._common import load_jsonl

DATA = Path(__file__).resolve().parents[1] / "dataset"
COMMITTED = ["seed_pairs.jsonl", "curated_pairs.jsonl", "train_pairs.jsonl"]


@pytest.mark.parametrize("name", COMMITTED)
def test_dataset_is_well_formed(name):
    rows = load_jsonl(DATA / name)
    assert len(rows) >= 10, f"{name} should have a usable number of pairs"
    prompts = set()
    for r in rows:
        assert {"prompt", "chosen", "rejected"} <= r.keys()
        assert r["prompt"].strip() and r["chosen"].strip() and r["rejected"].strip()
        # The chosen review should be more substantial than the generic rejected one.
        assert len(r["chosen"]) > len(r["rejected"])
        prompts.add(r["prompt"])
    assert len(prompts) == len(rows), f"{name} has duplicate prompts"


def test_train_pairs_is_seed_plus_curated():
    n = {name: len(load_jsonl(DATA / name)) for name in COMMITTED}
    assert n["train_pairs.jsonl"] == n["seed_pairs.jsonl"] + n["curated_pairs.jsonl"]

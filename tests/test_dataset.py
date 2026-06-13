"""Validate the committed seed DPO dataset (CI-safe, no GPU/model)."""

from __future__ import annotations

from pathlib import Path

from model._common import load_jsonl

SEED = Path(__file__).resolve().parents[1] / "dataset" / "seed_pairs.jsonl"


def test_seed_pairs_load_and_are_well_formed():
    rows = load_jsonl(SEED)
    assert len(rows) >= 10, "seed set should have a usable number of pairs"
    for r in rows:
        assert {"prompt", "chosen", "rejected"} <= r.keys()
        assert r["prompt"].strip() and r["chosen"].strip() and r["rejected"].strip()
        # The chosen review should be more substantial than the generic rejected one.
        assert len(r["chosen"]) > len(r["rejected"])

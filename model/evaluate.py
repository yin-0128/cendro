"""Lightweight eval harness for a trained Cendro model.

Runs the model over held-out code samples and checks each review is non-empty and
references a concrete issue (a cheap proxy for "specific, not generic"). Optional
BERTScore against reference reviews when --references is given.

    python model/evaluate.py --model model/output/cendro-7b
    python model/evaluate.py --model model/output/cendro-7b --dataset dataset/seed_pairs.jsonl

    # Prove the fine-tune actually helped: score the BASE model on the same prompts and
    # print both side by side (use a HELD-OUT dataset not seen during training).
    python model/evaluate.py --model model/output/cendro-7b \
        --dataset dataset/heldout_pairs.jsonl --compare-base

Defaults to evaluating against the seed dataset's prompts/chosen reviews.
"""

from __future__ import annotations

import argparse

from model._common import load_jsonl

# Words that signal the review names a concrete problem rather than hand-waving.
SPECIFIC_MARKERS = (
    "bug", "error", "issue", "leak", "race", "O(", "complexity", "null", "none",
    "exception", "overflow", "injection", "validate", "boundary", "off-by", "mutat",
    "deadlock", "unbounded", "edge case", "security", "performance",
)


def looks_specific(review: str) -> bool:
    low = review.lower()
    return len(review.strip()) > 40 and any(m.lower() in low for m in SPECIFIC_MARKERS)


def _run(model, tokenizer, rows: list[dict], max_new_tokens: int) -> tuple[list[str], int]:
    """Generate a review for each prompt; return (reviews, count that look specific)."""
    import torch

    generated: list[str] = []
    specific_hits = 0
    for r in rows:
        messages = [{"role": "user", "content": r["prompt"]}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
        review = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        generated.append(review)
        if looks_specific(review):
            specific_hits += 1
    return generated, specific_hits


def _report(label: str, generated: list[str], hits: int, rows: list[dict], references: bool) -> None:
    n = len(rows)
    pct = 100 * hits / max(n, 1)
    print(f"[{label}] non-empty + specific: {hits}/{n} ({pct:.0f}%)")
    if references:
        from bert_score import score

        refs = [r["chosen"] for r in rows]
        _, _, f1 = score(generated, refs, lang="en", verbose=False)
        print(f"[{label}] BERTScore F1 vs chosen: {f1.mean().item():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Cendro model.")
    parser.add_argument("--model", required=True, help="Path to the trained adapter/model dir.")
    parser.add_argument("--dataset", default="dataset/seed_pairs.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument(
        "--references", action="store_true", help="Also compute BERTScore vs `chosen`."
    )
    parser.add_argument(
        "--compare-base", action="store_true",
        help="Also score the base model on the same prompts to show the fine-tune delta.",
    )
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_jsonl(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # bf16 only on Ampere+ (e.g. local 4060); fall back to fp16 on Turing (Colab T4).
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto"
    )
    base.eval()

    print(f"Samples evaluated: {len(rows)}")

    # Score the base model FIRST (before attaching the adapter) so we reuse one model copy
    # and stay inside 8GB VRAM.
    if args.compare_base:
        base_gen, base_hits = _run(base, tokenizer, rows, args.max_new_tokens)
        _report("base", base_gen, base_hits, rows, args.references)

    model = PeftModel.from_pretrained(base, args.model)
    model.eval()
    tuned_gen, tuned_hits = _run(model, tokenizer, rows, args.max_new_tokens)
    _report("cendro", tuned_gen, tuned_hits, rows, args.references)

    if args.compare_base:
        delta = 100 * (tuned_hits - base_hits) / max(len(rows), 1)
        print(f"Specificity delta (cendro - base): {delta:+.0f} percentage points")


if __name__ == "__main__":
    main()

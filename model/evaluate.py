"""Lightweight eval harness for a trained Cendro model.

Runs the model over held-out code samples and checks each review is non-empty and
references a concrete issue (a cheap proxy for "specific, not generic"). Optional
BERTScore against reference reviews when --references is given.

    python model/evaluate.py --model model/output/cendro-3b
    python model/evaluate.py --model model/output/cendro-3b --dataset dataset/seed_pairs.jsonl

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Cendro model.")
    parser.add_argument("--model", required=True, help="Path to the trained adapter/model dir.")
    parser.add_argument("--dataset", default="dataset/seed_pairs.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument(
        "--references", action="store_true", help="Also compute BERTScore vs `chosen`."
    )
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_jsonl(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, args.model)
    model.eval()

    generated: list[str] = []
    specific_hits = 0
    for r in rows:
        messages = [{"role": "user", "content": r["prompt"]}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        review = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        generated.append(review)
        if looks_specific(review):
            specific_hits += 1

    n = len(rows)
    print(f"Samples evaluated: {n}")
    print(f"Non-empty + specific: {specific_hits}/{n} ({100 * specific_hits / max(n, 1):.0f}%)")

    if args.references:
        from bert_score import score

        refs = [r["chosen"] for r in rows]
        _, _, f1 = score(generated, refs, lang="en", verbose=False)
        print(f"BERTScore F1 vs chosen: {f1.mean().item():.3f}")


if __name__ == "__main__":
    main()

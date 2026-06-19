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

    # The TRUSTWORTHY gate: a blind pairwise win-rate judged by a free Groq model, plus a
    # median-length report so a "win" that is just longer/wordier gets caught. Also dump the
    # base vs cendro generations side by side to eyeball them.
    export GROQ_API_KEY=gsk_...
    python model/evaluate.py --model model/output/cendro-7b \
        --dataset dataset/heldout_pairs.jsonl --judge --dump eval_dump.jsonl

Defaults to evaluating against the seed dataset's prompts/chosen reviews.

Note: the keyword `looks_specific` % and BERTScore are CHEAP SANITY CHECKS, not the gate.
A model that merely got longer/more keyword-heavy scores higher on them while being no
better — use --judge (blind win-rate) as the real quality signal.
"""

from __future__ import annotations

import argparse
import json
import os

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
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        review = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        generated.append(review)
        if looks_specific(review):
            specific_hits += 1
    return generated, specific_hits


def _report(
    label: str, generated: list[str], hits: int, rows: list[dict], references: bool
) -> None:
    n = len(rows)
    pct = 100 * hits / max(n, 1)
    print(f"[{label}] non-empty + specific: {hits}/{n} ({pct:.0f}%)")
    if references:
        from bert_score import score

        refs = [r["chosen"] for r in rows]
        _, _, f1 = score(generated, refs, lang="en", verbose=False)
        print(f"[{label}] BERTScore F1 vs chosen: {f1.mean().item():.3f}")


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Blind pairwise judge. The instruction explicitly tells the judge NOT to reward length, so a
# fine-tune that only learned to be wordier does not win here (that was the original failure).
_PAIRWISE_INSTRUCTION = """You are judging two code reviews of the SAME code. Pick the one that \
is more correct, specific, and actionable for a developer. Reward catching real bugs and giving \
correct fixes. Do NOT reward length or padding -- a short review that nails the real issue beats \
a long one that is vague, wrong, or invents problems. If they are genuinely equal, answer "tie".

Review task:
{prompt}

--- Review A ---
{a}

--- Review B ---
{b}

Respond with ONLY strict JSON: {{"winner": "A", "reason": "<one sentence>"}} where winner is \
"A", "B", or "tie"."""


def judge_pair(prompt: str, a: str, b: str, *, model: str) -> str:
    """Ask a free Groq judge which review (A or B) is better. Returns 'A', 'B', or 'tie'."""
    import urllib.request

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com and "
            "`export GROQ_API_KEY=gsk_...`."
        )
    content = _PAIRWISE_INSTRUCTION.format(prompt=prompt, a=a, b=b)
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        GROQ_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # Cloudflare in front of Groq blocks urllib's default UA with 403 (error 1010).
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Cendro/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    raw = body["choices"][0]["message"]["content"].strip()
    winner = str(json.loads(raw).get("winner", "tie")).strip().upper()
    return winner if winner in {"A", "B"} else "tie"


def win_rate(
    rows: list[dict], base_gen: list[str], tuned_gen: list[str], model: str, seed: int
) -> float:
    """Blind pairwise win-rate of cendro vs base via a Groq judge (randomized A/B order)."""
    import random

    rng = random.Random(seed)
    cendro_wins = base_wins = ties = 0
    for r, b, c in zip(rows, base_gen, tuned_gen, strict=False):
        cendro_is_a = rng.random() < 0.5  # randomize position to cancel A/B bias
        a_text, b_text = (c, b) if cendro_is_a else (b, c)
        try:
            winner = judge_pair(r["prompt"], a_text, b_text, model=model)
        except Exception as exc:  # one bad verdict shouldn't kill the run
            print(f"  judge skip: {exc}")
            ties += 1
            continue
        if winner == "tie":
            ties += 1
        elif (winner == "A") == cendro_is_a:
            cendro_wins += 1
        else:
            base_wins += 1
    decided = cendro_wins + base_wins
    rate = 100 * cendro_wins / decided if decided else 0.0
    print(f"[judge] cendro wins {cendro_wins}, base wins {base_wins}, ties {ties} (of {len(rows)})")
    print(f"[judge] cendro win-rate vs base (ties excluded): {rate:.0f}%  -- gate: >= ~60%")
    return rate


def length_report(base_gen: list[str] | None, tuned_gen: list[str]) -> None:
    """Median review length so a win that is merely longer/wordier is visible, not hidden."""
    import statistics

    cl = statistics.median(len(x) for x in tuned_gen)
    if base_gen:
        bl = statistics.median(len(x) for x in base_gen)
        ratio = f"{cl / bl:.2f}x" if bl else "inf"
        print(f"[length] median review chars -- base {bl:.0f}, cendro {cl:.0f} ({ratio})")
    else:
        print(f"[length] median review chars -- cendro {cl:.0f}")


def dump(path: str, rows: list[dict], base_gen: list[str] | None, tuned_gen: list[str]) -> None:
    """Write {prompt, base, cendro, chosen} per row so you can diff generations by eye."""
    with open(path, "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            rec = {"prompt": r["prompt"], "cendro": tuned_gen[i], "chosen": r.get("chosen", "")}
            if base_gen:
                rec["base"] = base_gen[i]
            fh.write(json.dumps(rec) + "\n")
    print(f"[dump] wrote {len(rows)} rows to {path}")


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
    parser.add_argument(
        "--judge", action="store_true",
        help="Blind pairwise win-rate vs base via a free Groq judge (the trustworthy gate). "
        "Needs GROQ_API_KEY.",
    )
    parser.add_argument(
        "--judge-model", default="llama-3.3-70b-versatile", help="Groq judge model id."
    )
    parser.add_argument(
        "--dump", default=None, help="Write {prompt, base, cendro, chosen} per row to this JSONL."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for judge A/B randomization.")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_jsonl(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # bf16 only on Ampere+ (e.g. local 4060); fall back to fp16 on Turing (Colab T4).
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Load 4-bit so the 7B fits in 8GB VRAM (4060 / Colab T4) with no slow CPU offload —
    # this also matches how Ollama serves the model, so the eval reflects real behavior.
    from transformers import BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto"
    )
    base.eval()

    print(f"Samples evaluated: {len(rows)}")

    # --judge and --dump both need the base generations alongside cendro's.
    need_base = args.compare_base or args.judge or args.dump

    # Score the base model FIRST (before attaching the adapter) so we reuse one model copy
    # and stay inside 8GB VRAM.
    base_gen: list[str] | None = None
    if need_base:
        base_gen, base_hits = _run(base, tokenizer, rows, args.max_new_tokens)
        if args.compare_base:
            _report("base", base_gen, base_hits, rows, args.references)

    model = PeftModel.from_pretrained(base, args.model)
    model.eval()
    tuned_gen, tuned_hits = _run(model, tokenizer, rows, args.max_new_tokens)
    _report("cendro", tuned_gen, tuned_hits, rows, args.references)

    if args.compare_base:
        delta = 100 * (tuned_hits - base_hits) / max(len(rows), 1)
        print(f"Specificity delta (cendro - base): {delta:+.0f} percentage points")

    if need_base:
        length_report(base_gen, tuned_gen)
    if args.dump:
        dump(args.dump, rows, base_gen, tuned_gen)
    if args.judge:
        win_rate(rows, base_gen, tuned_gen, args.judge_model, args.seed)


if __name__ == "__main__":
    main()

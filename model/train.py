"""QLoRA supervised fine-tune (SFT) warm-up.

Optional first stage before DPO: teaches the base model the Cendro review format on
{prompt, chosen} pairs (the `chosen` side of the preference data). 4-bit + LoRA, sized
for 8GB VRAM. Run DPO afterwards with dpo_train.py.

    python model/train.py --config configs/qlora_3b.yaml \
        --dataset dataset/seed_pairs.jsonl --output model/output/sft
"""

from __future__ import annotations

import argparse

from model._common import (
    Config,
    load_config,
    load_jsonl,
    load_quantized_model,
    load_tokenizer,
    lora_config,
)


def build_sft_dataset(rows: list[dict], tokenizer):
    """Turn {prompt, chosen} rows into chat-formatted text for SFT."""
    from datasets import Dataset

    texts = []
    for r in rows:
        # `prompt` is the user turn (see api/prompts.py); `chosen` is the target review.
        messages = [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["chosen"]},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return Dataset.from_dict({"text": texts})


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT warm-up for Cendro.")
    parser.add_argument("--config", default="configs/qlora_3b.yaml")
    parser.add_argument("--dataset", default="dataset/seed_pairs.jsonl")
    parser.add_argument("--output", default=None, help="Override output_dir from config.")
    args = parser.parse_args()

    cfg: Config = load_config(args.config)
    t = cfg.section("train")
    output_dir = args.output or t.get("output_dir", "model/output/sft")

    from peft import prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    tokenizer = load_tokenizer(cfg)
    model = prepare_model_for_kbit_training(load_quantized_model(cfg))
    dataset = build_sft_dataset(load_jsonl(args.dataset), tokenizer)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=t.get("num_train_epochs", 1),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        learning_rate=t.get("learning_rate", 1e-4),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        logging_steps=t.get("logging_steps", 5),
        save_steps=t.get("save_steps", 50),
        bf16=t.get("bf16", True),
        optim=t.get("optim", "paged_adamw_8bit"),
        max_length=t.get("max_length", 1024),
        seed=t.get("seed", 42),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config(cfg),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"SFT adapter saved to {output_dir}")


if __name__ == "__main__":
    main()

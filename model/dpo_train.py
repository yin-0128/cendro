"""DPO (Direct Preference Optimization) fine-tune.

Trains the model to prefer specific, opinionated reviews (`chosen`) over generic ones
(`rejected`) on {prompt, chosen, rejected} JSONL. 4-bit QLoRA, sized for 8GB VRAM.

    python model/dpo_train.py --config configs/qlora_7b.yaml \
        --dataset dataset/train_pairs.jsonl --output model/output/cendro-7b

Optionally start from an SFT adapter produced by train.py via --sft-adapter.
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


def build_dpo_dataset(rows: list[dict]):
    """DPOTrainer expects columns: prompt, chosen, rejected."""
    from datasets import Dataset

    return Dataset.from_list(
        [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO fine-tune for Cendro.")
    parser.add_argument("--config", default="configs/qlora_7b.yaml")
    parser.add_argument("--dataset", default="dataset/train_pairs.jsonl")
    parser.add_argument("--output", default=None, help="Override output_dir from config.")
    parser.add_argument(
        "--sft-adapter", default=None, help="Optional SFT LoRA adapter to start from."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in output_dir (e.g. after a Colab disconnect).",
    )
    args = parser.parse_args()

    cfg: Config = load_config(args.config)
    t = cfg.section("train")
    output_dir = args.output or t.get("output_dir", "model/output/cendro-7b")
    beta = cfg.section("dpo").get("beta", 0.1)

    from peft import PeftModel, prepare_model_for_kbit_training
    from trl import DPOConfig, DPOTrainer

    tokenizer = load_tokenizer(cfg)
    model = prepare_model_for_kbit_training(load_quantized_model(cfg))
    if args.sft_adapter:
        model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)

    dataset = build_dpo_dataset(load_jsonl(args.dataset))

    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=beta,
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
        max_prompt_length=t.get("max_prompt_length", 768),
        seed=t.get("seed", 42),
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        # peft_config wires LoRA when not resuming from an SFT adapter.
        peft_config=None if args.sft_adapter else lora_config(cfg),
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"DPO adapter saved to {output_dir}")


if __name__ == "__main__":
    main()

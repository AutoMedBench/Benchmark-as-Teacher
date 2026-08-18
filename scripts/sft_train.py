#!/usr/bin/env python3
"""One-time full-parameter SFT over an admitted tokenized BaT view."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_prepared(path: Path) -> list[dict[str, list[int]]]:
    rows: list[dict[str, list[int]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"prepared_line_{line_number}_invalid")
            fields = {key: value.get(key) for key in ("input_ids", "attention_mask", "labels")}
            if any(not isinstance(item, list) for item in fields.values()):
                raise ValueError(f"prepared_line_{line_number}_invalid")
            lengths = {len(item) for item in fields.values()}
            if len(lengths) != 1 or not lengths.pop() or all(label == -100 for label in fields["labels"]):
                raise ValueError(f"prepared_line_{line_number}_loss_mask_invalid")
            rows.append(fields)  # type: ignore[arg-type]
    if not rows:
        raise ValueError("prepared_dataset_empty")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--deepspeed")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train_rows = read_prepared(args.train_file.resolve(strict=True))
    validation_rows = read_prepared(args.validation_file.resolve(strict=True)) if args.validation_file and args.validation_file.stat().st_size else []
    if args.max_steps <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("training_arguments_invalid")
    if args.dry_run:
        print(json.dumps({"status": "validated", "train_rows": len(train_rows), "validation_rows": len(validation_rows)}, sort_keys=True))
        return 0
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ModuleNotFoundError as exc:
        raise RuntimeError("sft_dependencies_not_installed") from exc

    class Rows(Dataset):
        def __init__(self, values: list[dict[str, list[int]]]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {key: torch.tensor(value, dtype=torch.long) for key, value in self.values[index].items()}

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        maximum = max(item["input_ids"].shape[0] for item in batch)
        result: dict[str, list[Any]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in batch:
            pad = maximum - item["input_ids"].shape[0]
            result["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, pad), value=0))
            result["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (0, pad), value=0))
            result["labels"].append(torch.nn.functional.pad(item["labels"], (0, pad), value=-100))
        return {key: torch.stack(value) for key, value in result.items()}

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16,
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir.resolve()),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if validation_rows else "no",
        eval_steps=args.save_steps if validation_rows else None,
        logging_steps=1,
        report_to=[],
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Rows(train_rows),
        eval_dataset=Rows(validation_rows) if validation_rows else None,
        data_collator=collate,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = args.output_dir.resolve() / "final"
    trainer.save_model(str(final_dir))
    AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    ).save_pretrained(str(final_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

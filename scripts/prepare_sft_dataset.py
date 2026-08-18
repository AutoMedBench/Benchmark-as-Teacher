#!/usr/bin/env python3
"""Render model-neutral semantic SFT rows through one tokenizer chat template."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

from rl.grpo.tool_protocol_contract import validate_runtime_protocol
from scripts.sft_tool_protocol import ToolProtocolError, render_semantic_example
from scripts.tokenizer_compat import load_chat_tokenizer


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"source_line_{line_number}_invalid_json") from exc
            if not isinstance(value, dict):
                raise ValueError(f"source_line_{line_number}_not_object")
            rows.append(value)
    if not rows:
        raise ValueError("sft_source_empty")
    return rows


def prepare_rows(tokenizer: Any, rows: list[dict[str, Any]], *, max_tokens: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            projection = render_semantic_example(tokenizer, row.get("messages"), row.get("target"))
        except ToolProtocolError as exc:
            rejected.append({"row": index, "reason": str(exc)})
            continue
        if len(projection.full_ids) > max_tokens:
            rejected.append({"row": index, "reason": "complete_target_exceeds_max_tokens"})
            continue
        prompt_length = len(projection.full_ids) - len(projection.target_ids)
        labels = [-100] * prompt_length + list(projection.target_ids)
        if len(labels) != len(projection.full_ids) or all(value == -100 for value in labels):
            rejected.append({"row": index, "reason": "loss_mask_invalid"})
            continue
        source_digest = _sha256_bytes(_stable(row).encode("utf-8"))
        trajectory = row.get("metadata", {}).get("trajectory_id") if isinstance(row.get("metadata"), dict) else None
        prepared.append(
            {
                "input_ids": list(projection.full_ids),
                "attention_mask": [1] * len(projection.full_ids),
                "labels": labels,
                "metadata": {
                    "source_row_sha256": source_digest,
                    "trajectory_group": str(trajectory or source_digest),
                    "semantic_action_sha256": projection.semantic_action_sha256,
                    "supervised_tokens_sha256": _sha256_bytes(_stable(list(projection.target_ids)).encode("utf-8")),
                },
            }
        )
    return prepared, rejected


def split_by_trajectory(rows: list[dict[str, Any]], *, validation_fraction: float, seed: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction_invalid")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["metadata"]["trajectory_group"], []).append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    validation_groups = set(keys[: round(len(keys) * validation_fraction)])
    train = [row for key in keys if key not in validation_groups for row in groups[key]]
    validation = [row for key in keys if key in validation_groups for row in groups[key]]
    return train, validation


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise FileExistsError(f"refusing to overwrite: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((_stable(row) + "\n").encode("utf-8") for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-tool-format", required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", default="bat-sft-v1")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    runtime_format = validate_runtime_protocol(args.runtime_tool_format)
    source = args.source.resolve(strict=True)
    source_bytes = source.read_bytes()
    tokenizer = load_chat_tokenizer(args.model, trust_remote_code=args.trust_remote_code)
    prepared, rejected = prepare_rows(tokenizer, _read_rows(source), max_tokens=args.max_tokens)
    if not prepared:
        raise ValueError("no_sft_rows_prepared")
    train, validation = split_by_trajectory(prepared, validation_fraction=args.validation_fraction, seed=args.seed)
    output = args.output_dir.resolve()
    train_bytes, validation_bytes = _jsonl(train), _jsonl(validation)
    _write_immutable(output / "train.jsonl", train_bytes)
    _write_immutable(output / "validation.jsonl", validation_bytes)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "model_scoped_tokenized_sft",
        "source_sha256": _sha256_bytes(source_bytes),
        "model_reference_sha256": _sha256_bytes(args.model.encode("utf-8")),
        "runtime_tool_format": runtime_format,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": _sha256_bytes(tokenizer.chat_template.encode("utf-8")),
        "max_tokens": args.max_tokens,
        "rows": {"train": len(train), "validation": len(validation), "rejected": len(rejected)},
        "rejections": rejected,
        "train_sha256": _sha256_bytes(train_bytes),
        "validation_sha256": _sha256_bytes(validation_bytes),
        "model_specific_serialization_in_shared_source": False,
    }
    _write_immutable(output / "manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

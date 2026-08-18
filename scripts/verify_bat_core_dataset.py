#!/usr/bin/env python3
"""Verify round digests, schema, and fixed three-pool composition."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rl.grpo.core_reward import e2e_reward_contract, stage_reward_contract
from rl.grpo.core_task_contract import STAGES, TASK_DEFINITION_VERSION, validate_prompt_binding
from scripts.build_bat_core_dataset import verify_round_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line_{line_number}_not_object")
            values.append(value)
    return values


def verify(train: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("artifact_kind") != "bat_three_pool_round":
        raise ValueError("manifest_invalid")
    if _sha256(train) != manifest.get("train", {}).get("sha256"):
        raise ValueError("train_digest_mismatch")
    rows = _rows(train)
    target, batches = manifest.get("target_stage"), manifest.get("batches")
    verify_round_rows(rows, target, batches)
    for row in rows:
        extra = row.get("extra_info")
        if not isinstance(extra, dict) or extra.get("task_definition_version") != TASK_DEFINITION_VERSION:
            raise ValueError("row_task_contract_invalid")
        validate_prompt_binding(row.get("prompt"), extra)
        expected_reward = e2e_reward_contract() if extra["target_stage"] == "E2E" else stage_reward_contract(extra["target_stage"])
        if extra.get("reward_contract") != expected_reward:
            raise ValueError("row_reward_contract_invalid")
        if extra.get("target_stage") not in (*STAGES, "E2E"):
            raise ValueError("row_stage_invalid")
    return {"status": "passed", "rows": len(rows), "batches": batches, "target_stage": target, "train_sha256": _sha256(train)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.train.resolve(strict=True), args.manifest.resolve(strict=True)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

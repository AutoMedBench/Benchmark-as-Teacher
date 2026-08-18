#!/usr/bin/env python3
"""Validate checkpoint inheritance and optimizer settings for a BaT round."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.bind_bat_gate_model import build_binding


def validate_round_start(
    prescription_path: Path,
    model_path: Path,
    *,
    optimizer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = json.loads(prescription_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("artifact_kind") != "bat_round_prescription" or payload.get("schema_version") != 1:
        raise ValueError("round_prescription_invalid")
    binding = build_binding(model_path, experiment_name="round-start", gate_step=0, arm="baseline")
    expected = payload.get("starting_checkpoint_sha256")
    if binding["model_identity_sha256"] != expected:
        raise ValueError("round_checkpoint_inheritance_mismatch")
    prescribed_optimizer = payload.get("optimizer")
    if not isinstance(prescribed_optimizer, dict):
        raise ValueError("round_optimizer_missing")
    if optimizer is not None and dict(optimizer) != prescribed_optimizer:
        raise ValueError("round_optimizer_mismatch")
    return {
        "status": "passed",
        "target_stage": payload.get("target_stage"),
        "checkpoint_sha256": expected,
        "optimizer": prescribed_optimizer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prescription", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_round_start(args.prescription, args.model_path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

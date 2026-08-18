#!/usr/bin/env python3
"""Write an immutable launch receipt after all BaT round-start checks pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from rl.grpo.tool_protocol_contract import runtime_protocol_binding
from scripts.validate_bat_core_round_start import validate_round_start
from scripts.verify_bat_core_dataset import verify


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_round_start_contract(
    *,
    model_path: Path,
    train_path: Path,
    manifest_path: Path,
    prescription_path: Path,
    runtime_tool_format: str,
) -> dict[str, Any]:
    start = validate_round_start(prescription_path, model_path)
    dataset = verify(train_path, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("target_stage") != start.get("target_stage"):
        raise ValueError("dataset_target_disagrees_with_prescription")
    return {
        "schema_version": 1,
        "artifact_kind": "bat_round_start_contract",
        "checkpoint_sha256": start["checkpoint_sha256"],
        "train_sha256": dataset["train_sha256"],
        "manifest_sha256": _sha256(manifest_path),
        "prescription_sha256": _sha256(prescription_path),
        "target_stage": start["target_stage"],
        "optimizer": start["optimizer"],
        "runtime_protocol": runtime_protocol_binding(runtime_tool_format),
    }


def write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise FileExistsError(f"refusing to overwrite: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prescription", type=Path, required=True)
    parser.add_argument("--runtime-tool-format", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_round_start_contract(model_path=args.model_path, train_path=args.train, manifest_path=args.manifest, prescription_path=args.prescription, runtime_tool_format=args.runtime_tool_format)
    write_immutable(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a path-free request contract for an external AutoMedBench evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def build_manifest(binding: Mapping[str, Any], *, protocol_id: str, repeats: int = 1) -> dict[str, Any]:
    if binding.get("artifact_kind") != "bat_gate_model_binding" or not isinstance(binding.get("model_identity_sha256"), str):
        raise ValueError("checkpoint_binding_invalid")
    if not isinstance(protocol_id, str) or not protocol_id.strip() or repeats < 1:
        raise ValueError("evaluation_request_invalid")
    return {
        "schema_version": 1,
        "artifact_kind": "bat_eval_request",
        "checkpoint_sha256": binding["model_identity_sha256"],
        "protocol_id": protocol_id.strip(),
        "repeats": repeats,
        "required_output": {
            "schema_version": 1,
            "status": "complete",
            "cells": {
                "identity_fields": ["track", "task_id", "repeat"],
                "score_fields": ["task_score", "stages"],
                "stage_fields": ["S1", "S2", "S3", "S4", "S5"],
                "valid_must_equal": True,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    manifest = build_manifest(binding, protocol_id=args.protocol_id, repeats=args.repeats)
    content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if args.output.exists() and args.output.read_bytes() != content:
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run an external evaluator and validate its output into BaT aggregates."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from scripts.bind_bat_gate_model import build_binding, write_immutable
from scripts.build_bat_core_eval_manifest import build_manifest
from scripts.summarize_bat_core_eval import build_summary


def _expand(command: Sequence[str], *, checkpoint: Path, output: Path, request: Path) -> list[str]:
    if not command:
        raise ValueError("evaluator_command_empty")
    replacements = {"{checkpoint}": str(checkpoint), "{output}": str(output), "{request}": str(request)}
    expanded = [replacements.get(item, item) for item in command]
    if str(output) not in expanded or str(checkpoint) not in expanded or str(request) not in expanded:
        raise ValueError("evaluator_command_placeholders_missing")
    return expanded


def run_evaluation(
    *,
    checkpoint: Path,
    output_dir: Path,
    protocol_id: str,
    command: Sequence[str],
    repeats: int = 1,
    experiment_name: str = "bat-eval",
    gate_step: int = 0,
    arm: str = "candidate",
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    binding = build_binding(checkpoint, experiment_name=experiment_name, gate_step=gate_step, arm=arm)
    binding_path = output_dir / "checkpoint-binding.json"
    write_immutable(binding_path, binding)
    request_payload = build_manifest(binding, protocol_id=protocol_id, repeats=repeats)
    request_path = output_dir / "eval-request.json"
    request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw_path = output_dir / "raw-evaluation.json"
    expanded = _expand(command, checkpoint=checkpoint, output=raw_path, request=request_path)
    completed = subprocess.run(expanded, stdin=subprocess.DEVNULL, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"external_evaluator_failed:{completed.returncode}")
    if not raw_path.is_file():
        raise RuntimeError("external_evaluator_output_missing")
    summary = build_summary(raw_path, checkpoint_sha256=binding["model_identity_sha256"])
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "artifact_kind": "bat_eval_run_receipt",
        "checkpoint_sha256": binding["model_identity_sha256"],
        "protocol_id": protocol_id,
        "command_sha256": hashlib.sha256(json.dumps(list(command), separators=(",", ":")).encode("utf-8")).hexdigest(),
        "raw_evaluation_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "status": "valid",
    }
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--experiment-name", default="bat-eval")
    parser.add_argument("--gate-step", type=int, default=0)
    parser.add_argument("--arm", choices=("baseline", "candidate"), default="candidate")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    summary = run_evaluation(checkpoint=args.checkpoint, output_dir=args.output_dir, protocol_id=args.protocol_id, command=command, repeats=args.repeats, experiment_name=args.experiment_name, gate_step=args.gate_step, arm=args.arm)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

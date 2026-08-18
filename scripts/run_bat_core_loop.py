#!/usr/bin/env python3
"""Orchestrate repeated evaluate-route-GRPO-evaluate-gate BaT rounds.

The evaluator and trainer remain explicit argv templates.  No shell is used,
and only aggregate stage selection crosses from evaluation into data routing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl.grpo.tool_protocol_contract import validate_runtime_protocol
from scripts.build_bat_core_dataset import (
    _stable,
    build_round,
    read_source,
    verify_round_rows,
)
from scripts.check_bat_core_gate import evaluate_gate, select_target
from scripts.run_bat_core_eval import run_evaluation


def _command(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise ValueError("command_template_invalid")
    return value


def _expand(template: Sequence[str], values: Mapping[str, str], required: Sequence[str]) -> list[str]:
    expanded = [values.get(item, item) for item in template]
    if any(values[key] not in expanded for key in required):
        raise ValueError("trainer_command_placeholders_missing")
    return expanded


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _materialize_round(
    *,
    source: Path,
    output_dir: Path,
    round_id: str,
    target_stage: str,
    batches: int,
) -> tuple[Path, Path]:
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = build_round(read_source(source), round_id=round_id, target_stage=target_stage, batches=batches, source_sha256=source_sha256)
    verify_round_rows(rows, target_stage, batches)
    output_dir.mkdir(parents=True, exist_ok=False)
    train = output_dir / "train.jsonl"
    content = b"".join((_stable(row) + "\n").encode("utf-8") for row in rows)
    train.write_bytes(content)
    manifest = output_dir / "manifest.json"
    _write(
        manifest,
        {
            "schema_version": 1,
            "artifact_kind": "bat_three_pool_round",
            "round_id": round_id,
            "target_stage": target_stage,
            "batches": batches,
            "rows": len(rows),
            "rows_per_batch": 8,
            "continuations_per_state": 4,
            "pool_rows_per_batch": {"s_target": 4, "s_mix": 2, "e2e": 2},
            "source": {"file": source.name, "sha256": source_sha256, "rows": 126},
            "train": {"file": train.name, "sha256": hashlib.sha256(content).hexdigest()},
            "model_specific_serialization": False,
            "assistant_targets_included": False,
        },
    )
    return train, manifest


def run_loop(
    *,
    source: Path,
    initial_checkpoint: Path,
    work_dir: Path,
    evaluator_command: Sequence[str],
    trainer_command: Sequence[str],
    protocol_id: str,
    runtime_tool_format: str,
    rounds: int,
    batches: int,
) -> dict[str, Any]:
    if rounds < 1 or batches < 1:
        raise ValueError("loop_size_invalid")
    validate_runtime_protocol(runtime_tool_format)
    source = source.resolve(strict=True)
    current_checkpoint = initial_checkpoint.resolve(strict=True)
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    baseline_summary = run_evaluation(
        checkpoint=current_checkpoint,
        output_dir=work_dir / "baseline-eval",
        protocol_id=protocol_id,
        command=evaluator_command,
        experiment_name="bat-loop",
        gate_step=0,
        arm="baseline",
    )
    target_stage, _ = select_target(
        baseline_summary["stage_scores"],
        baseline_summary["stage_scores"],
        promoted=True,
    )
    history: list[dict[str, Any]] = []
    for round_number in range(1, rounds + 1):
        round_id = f"r{round_number:02d}"
        root = work_dir / round_id
        root.mkdir()
        train, manifest = _materialize_round(
            source=source,
            output_dir=root / "data",
            round_id=round_id,
            target_stage=target_stage,
            batches=batches,
        )
        candidate = root / "candidate"
        health = root / "training-health.json"
        values = {
            "{checkpoint}": str(current_checkpoint),
            "{dataset}": str(train),
            "{manifest}": str(manifest),
            "{output}": str(candidate),
            "{health}": str(health),
            "{round_id}": round_id,
            "{target_stage}": target_stage,
            "{runtime_tool_format}": runtime_tool_format,
        }
        command = _expand(
            trainer_command,
            values,
            required=("{checkpoint}", "{dataset}", "{output}", "{health}", "{runtime_tool_format}"),
        )
        completed = subprocess.run(command, stdin=subprocess.DEVNULL, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"trainer_failed:{round_id}:{completed.returncode}")
        if not candidate.is_dir() or not health.is_file():
            raise RuntimeError(f"trainer_artifacts_missing:{round_id}")
        health_payload = json.loads(health.read_text(encoding="utf-8"))
        candidate_summary = run_evaluation(
            checkpoint=candidate,
            output_dir=root / "candidate-eval",
            protocol_id=protocol_id,
            command=evaluator_command,
            experiment_name="bat-loop",
            gate_step=50,
            arm="candidate",
        )
        gate, prescription = evaluate_gate(
            baseline_summary,
            candidate_summary,
            training_healthy=isinstance(health_payload, dict) and health_payload.get("status") == "passed",
        )
        _write(root / "gate.json", gate)
        _write(root / "prescription.json", prescription)
        history.append(
            {
                "round_id": round_id,
                "target_stage": target_stage,
                "gate_status": gate["status"],
                "starting_checkpoint_sha256": prescription["starting_checkpoint_sha256"],
                "candidate_checkpoint_sha256": gate["candidate_checkpoint_sha256"],
            }
        )
        if gate["promoted"]:
            current_checkpoint = candidate
            baseline_summary = candidate_summary
        target_stage = prescription["target_stage"]
        _write(
            work_dir / "state.json",
            {
                "schema_version": 1,
                "completed_rounds": round_number,
                "current_checkpoint_sha256": baseline_summary["checkpoint_sha256"],
                "next_target_stage": target_stage,
                "history": history,
            },
        )
    return {
        "status": "complete",
        "rounds": rounds,
        "current_checkpoint_sha256": baseline_summary["checkpoint_sha256"],
        "next_target_stage": target_stage,
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--evaluator-command", type=Path, required=True)
    parser.add_argument("--trainer-command", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--runtime-tool-format", required=True)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--batches", type=int, default=378)
    args = parser.parse_args()
    result = run_loop(
        source=args.source,
        initial_checkpoint=args.initial_checkpoint,
        work_dir=args.work_dir,
        evaluator_command=_command(args.evaluator_command),
        trainer_command=_command(args.trainer_command),
        protocol_id=args.protocol_id,
        runtime_tool_format=args.runtime_tool_format,
        rounds=args.rounds,
        batches=args.batches,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

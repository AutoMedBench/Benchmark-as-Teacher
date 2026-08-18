#!/usr/bin/env python3
"""Apply paired BaT promotion gates and select the next target stage."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from rl.reward.automedbench import STAGES, finite_stage_scores


THRESHOLDS = {
    "maximum_overall_drop": 0.01,
    "maximum_task_drop": 0.03,
    "maximum_per_track_drop": 0.05,
    "maximum_late_stage_drop": 0.03,
    "catastrophic_overall_drop": 0.20,
}


def _summary(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("artifact_kind") != "bat_eval_summary" or value.get("status") != "valid":
        raise ValueError("eval_summary_invalid")
    means = value.get("means")
    if not isinstance(means, Mapping) or any(not isinstance(means.get(key), (int, float)) or not math.isfinite(float(means[key])) for key in ("task", "agentic", "overall")):
        raise ValueError("eval_means_invalid")
    finite_stage_scores(value.get("stage_scores"))
    checkpoint = value.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or len(checkpoint) != 64:
        raise ValueError("eval_checkpoint_binding_invalid")
    cells = value.get("cells")
    tracks = value.get("track_scores")
    if not isinstance(cells, list) or not cells or not isinstance(tracks, Mapping) or not tracks:
        raise ValueError("eval_pairing_evidence_invalid")
    if any(
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
        for score in tracks.values()
    ):
        raise ValueError("eval_track_scores_invalid")
    return dict(value)


def _cell_keys(summary: Mapping[str, Any]) -> set[tuple[str, str, int]]:
    return {(cell["track"], cell["task_id"], cell["repeat"]) for cell in summary["cells"]}


def select_target(baseline_stages: Mapping[str, Any], candidate_stages: Mapping[str, Any], *, promoted: bool) -> tuple[str, str]:
    baseline, candidate = finite_stage_scores(baseline_stages), finite_stage_scores(candidate_stages)
    if not promoted:
        regression = {stage: baseline[stage] - candidate[stage] for stage in STAGES}
        positive = [stage for stage in STAGES if regression[stage] > 0]
        if positive:
            maximum = max(regression[stage] for stage in positive)
            return next(stage for stage in STAGES if math.isclose(regression[stage], maximum, abs_tol=1e-12)), "largest_positive_regression"
    minimum = min(candidate.values())
    return next(stage for stage in STAGES if math.isclose(candidate[stage], minimum, abs_tol=1e-12)), "lowest_candidate_stage"


def evaluate_gate(baseline_value: Mapping[str, Any], candidate_value: Mapping[str, Any], *, training_healthy: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline, candidate = _summary(baseline_value), _summary(candidate_value)
    if _cell_keys(baseline) != _cell_keys(candidate):
        raise ValueError("paired_cells_mismatch")
    overall_drop = float(baseline["means"]["overall"]) - float(candidate["means"]["overall"])
    task_drop = float(baseline["means"]["task"]) - float(candidate["means"]["task"])
    baseline_tracks, candidate_tracks = baseline.get("track_scores"), candidate.get("track_scores")
    if not isinstance(baseline_tracks, Mapping) or not isinstance(candidate_tracks, Mapping) or set(baseline_tracks) != set(candidate_tracks):
        raise ValueError("paired_tracks_mismatch")
    track_drops = {track: float(baseline_tracks[track]) - float(candidate_tracks[track]) for track in baseline_tracks}
    stage_drops = {stage: float(baseline["stage_scores"][stage]) - float(candidate["stage_scores"][stage]) for stage in STAGES}
    checks = {
        "paired_cells": True,
        "training_health": training_healthy,
        "overall_noninferiority": overall_drop <= THRESHOLDS["maximum_overall_drop"],
        "task_noninferiority": task_drop <= THRESHOLDS["maximum_task_drop"],
        "per_track_regression": max(track_drops.values(), default=0.0) <= THRESHOLDS["maximum_per_track_drop"],
        "late_stage_regression": max(stage_drops[stage] for stage in ("S3", "S4", "S5")) <= THRESHOLDS["maximum_late_stage_drop"],
        "catastrophic_forgetting": overall_drop < THRESHOLDS["catastrophic_overall_drop"],
    }
    promoted = all(checks.values())
    target, reason = select_target(baseline["stage_scores"], candidate["stage_scores"], promoted=promoted)
    gate = {
        "schema_version": 1,
        "artifact_kind": "bat_promotion_gate",
        "status": "promote" if promoted else "hold",
        "promoted": promoted,
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "checks": checks,
        "deltas": {"overall_drop": overall_drop, "task_drop": task_drop, "track_drops": track_drops, "stage_drops": stage_drops},
        "thresholds": dict(THRESHOLDS),
    }
    prescription = {
        "schema_version": 1,
        "artifact_kind": "bat_round_prescription",
        "target_stage": target,
        "selection_reason": reason,
        "starting_checkpoint_sha256": candidate["checkpoint_sha256"] if promoted else baseline["checkpoint_sha256"],
        "promotion_status": gate["status"],
        "pool_rows_per_batch": {"s_target": 4, "s_mix": 2, "e2e": 2},
        "optimizer": {
            "learning_rate": 2.5e-7,
            "kl_loss_coefficient": 0.006,
            "normalize_advantage_by_std": False,
            "loss_aggregation": "sequence_mean_then_token_mean",
            "steps": 50,
            "continuations_per_state": 4,
        },
        "aggregate_stage_scores": candidate["stage_scores"],
    }
    return gate, prescription


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--training-health", type=Path)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--prescription-output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    healthy = True
    if args.training_health:
        health = json.loads(args.training_health.read_text(encoding="utf-8"))
        healthy = isinstance(health, dict) and health.get("status") == "passed"
    gate, prescription = evaluate_gate(baseline, candidate, training_healthy=healthy)
    for path, payload in ((args.gate_output, gate), (args.prescription_output, prescription)):
        content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    print(json.dumps({"gate": gate, "prescription": prescription}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

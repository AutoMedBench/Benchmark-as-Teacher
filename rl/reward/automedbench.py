"""Deterministic AutoMedBench stage aggregation used by BaT rewards and gates."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


STAGES = ("S1", "S2", "S3", "S4", "S5")
STAGE_WEIGHTS = {"S1": 0.25, "S2": 0.15, "S3": 0.35, "S4": 0.15, "S5": 0.10}


def unit(value: Any, label: str = "score") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}_invalid")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}_invalid") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label}_invalid")
    return result


def stage_check_score(
    evidence: Mapping[str, Any],
    stage: str,
    item_ids: Sequence[str],
) -> tuple[float, dict[str, float]]:
    if stage not in STAGES:
        raise ValueError("stage_invalid")
    checks_by_stage = evidence.get("checks")
    checks = checks_by_stage.get(stage) if isinstance(checks_by_stage, Mapping) else None
    if not isinstance(checks, Mapping) or not item_ids:
        raise ValueError("stage_evidence_missing")
    if set(checks) != set(item_ids):
        raise ValueError("stage_rubric_evidence_mismatch")
    scores = {item: float(checks.get(item) is True) for item in item_ids}
    return sum(scores.values()) / len(scores), scores


def compute_agentic_score(
    stage_scores: Mapping[str, Any],
    *,
    applicable: Sequence[str] | None = None,
) -> float:
    selected = tuple(applicable or STAGES)
    if not selected or any(stage not in STAGES for stage in selected):
        raise ValueError("applicable_stages_invalid")
    denominator = sum(STAGE_WEIGHTS[stage] for stage in selected)
    return sum(unit(stage_scores[stage], stage) * STAGE_WEIGHTS[stage] for stage in selected) / denominator


def strict_overall(task_score: Any, agentic_score: Any) -> float:
    return 0.5 * unit(task_score, "task_score") + 0.5 * unit(agentic_score, "agentic_score")


def finite_stage_scores(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(STAGES):
        raise ValueError("stage_scores_invalid")
    return {stage: unit(value[stage], stage) for stage in STAGES}


__all__ = [
    "STAGES",
    "STAGE_WEIGHTS",
    "compute_agentic_score",
    "finite_stage_scores",
    "stage_check_score",
    "strict_overall",
    "unit",
]

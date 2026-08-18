"""Contract-bound deterministic reward for BaT stage and E2E rollouts."""
from __future__ import annotations

from typing import Any, Mapping

from rl.grpo.core_task_contract import STAGES, stable_sha256, task_contract_from_extra
from rl.reward.automedbench import STAGE_WEIGHTS, stage_check_score


REWARD_VERSION = "bat_core_reward_v1"
_ITEMS: dict[str, tuple[str, ...]] = {
    "S1": ("plan_exists", "pipeline_sections_present"),
    "S2": (
        "environment_ready",
        "compatibility_checked",
        "model_loaded",
        "checkpoint_declared",
        "pipeline_exists",
    ),
    "S3": ("small_case_executed", "validation_receipt_valid"),
    "S4": ("full_inference_complete", "completion_receipt_valid"),
    "S5": ("submission_complete", "submission_receipt_valid"),
}


def stage_reward_contract(stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError("stage_invalid")
    return {
        "schema_version": 1,
        "mode": "bat_stage_rubric",
        "target_stage": stage,
        "items": [{"id": item, "weight": 1.0} for item in _ITEMS[stage]],
        "aggregation": "equal_weight_mean",
        "verified_tool_execution_required": True,
    }


def e2e_reward_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "bat_e2e_rubric",
        "target_stage": "E2E",
        "stages": [
            {
                "id": stage,
                "weight": STAGE_WEIGHTS[stage],
                "items": [{"id": item, "weight": 1.0} for item in _ITEMS[stage]],
            }
            for stage in STAGES
        ],
        "stage_mean_weight": 0.5,
        "final_outcome_weight": 0.5,
        "verified_tool_execution_required": True,
    }


def _without_digest(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(field, None)
    return result


def _authenticated_runtime(extra: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = extra.get("bat_agentic_summary")
    evidence = extra.get("bat_sandbox_evidence")
    if not isinstance(summary, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError("sandbox_evidence_missing")
    summary, evidence = dict(summary), dict(evidence)
    if summary.get("summary_sha256") != stable_sha256(_without_digest(summary, "summary_sha256")):
        raise ValueError("agentic_summary_digest_invalid")
    if evidence.get("evidence_sha256") != stable_sha256(_without_digest(evidence, "evidence_sha256")):
        raise ValueError("sandbox_evidence_digest_invalid")
    if summary.get("evidence_sha256") != evidence.get("evidence_sha256"):
        raise ValueError("sandbox_summary_binding_invalid")
    contract = task_contract_from_extra(extra)
    public_contract = evidence.get("contract")
    if not isinstance(public_contract, Mapping):
        raise ValueError("sandbox_contract_missing")
    if public_contract.get("contract_sha256") != contract["contract_sha256"]:
        raise ValueError("sandbox_contract_mismatch")
    if summary.get("contract_sha256") != contract["contract_sha256"]:
        raise ValueError("summary_contract_mismatch")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifact_evidence_missing")
    return summary, dict(artifacts)


def _expected_contract(extra: Mapping[str, Any]) -> dict[str, Any]:
    target = str(extra.get("target_stage") or "").upper()
    return e2e_reward_contract() if target == "E2E" else stage_reward_contract(target)


def _validate_reward_contract(extra: Mapping[str, Any]) -> dict[str, Any]:
    observed = extra.get("reward_contract")
    expected = _expected_contract(extra)
    if observed != expected:
        raise ValueError("reward_contract_mismatch")
    return expected


def _score_stage(artifacts: Mapping[str, Any], stage: str) -> tuple[float, dict[str, float]]:
    return stage_check_score(artifacts, stage, _ITEMS[stage])


def compute_score_with_breakdown(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: Mapping[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Score only authenticated runtime evidence; prose cannot earn credit."""

    del data_source, solution_str, ground_truth
    extra = dict(extra_info or {})
    target = str(extra.get("target_stage") or "").upper()
    try:
        _validate_reward_contract(extra)
        summary, artifacts = _authenticated_runtime(extra)
        if target == "E2E":
            stage_scores: dict[str, float] = {}
            item_scores: dict[str, float] = {}
            for stage in STAGES:
                stage_score, items = _score_stage(artifacts, stage)
                stage_scores[stage] = stage_score
                item_scores.update({f"{stage}.{key}": value for key, value in items.items()})
            stage_mean = sum(STAGE_WEIGHTS[stage] * stage_scores[stage] for stage in STAGES)
            final_outcome_value = artifacts.get("task_score")
            if (
                isinstance(final_outcome_value, bool)
                or not isinstance(final_outcome_value, (int, float))
                or not 0.0 <= float(final_outcome_value) <= 1.0
            ):
                raise ValueError("task_score_invalid")
            final_outcome = float(final_outcome_value)
            raw = 0.5 * stage_mean + 0.5 * final_outcome
        else:
            raw, item_scores = _score_stage(artifacts, target)
            stage_scores = {target: raw}
        successful_calls = int(summary.get("successful_tool_calls") or 0)
        workspace_changed = summary.get("workspace_changed") is True
        execution_verified = successful_calls > 0 and workspace_changed
        cap = 1.0 if execution_verified else 0.0
        score = min(raw, cap)
        result = {
            "score": score,
            "raw_score": raw,
            "gate_cap": cap,
            "gate_reason": None if execution_verified else "verified_execution_missing",
            "mode": "bat_core_rubric",
            "target_stage": target,
            "reward_version": REWARD_VERSION,
            "item_scores": item_scores,
            "stage_scores": stage_scores,
            "agentic_tool_calls": int(summary.get("tool_calls") or 0),
            "agentic_successful_tool_calls": successful_calls,
            "agentic_workspace_changed": workspace_changed,
            "agentic_current_stage_complete": summary.get("current_stage_complete") is True,
            "contract_sha256": summary.get("contract_sha256"),
            "evidence_sha256": summary.get("evidence_sha256"),
            "verified": True,
        }
    except (KeyError, TypeError, ValueError) as exc:
        result = {
            "score": 0.0,
            "raw_score": 0.0,
            "gate_cap": 0.0,
            "gate_reason": "reward_evidence_invalid",
            "mode": "bat_core_rubric",
            "target_stage": target or None,
            "reward_version": REWARD_VERSION,
            "item_scores": {},
            "stage_scores": {},
            "verified": False,
            "error": str(exc),
        }
    return result


def compute_score(*args: Any, **kwargs: Any) -> float:
    return float(compute_score_with_breakdown(*args, **kwargs)["score"])


def is_verified_reward_payload(payload: Any) -> bool:
    return bool(
        isinstance(payload, Mapping)
        and payload.get("verified") is True
        and payload.get("reward_version") == REWARD_VERSION
        and isinstance(payload.get("score"), (int, float))
        and 0.0 <= float(payload["score"]) <= 1.0
        and isinstance(payload.get("contract_sha256"), str)
        and isinstance(payload.get("evidence_sha256"), str)
    )


__all__ = [
    "REWARD_VERSION",
    "compute_score",
    "compute_score_with_breakdown",
    "e2e_reward_contract",
    "is_verified_reward_payload",
    "stage_reward_contract",
]

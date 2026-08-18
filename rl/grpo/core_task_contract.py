"""Authoritative track-native task contracts for agentic core BaT rounds.

The dataset builder, sandbox materializer, reward evidence, and admission
verifier all consume this module.  A training row is therefore admitted only
when its visible prompt describes the exact files and output schema that the
sandbox will materialize.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence


TASK_DEFINITION_VERSION = "bat_core_track_native_v3"
PROMPT_VERSION = "bat_core_agentic_prompt_v7_execute_code_arguments"
AGENTIC_BUDGET_VERSION = "bat_core_multiturn_budget_v3"
AGENTIC_MAX_ASSISTANT_TURNS = 24
AGENTIC_MAX_USER_TURNS = 23
AGENTIC_MAX_TOOL_CALLS = 23
AGENTIC_MAX_RESPONSE_TOKENS = 16384
AGENTIC_MAX_TOOL_RESPONSE_CHARS = 2048
AGENTIC_MAX_PROMPT_TOKENS = 4096
AGENTIC_MAX_MODEL_TOKENS = 20480
TRACKS = (
    "classification",
    "detection",
    "enhancement",
    "report",
    "segmentation",
    "synthesis",
    "vqa",
)
STAGES = ("S1", "S2", "S3", "S4", "S5")
ALLOWED_STAGES = ("E2E", *STAGES)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")


class CoreTaskContractError(ValueError):
    """A row does not bind to the authoritative task definition."""


_TASKS: dict[str, dict[str, Any]] = {
    "classification": {
        "task_name": "synthetic lesion-risk classification",
        "objective": (
            "fit a reproducible binary classifier from labeled numeric lesion records "
            "and predict benign versus suspicious risk for every test record"
        ),
        "inventory": ("data/train.csv", "data/test.csv"),
        "id_column": "sample_id",
        "required_columns": ("sample_id", "predicted_label", "confidence"),
        "output_kind": "classification_rows",
        "allowed_values": {"predicted_label": ("benign", "suspicious")},
    },
    "detection": {
        "task_name": "synthetic lesion localization",
        "objective": (
            "use labeled training images and bounding boxes to localize the bright lesion "
            "in every test image"
        ),
        "inventory": (
            "data/train_annotations.csv",
            "data/test_manifest.csv",
            "data/images/",
        ),
        "id_column": "sample_id",
        "required_columns": (
            "sample_id",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "confidence",
        ),
        "output_kind": "detection_boxes",
        "allowed_values": {},
    },
    "enhancement": {
        "task_name": "synthetic low-dose image denoising",
        "objective": (
            "learn a non-identity denoising procedure from paired noisy and clean images "
            "and enhance every noisy test image"
        ),
        "inventory": (
            "data/train_manifest.csv",
            "data/test_manifest.csv",
            "data/noisy/",
            "data/clean/",
        ),
        "id_column": "sample_id",
        "required_columns": ("sample_id", "output_path"),
        "output_kind": "enhanced_image_paths",
        "allowed_values": {},
        "output_shape": (8, 8),
    },
    "report": {
        "task_name": "synthetic radiology-report normalization",
        "objective": (
            "derive a reproducible text-cleaning pipeline from paired examples and produce "
            "one normalized report for every test record"
        ),
        "inventory": ("data/train_reports.csv", "data/test_reports.csv"),
        "id_column": "report_id",
        "required_columns": ("report_id", "cleaned_text"),
        "output_kind": "cleaned_report_rows",
        "allowed_values": {},
    },
    "segmentation": {
        "task_name": "synthetic lesion segmentation",
        "objective": (
            "use paired training images and binary masks to create one 8-by-8 binary mask "
            "for every test image"
        ),
        "inventory": (
            "data/train_manifest.csv",
            "data/test_manifest.csv",
            "data/images/",
            "data/masks/",
        ),
        "id_column": "sample_id",
        "required_columns": ("sample_id", "mask_path"),
        "output_kind": "segmentation_mask_paths",
        "allowed_values": {},
        "output_shape": (8, 8),
    },
    "synthesis": {
        "task_name": "synthetic image super-resolution",
        "objective": (
            "use paired low- and high-resolution training images to synthesize one 8-by-8 "
            "high-resolution image for every 4-by-4 test input"
        ),
        "inventory": (
            "data/train_manifest.csv",
            "data/test_manifest.csv",
            "data/low_res/",
            "data/high_res/",
        ),
        "id_column": "sample_id",
        "required_columns": ("sample_id", "output_path"),
        "output_kind": "synthesized_image_paths",
        "allowed_values": {},
        "output_shape": (8, 8),
    },
    "vqa": {
        "task_name": "synthetic medical visual question answering",
        "objective": (
            "learn the labeled visual marker rule and answer every yes-or-no question "
            "associated with a test image"
        ),
        "inventory": (
            "data/train_questions.csv",
            "data/test_questions.csv",
            "data/images/",
        ),
        "id_column": "question_id",
        "required_columns": ("question_id", "answer"),
        "output_kind": "vqa_answer_rows",
        "allowed_values": {"answer": ("yes", "no")},
    },
}

STAGE_REQUIRED_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "S1": ("plan/plan.md",),
    "S2": (
        "setup/environment.json",
        "setup/compatibility.json",
        "model/checkpoint.json",
        "setup/model_load.json",
        "pipeline/infer.py",
    ),
    "S3": ("validation/pilot.csv", "validation/s3_validation.json"),
    "S4": ("outputs/predictions.csv", "validation/s4_completion.json"),
    "S5": ("outputs/submission.csv", "validation/s5_submission.json"),
}

_STAGE_NAMES = {
    "S1": "plan",
    "S2": "setup",
    "S3": "validate one case",
    "S4": "run complete inference",
    "S5": "validate and submit",
    "E2E": "complete plan through submission",
}


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CoreTaskContractError("task_contract_not_stable_json") from exc


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise CoreTaskContractError(f"{label}_invalid")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CoreTaskContractError(f"{label}_invalid")
    return value


def expected_ids(track: str) -> tuple[str, ...]:
    if track == "vqa":
        prefix = "question"
    elif track == "report":
        prefix = "report"
    else:
        prefix = "test"
    return tuple(f"{prefix}_{index:03d}" for index in range(6))


def provided_handoff_artifacts(stage: str) -> tuple[str, ...]:
    if stage == "E2E":
        return ()
    index = STAGES.index(stage)
    return tuple(
        path
        for prior_stage in STAGES[:index]
        for path in STAGE_REQUIRED_ARTIFACTS[prior_stage]
    )


def current_required_artifacts(stage: str) -> tuple[str, ...]:
    if stage == "E2E":
        return tuple(
            path
            for item in STAGES
            for path in STAGE_REQUIRED_ARTIFACTS[item]
        )
    return STAGE_REQUIRED_ARTIFACTS[stage]


def agentic_budget_contract() -> dict[str, Any]:
    """Return the one bounded multi-turn budget shared by data and runtime."""

    return {
        "version": AGENTIC_BUDGET_VERSION,
        "max_assistant_turns": AGENTIC_MAX_ASSISTANT_TURNS,
        "max_user_turns": AGENTIC_MAX_USER_TURNS,
        "max_tool_calls": AGENTIC_MAX_TOOL_CALLS,
        "max_response_tokens": AGENTIC_MAX_RESPONSE_TOKENS,
        "max_tool_response_chars": AGENTIC_MAX_TOOL_RESPONSE_CHARS,
        "max_prompt_tokens": AGENTIC_MAX_PROMPT_TOKENS,
        "max_model_tokens": AGENTIC_MAX_MODEL_TOKENS,
    }


def task_contract_from_extra(extra_info: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(extra_info, Mapping):
        raise CoreTaskContractError("extra_info_invalid")
    state_id = _safe_id(extra_info.get("state_id"), "state_id")
    source_state_id = _safe_id(extra_info.get("source_state_id"), "source_state_id")
    track_value = extra_info.get("track")
    if not isinstance(track_value, str) or track_value.lower() not in _TASKS:
        raise CoreTaskContractError("track_unsupported")
    track = track_value.lower()
    stage_value = extra_info.get("target_stage")
    if not isinstance(stage_value, str) or stage_value.upper() not in ALLOWED_STAGES:
        raise CoreTaskContractError("target_stage_unsupported")
    stage = stage_value.upper()
    view_index = _integer(extra_info.get("view_index"), "view_index")
    if extra_info.get("task_definition_version") != TASK_DEFINITION_VERSION:
        raise CoreTaskContractError("task_definition_version_unsupported")
    definition = _TASKS[track]
    contract = {
        "schema_version": 3,
        "task_definition_version": TASK_DEFINITION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "state_id": state_id,
        "source_state_id": source_state_id,
        "view_index": view_index,
        "task_id": f"bat-core-{track}",
        "task_name": definition["task_name"],
        "track": track,
        "target_stage": stage,
        "objective": definition["objective"],
        "inventory": list(definition["inventory"]),
        "expected_ids": list(expected_ids(track)),
        "required_output": "outputs/submission.csv",
        "id_column": definition["id_column"],
        "required_columns": list(definition["required_columns"]),
        "output_kind": definition["output_kind"],
        "allowed_values": {
            key: list(values) for key, values in definition["allowed_values"].items()
        },
        "agentic_budget": agentic_budget_contract(),
        "provided_handoff_artifacts": list(provided_handoff_artifacts(stage)),
        "current_required_artifacts": list(current_required_artifacts(stage)),
    }
    if "output_shape" in definition:
        contract["output_shape"] = list(definition["output_shape"])
    contract["contract_sha256"] = stable_sha256(contract)
    return contract


def render_prompt(extra_info: Mapping[str, Any]) -> list[dict[str, str]]:
    contract = task_contract_from_extra(extra_info)
    stage = contract["target_stage"]
    provided = contract["provided_handoff_artifacts"]
    required = contract["current_required_artifacts"]
    handoff_text = (
        "No earlier-stage artifacts are provided; begin from the raw task data."
        if not provided
        else (
            "The sandbox begins at a sealed stage boundary with these earlier-stage "
            "artifacts already present: " + ", ".join(provided) + "."
        )
    )
    user = "\n".join(
        [
            f"Task instance: {contract['state_id']}",
            f"Track: {contract['track']}",
            f"Task: {contract['task_name']}",
            f"Objective: {contract['objective']}.",
            "",
            "Exact sandbox inventory:",
            *[f"- {path}" for path in contract["inventory"]],
            "- task/spec.json (authoritative machine-readable contract)",
            "- task/README.md (stage and receipt instructions)",
            "",
            f"Current BaT target: {stage} ({_STAGE_NAMES[stage]}).",
            handoff_text,
            (
                "Complete the entire S1-S5 workflow in this persistent workspace."
                if stage == "E2E"
                else f"Complete {stage} only; do not replace or claim later-stage work."
            ),
            "Required artifacts for this target:",
            *[f"- {path}" for path in required],
            "",
            f"Final submission path: {contract['required_output']}",
            "Exact final columns: " + ", ".join(contract["required_columns"]),
            f"Expected identifiers are listed in task/spec.json under {contract['id_column']}.",
            (
                "Runtime budget: at most "
                f"{AGENTIC_MAX_TOOL_CALLS} execute_code calls across "
                f"{AGENTIC_MAX_ASSISTANT_TURNS} assistant turns, with an "
                f"{AGENTIC_MAX_RESPONSE_TOKENS}-token cumulative rollout response budget "
                f"and {AGENTIC_MAX_TOOL_RESPONSE_CHARS}-character tool observations. "
                "Inspect selectively and reserve enough calls and tokens to execute and "
                "validate the required artifacts."
            ),
            "Use execute_code for every filesystem or computation action and inspect observed results.",
            (
                "Every execute_code call must contain exactly two arguments: language "
                "(python or bash) and code (literal executable source). Never use task, "
                "description, prompt, or subagent_type arguments. If work remains after an "
                "observation, issue another valid execute_code call and execute the code; do "
                "not merely paste a code block in prose."
            ),
            (
                "First-action requirement: your first assistant turn MUST be an execute_code "
                "tool call, not prose. Begin with a read-only inspection of task/spec.json and "
                "the relevant manifests, then use the observed evidence to perform the target."
            ),
            "Network access is disabled. The task is solvable with Python's standard library; "
            "do not assume scikit-learn or fabricate package/model availability.",
            "Do not fabricate artifacts, hand-edit predictions to satisfy checks, or report success "
            "without persisted evidence.",
        ]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an agentic medical-ML engineer working in a persistent isolated "
                "sandbox. Follow the exact task contract and ground every claim in tool-observed "
                "evidence. The only available function is execute_code, whose arguments are exactly "
                "language and code. Never emit task/subagent-style arguments. Do not provide a plan "
                "or final answer before your first execute_code call."
            ),
        },
        {"role": "user", "content": user},
    ]


def validate_prompt_binding(
    prompt: Any,
    extra_info: Mapping[str, Any],
) -> list[dict[str, str]]:
    if not isinstance(prompt, Sequence) or isinstance(prompt, (str, bytes)):
        raise CoreTaskContractError("prompt_invalid")
    observed: list[dict[str, str]] = []
    for message in prompt:
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise CoreTaskContractError("prompt_message_invalid")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user"} or not isinstance(content, str):
            raise CoreTaskContractError("prompt_message_invalid")
        observed.append({"role": role, "content": content})
    expected = render_prompt(extra_info)
    if observed != expected:
        raise CoreTaskContractError("prompt_task_contract_mismatch")
    return observed


__all__ = [
    "AGENTIC_BUDGET_VERSION",
    "AGENTIC_MAX_ASSISTANT_TURNS",
    "AGENTIC_MAX_MODEL_TOKENS",
    "AGENTIC_MAX_PROMPT_TOKENS",
    "AGENTIC_MAX_RESPONSE_TOKENS",
    "AGENTIC_MAX_TOOL_CALLS",
    "AGENTIC_MAX_TOOL_RESPONSE_CHARS",
    "AGENTIC_MAX_USER_TURNS",
    "ALLOWED_STAGES",
    "CoreTaskContractError",
    "PROMPT_VERSION",
    "STAGES",
    "STAGE_REQUIRED_ARTIFACTS",
    "TASK_DEFINITION_VERSION",
    "TRACKS",
    "agentic_budget_contract",
    "current_required_artifacts",
    "expected_ids",
    "provided_handoff_artifacts",
    "render_prompt",
    "stable_sha256",
    "task_contract_from_extra",
    "validate_prompt_binding",
]

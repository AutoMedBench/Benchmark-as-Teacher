"""Synthetic, resettable BaT task workspaces and VERL tool adapter."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from rl.grpo.core_task_contract import (
    AGENTIC_MAX_RESPONSE_TOKENS,
    AGENTIC_MAX_TOOL_CALLS,
    STAGES,
    stable_sha256,
    task_contract_from_extra,
    validate_prompt_binding,
)
from rl.grpo.core_workspace_tool import (
    CoreWorkspaceExecuteCodeTool,
    WorkspaceCapability,
)
from rl.grpo.tool_protocol_contract import runtime_protocol_binding
from rl.sandbox.container_runtime import PinnedContainerRuntime, validate_image_reference
from rl.sandbox.state_package_runtime import (
    SnapshotLimits,
    fingerprint_workspace,
    materialize_snapshot,
    seal_snapshot_manifest,
    validate_snapshot_manifest,
)


TOOL_NAME = "execute_code"
SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_LIMITS = SnapshotLimits(
    max_entries=512,
    max_files=384,
    max_directories=128,
    max_file_bytes=2 * 1024 * 1024,
    max_total_bytes=32 * 1024 * 1024,
    max_manifest_bytes=48 * 1024 * 1024,
)


class CoreAgenticSandboxError(RuntimeError):
    """A task row cannot be materialized or executed safely."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _pgm_pixels(pixels: Sequence[int], width: int, height: int) -> bytes:
    return f"P2\n{width} {height}\n255\n{' '.join(str(value) for value in pixels)}\n".encode("ascii")


def _base_pixels(seed: int, index: int, size: int = 8) -> list[int]:
    return [40 + ((seed + index * 17 + x * 7 + y * 11) % 80) for y in range(size) for x in range(size)]


def _lesion_pixels(seed: int, index: int) -> tuple[list[int], tuple[int, int, int, int]]:
    pixels = _base_pixels(seed, index)
    x_min = 1 + ((seed + index) % 3)
    y_min = 1 + (((seed // 7) + index * 2) % 3)
    x_max, y_max = x_min + 2, y_min + 2
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            pixels[y * 8 + x] = 240
    return pixels, (x_min, y_min, x_max, y_max)


def _nearest_upscale(pixels: Sequence[int], size: int = 4) -> list[int]:
    return [pixels[(y // 2) * size + (x // 2)] for y in range(size * 2) for x in range(size * 2)]


def _directory(path: str) -> dict[str, Any]:
    return {"path": path, "type": "directory", "mode": "0755"}


def _file(path: str, content: bytes, mode: int = 0o644) -> dict[str, Any]:
    import base64

    return {
        "path": path,
        "type": "file",
        "mode": f"0{mode:o}",
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "content_base64": base64.b64encode(content).decode("ascii"),
    }


def _task_data(contract: Mapping[str, Any]) -> list[tuple[str, bytes]]:
    track = str(contract["track"])
    seed = int(hashlib.sha256(str(contract["source_state_id"]).encode()).hexdigest()[:8], 16)
    expected = list(contract["expected_ids"])
    rows: list[tuple[str, bytes]] = []
    if track == "classification":
        train = []
        for i in range(18):
            feature_a, feature_b = i % 5, (i * 3) % 7
            train.append([f"train_{i:03d}", feature_a, feature_b, "suspicious" if feature_a + feature_b >= 5 else "benign"])
        test = [[identifier, (i + 1) % 5, (i * 2 + 1) % 7] for i, identifier in enumerate(expected)]
        rows.extend(
            [
                ("data/train.csv", _csv_bytes(["sample_id", "feature_a", "feature_b", "label"], train)),
                ("data/test.csv", _csv_bytes(["sample_id", "feature_a", "feature_b"], test)),
            ]
        )
    elif track == "report":
        train = [[f"train_{i:03d}", f"Finding   {i}.  No acute process.", f"Finding {i}. No acute process."] for i in range(12)]
        test = [[identifier, f"Study   {i}.  Mild opacity."] for i, identifier in enumerate(expected)]
        rows.extend(
            [
                ("data/train_reports.csv", _csv_bytes(["report_id", "raw_text", "cleaned_text"], train)),
                ("data/test_reports.csv", _csv_bytes(["report_id", "raw_text"], test)),
            ]
        )
    elif track == "vqa":
        train = [[f"question_train_{i:03d}", f"data/images/train_{i:03d}.pgm", "Is the center bright?", "yes" if i % 2 else "no"] for i in range(10)]
        test = [[identifier, f"data/images/{identifier}.pgm", "Is the center bright?"] for identifier in expected]
        rows.extend(
            [
                ("data/train_questions.csv", _csv_bytes(["question_id", "image_path", "question", "answer"], train)),
                ("data/test_questions.csv", _csv_bytes(["question_id", "image_path", "question"], test)),
            ]
        )
        for i in range(10):
            pixels = _base_pixels(seed, i)
            pixels[4 * 8 + 4] = 255 if i % 2 else 0
            rows.append((f"data/images/train_{i:03d}.pgm", _pgm_pixels(pixels, 8, 8)))
        for i, identifier in enumerate(expected):
            pixels = _base_pixels(seed + 3, i)
            pixels[4 * 8 + 4] = 255 if (seed + i) % 2 else 0
            rows.append((f"data/images/{identifier}.pgm", _pgm_pixels(pixels, 8, 8)))
    else:
        training_ids = [f"train_{i:03d}" for i in range(8)]
        if track == "detection":
            annotations = []
            for i, item in enumerate(training_ids):
                _pixels, box = _lesion_pixels(seed, i)
                annotations.append([item, f"data/images/{item}.pgm", *box])
            rows.extend(
                [
                    ("data/train_annotations.csv", _csv_bytes(["sample_id", "image_path", "x_min", "y_min", "x_max", "y_max"], annotations)),
                    ("data/test_manifest.csv", _csv_bytes(["sample_id", "image_path"], [[item, f"data/images/{item}.pgm"] for item in expected])),
                ]
            )
            folders = [("images", training_ids + expected)]
        elif track == "segmentation":
            rows.extend(
                [
                    ("data/train_manifest.csv", _csv_bytes(["sample_id", "image_path", "mask_path"], [[item, f"data/images/{item}.pgm", f"data/masks/{item}.pgm"] for item in training_ids])),
                    ("data/test_manifest.csv", _csv_bytes(["sample_id", "image_path"], [[item, f"data/images/{item}.pgm"] for item in expected])),
                ]
            )
            folders = [("images", training_ids + expected), ("masks", training_ids)]
        elif track == "enhancement":
            rows.extend(
                [
                    ("data/train_manifest.csv", _csv_bytes(["sample_id", "noisy_path", "clean_path"], [[item, f"data/noisy/{item}.pgm", f"data/clean/{item}.pgm"] for item in training_ids])),
                    ("data/test_manifest.csv", _csv_bytes(["sample_id", "noisy_path"], [[item, f"data/noisy/{item}.pgm"] for item in expected])),
                ]
            )
            folders = [("noisy", training_ids + expected), ("clean", training_ids)]
        elif track == "synthesis":
            rows.extend(
                [
                    ("data/train_manifest.csv", _csv_bytes(["sample_id", "low_path", "high_path"], [[item, f"data/low_res/{item}.pgm", f"data/high_res/{item}.pgm"] for item in training_ids])),
                    ("data/test_manifest.csv", _csv_bytes(["sample_id", "low_path"], [[item, f"data/low_res/{item}.pgm"] for item in expected])),
                ]
            )
            folders = [("low_res", training_ids + expected), ("high_res", training_ids)]
        else:
            raise CoreAgenticSandboxError("track_unsupported")
        for folder, identifiers in folders:
            for i, identifier in enumerate(identifiers):
                if track in {"detection", "segmentation"}:
                    pixels, box = _lesion_pixels(seed, i)
                    if folder == "masks":
                        mask = [0] * 64
                        for y in range(box[1], box[3] + 1):
                            for x in range(box[0], box[2] + 1):
                                mask[y * 8 + x] = 255
                        pixels = mask
                    content = _pgm_pixels(pixels, 8, 8)
                elif track == "enhancement":
                    clean = _base_pixels(seed, i)
                    if folder == "noisy":
                        pixels = [max(0, min(255, value + (18 if j % 2 else -18))) for j, value in enumerate(clean)]
                    else:
                        pixels = clean
                    content = _pgm_pixels(pixels, 8, 8)
                elif track == "synthesis":
                    low = _base_pixels(seed, i, 4)
                    pixels = low if folder == "low_res" else _nearest_upscale(low)
                    size = 4 if folder == "low_res" else 8
                    content = _pgm_pixels(pixels, size, size)
                else:
                    raise CoreAgenticSandboxError("track_image_generation_invalid")
                rows.append((f"data/{folder}/{identifier}.pgm", content))
    return rows


def _default_value(column: str, identifier: str, track: str) -> Any:
    if column in {"sample_id", "question_id", "report_id"}:
        return identifier
    if column == "predicted_label":
        return "benign"
    if column == "confidence":
        return "0.5"
    if column == "answer":
        return "no"
    if column == "cleaned_text":
        return "Synthetic normalized report."
    if column in {"output_path", "mask_path"}:
        return f"outputs/artifacts/{identifier}.pgm"
    if column in {"x_min", "y_min"}:
        return "2"
    if column in {"x_max", "y_max"}:
        return "5"
    return "0"


def _prediction_csv(contract: Mapping[str, Any], identifiers: Sequence[str]) -> bytes:
    columns = list(contract["required_columns"])
    return _csv_bytes(
        columns,
        [[_default_value(column, identifier, str(contract["track"])) for column in columns] for identifier in identifiers],
    )


def _receipt(stage: str, path: str, payload: bytes, **fields: Any) -> bytes:
    return _json_bytes(
        {
            "schema_version": 1,
            "stage": stage,
            "artifact": path,
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            **fields,
        }
    )


def _handoff_files(contract: Mapping[str, Any]) -> list[tuple[str, bytes]]:
    target = str(contract["target_stage"])
    if target == "E2E":
        return []
    completed = STAGES[: STAGES.index(target)]
    result: list[tuple[str, bytes]] = []
    if "S1" in completed:
        result.append(("plan/plan.md", b"# Plan\n\nPreprocess inputs, run inference, then postprocess and validate outputs.\n"))
    if "S2" in completed:
        result.extend(
            [
                ("setup/environment.json", _json_bytes({"status": "ready", "runtime": "python-stdlib"})),
                ("setup/compatibility.json", _json_bytes({"status": "compatible", "network_required": False})),
                ("model/checkpoint.json", _json_bytes({"method": "deterministic-baseline", "external_weights": False})),
                ("setup/model_load.json", _json_bytes({"loaded": True, "method": "stdlib"})),
                ("pipeline/infer.py", b"# Stage handoff: replace with task-specific inference when needed.\nfrom pathlib import Path\nassert Path('task/spec.json').is_file()\n"),
            ]
        )
    if "S3" in completed:
        pilot = _prediction_csv(contract, list(contract["expected_ids"])[:1])
        result.extend(
            [
                ("validation/pilot.csv", pilot),
                ("validation/s3_validation.json", _receipt("S3", "validation/pilot.csv", pilot, schema_valid=True, values_valid=True)),
            ]
        )
    if "S4" in completed:
        predictions = _prediction_csv(contract, list(contract["expected_ids"]))
        result.extend(
            [
                ("outputs/predictions.csv", predictions),
                ("validation/s4_completion.json", _receipt("S4", "outputs/predictions.csv", predictions, expected_count=len(contract["expected_ids"]), attempted_count=len(contract["expected_ids"]), valid_count=len(contract["expected_ids"]))),
            ]
        )
    return result


def sandbox_snapshot_for_row(
    prompt: Any,
    extra_info: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a sealed snapshot from held-out-safe abstract row metadata."""

    validate_prompt_binding(prompt, extra_info)
    contract = task_contract_from_extra(extra_info)
    prompt_sha256 = stable_sha256(prompt)
    public_spec = dict(contract)
    public_spec["prompt_sha256"] = prompt_sha256
    spec_bytes = _json_bytes(public_spec)
    handoffs = _handoff_files(contract)
    contract.update(
        {
            "prompt_sha256": prompt_sha256,
            "task_spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
            "provided_handoff_sha256": {
                path: hashlib.sha256(content).hexdigest() for path, content in handoffs
            },
        }
    )
    files = [
        ("task/spec.json", spec_bytes),
        (
            "task/README.md",
            (
                "Use execute_code to inspect the task, create only the requested stage "
                "artifacts, validate them, and retain machine-readable receipts.\n"
            ).encode("utf-8"),
        ),
        *_task_data(contract),
        *handoffs,
    ]
    directory_names: set[str] = set()
    for path, _ in files:
        parts = Path(path).parts[:-1]
        for index in range(1, len(parts) + 1):
            directory_names.add("/".join(parts[:index]))
    entries = [_directory(path) for path in sorted(directory_names, key=lambda p: (p.count("/"), p))]
    entries.extend(_file(path, content, 0o755 if path.endswith(".py") else 0o644) for path, content in sorted(files))
    return seal_snapshot_manifest(entries, limits=_LIMITS), contract


def _read_regular(workspace: Path, relative: str, limit: int = 2 * 1024 * 1024) -> bytes | None:
    path = workspace / relative
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError:
        return None
    if workspace not in resolved.parents or resolved != path or not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        return None
    return path.read_bytes()


def _json_file(workspace: Path, relative: str) -> dict[str, Any] | None:
    raw = _read_regular(workspace, relative)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _csv_evidence(workspace: Path, relative: str, contract: Mapping[str, Any], expected: Sequence[str]) -> dict[str, Any]:
    raw = _read_regular(workspace, relative)
    base = {"path": relative, "exists": raw is not None, "sha256": hashlib.sha256(raw).hexdigest() if raw else None}
    if raw is None:
        return {**base, "schema_exact": False, "row_count": 0, "complete": False}
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error):
        return {**base, "schema_exact": False, "row_count": 0, "complete": False}
    columns = list(contract["required_columns"])
    id_column = str(contract["id_column"])
    ids = [row.get(id_column) for row in rows]
    schema_exact = reader.fieldnames == columns
    allowed = dict(contract.get("allowed_values") or {})
    values_valid = True
    for row in rows:
        for field, choices in allowed.items():
            values_valid = values_valid and row.get(field) in choices
        if "confidence" in columns:
            try:
                confidence = float(row.get("confidence", ""))
            except ValueError:
                confidence = -1.0
            values_valid = values_valid and 0.0 <= confidence <= 1.0
        if contract.get("output_kind") == "detection_boxes":
            try:
                box = tuple(int(row[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
            except (KeyError, ValueError):
                box = (-1, -1, -1, -1)
            values_valid = values_valid and 0 <= box[0] <= box[2] < 8 and 0 <= box[1] <= box[3] < 8
        if "cleaned_text" in columns:
            text = row.get("cleaned_text", "")
            values_valid = values_valid and bool(text.strip()) and "<" not in text and ">" not in text
        for field in ("output_path", "mask_path"):
            if field in columns:
                path = row.get(field, "")
                values_valid = values_valid and path.startswith("outputs/artifacts/") and path.endswith(".pgm") and ".." not in Path(path).parts
    complete = schema_exact and values_valid and len(rows) == len(expected) and len(ids) == len(set(ids)) and set(ids) == set(expected)
    return {**base, "schema_exact": schema_exact, "values_valid": values_valid, "row_count": len(rows), "complete": complete}


def _csv_rows(workspace: Path, relative: str) -> list[dict[str, str]]:
    raw = _read_regular(workspace, relative)
    if raw is None:
        return []
    try:
        return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error):
        return []


def _pgm_info(raw: bytes | None) -> tuple[int, int, list[int]] | None:
    if raw is None:
        return None
    try:
        tokens = raw.decode("ascii").split()
        if len(tokens) < 4 or tokens[0] != "P2" or int(tokens[3]) != 255:
            return None
        width, height = int(tokens[1]), int(tokens[2])
        pixels = [int(value) for value in tokens[4:]]
    except (UnicodeDecodeError, ValueError):
        return None
    if width <= 0 or height <= 0 or len(pixels) != width * height or any(not 0 <= value <= 255 for value in pixels):
        return None
    return width, height, pixels


def _output_pgm(workspace: Path, relative: Any) -> tuple[int, int, list[int]] | None:
    if not isinstance(relative, str) or not relative.startswith("outputs/artifacts/"):
        return None
    path = workspace / relative
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if workspace not in resolved.parents or resolved != path or path.is_symlink():
        return None
    return _pgm_info(_read_regular(workspace, relative))


def _task_score(workspace: Path, contract: Mapping[str, Any], relative: str) -> float:
    rows = _csv_rows(workspace, relative)
    id_column = str(contract["id_column"])
    by_id = {row.get(id_column): row for row in rows}
    expected_ids = list(contract["expected_ids"])
    track = str(contract["track"])
    correct = 0
    test_rows: dict[str, dict[str, str]] = {}
    if track == "classification":
        test_rows = {row["sample_id"]: row for row in _csv_rows(workspace, "data/test.csv")}
    elif track == "report":
        test_rows = {row["report_id"]: row for row in _csv_rows(workspace, "data/test_reports.csv")}
    for index, identifier in enumerate(expected_ids):
        row = by_id.get(identifier)
        if row is None:
            continue
        valid = False
        if track == "classification":
            source = test_rows.get(identifier, {})
            try:
                label = "suspicious" if int(source["feature_a"]) + int(source["feature_b"]) >= 5 else "benign"
            except (KeyError, ValueError):
                label = ""
            valid = row.get("predicted_label") == label
        elif track == "report":
            source = test_rows.get(identifier, {})
            expected_text = " ".join(source.get("raw_text", "").split())
            valid = row.get("cleaned_text") == expected_text
        elif track == "vqa":
            image = _pgm_info(_read_regular(workspace, f"data/images/{identifier}.pgm"))
            expected_answer = "yes" if image and image[2][4 * 8 + 4] >= 200 else "no"
            valid = row.get("answer") == expected_answer
        elif track == "detection":
            image = _pgm_info(_read_regular(workspace, f"data/images/{identifier}.pgm"))
            bright = [(i % 8, i // 8) for i, value in enumerate(image[2] if image else []) if value >= 200]
            expected_box = (
                min(x for x, _ in bright), min(y for _, y in bright), max(x for x, _ in bright), max(y for _, y in bright)
            ) if bright else ()
            try:
                observed_box = tuple(int(row[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
            except (KeyError, ValueError):
                observed_box = ()
            valid = observed_box == expected_box
        elif track == "segmentation":
            image = _pgm_info(_read_regular(workspace, f"data/images/{identifier}.pgm"))
            output = _output_pgm(workspace, row.get("mask_path"))
            expected_mask = [255 if value >= 200 else 0 for value in image[2]] if image else []
            valid = bool(output and output[:2] == (8, 8) and output[2] == expected_mask)
        elif track == "enhancement":
            noisy = _pgm_info(_read_regular(workspace, f"data/noisy/{identifier}.pgm"))
            output = _output_pgm(workspace, row.get("output_path"))
            expected_clean = [max(0, min(255, value - (18 if j % 2 else -18))) for j, value in enumerate(noisy[2])] if noisy else []
            valid = bool(output and output[:2] == (8, 8) and output[2] == expected_clean)
        elif track == "synthesis":
            low = _pgm_info(_read_regular(workspace, f"data/low_res/{identifier}.pgm"))
            output = _output_pgm(workspace, row.get("output_path"))
            expected_high = _nearest_upscale(low[2]) if low and low[:2] == (4, 4) else []
            valid = bool(output and output[:2] == (8, 8) and output[2] == expected_high)
        correct += int(valid)
    return correct / len(expected_ids) if expected_ids else 0.0


def _valid_receipt(workspace: Path, relative: str, stage: str, artifact: Mapping[str, Any]) -> bool:
    value = _json_file(workspace, relative)
    return bool(
        value
        and value.get("schema_version") == 1
        and value.get("stage") == stage
        and value.get("artifact") == artifact.get("path")
        and value.get("artifact_sha256") == artifact.get("sha256")
    )


def artifact_evidence_for_workspace(workspace: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    spec = _read_regular(workspace, "task/spec.json")
    handoff_expected = dict(contract.get("provided_handoff_sha256") or {})
    changed = [path for path, digest in handoff_expected.items() if hashlib.sha256(_read_regular(workspace, path) or b"").hexdigest() != digest]
    plan = (_read_regular(workspace, "plan/plan.md") or b"").decode("utf-8", errors="replace").lower()
    s1_checks = {
        "plan_exists": bool(plan),
        "pipeline_sections_present": all(
            word in plan for word in ("preprocess", "inference", "postprocess")
        ),
    }
    s1 = all(s1_checks.values())
    s2_checks = {
        "environment_ready": (_json_file(workspace, "setup/environment.json") or {}).get("status") == "ready",
        "compatibility_checked": (_json_file(workspace, "setup/compatibility.json") or {}).get("status") == "compatible",
        "model_loaded": (_json_file(workspace, "setup/model_load.json") or {}).get("loaded") is True,
        "checkpoint_declared": bool((_json_file(workspace, "model/checkpoint.json") or {}).get("method")),
        "pipeline_exists": bool(_read_regular(workspace, "pipeline/infer.py")),
    }
    s2 = all(s2_checks.values())
    expected = list(contract["expected_ids"])
    pilot = _csv_evidence(workspace, "validation/pilot.csv", contract, expected[:1])
    predictions = _csv_evidence(workspace, "outputs/predictions.csv", contract, expected)
    submission = _csv_evidence(workspace, str(contract["required_output"]), contract, expected)
    task_score = _task_score(
        workspace,
        contract,
        str(contract["required_output"]),
    ) if submission["complete"] else 0.0
    s3_checks = {
        "small_case_executed": bool(pilot["complete"]),
        "validation_receipt_valid": _valid_receipt(
            workspace, "validation/s3_validation.json", "S3", pilot
        ),
    }
    s4_checks = {
        "full_inference_complete": bool(predictions["complete"]),
        "completion_receipt_valid": _valid_receipt(
            workspace, "validation/s4_completion.json", "S4", predictions
        ),
    }
    s5_checks = {
        "submission_complete": bool(submission["complete"]),
        "submission_receipt_valid": _valid_receipt(
            workspace, "validation/s5_submission.json", "S5", submission
        ),
    }
    s3, s4, s5 = all(s3_checks.values()), all(s4_checks.values()), all(s5_checks.values())
    stages = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5}
    target = str(contract["target_stage"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "contract_verified": spec is not None and hashlib.sha256(spec).hexdigest() == contract["task_spec_sha256"],
        "stage_entry_intact": not changed,
        "changed_handoff_artifacts": changed,
        "target_stage": target,
        "stage_complete": stages,
        "current_stage_complete": all(stages.values()) if target == "E2E" else stages[target],
        "task_score": task_score,
        "checks": {
            "S1": s1_checks,
            "S2": s2_checks,
            "S3": s3_checks,
            "S4": s4_checks,
            "S5": s5_checks,
        },
        "artifacts": {"pilot": pilot, "predictions": predictions, "submission": submission},
    }
    payload["evidence_sha256"] = stable_sha256(payload)
    return payload


@dataclass(frozen=True)
class CoreSandboxLease:
    capability: WorkspaceCapability
    workspace: Path
    rollout_id: str
    contract: dict[str, Any]


class CoreSandboxExecuteCodeTool:
    """VERL-compatible tool with one fresh persistent workspace per rollout."""

    def __init__(self, config: Mapping[str, Any], tool_schema: Any) -> None:
        plain = dict(config) if isinstance(config, Mapping) else {}
        allowed = {"type", "workspace_root", "receipts_root", "timeout_seconds", "image", "runtime_binary"}
        if set(plain) - allowed or plain.get("type") != "native":
            raise CoreAgenticSandboxError("tool_config_invalid")
        self.workspace_root = self._root(plain.get("workspace_root"), "workspace_root")
        self.receipts_root = self._root(plain.get("receipts_root"), "receipts_root")
        if self.workspace_root.parent != self.receipts_root.parent or self.workspace_root == self.receipts_root:
            raise CoreAgenticSandboxError("tool_roots_must_be_siblings")
        timeout = plain.get("timeout_seconds", 120)
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 600:
            raise CoreAgenticSandboxError("tool_timeout_invalid")
        image = validate_image_reference(plain.get("image") or os.environ.get("BAT_SANDBOX_IMAGE"))
        runtime = PinnedContainerRuntime(
            allowed_root=self.workspace_root,
            image=image,
            binary=Path(str(plain.get("runtime_binary") or "docker")),
            timeout_seconds=timeout,
        )
        self.name = TOOL_NAME
        self.tool_schema = tool_schema
        self._inner = CoreWorkspaceExecuteCodeTool(runtime=runtime, tool_schema=tool_schema, max_steps=AGENTIC_MAX_TOOL_CALLS)
        self._write_lock = threading.RLock()

    @staticmethod
    def _root(value: Any, label: str) -> Path:
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise CoreAgenticSandboxError(f"{label}_invalid")
        path = Path(value)
        try:
            resolved = path.resolve(strict=True)
            info = path.lstat()
        except OSError as exc:
            raise CoreAgenticSandboxError(f"{label}_invalid") from exc
        if resolved != path or path.is_symlink() or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or path == Path("/"):
            raise CoreAgenticSandboxError(f"{label}_invalid")
        return path

    def prepare_trajectory(self, *, prompt: Any, extra_info: Any) -> CoreSandboxLease:
        manifest, contract = sandbox_snapshot_for_row(prompt, extra_info)
        snapshot = validate_snapshot_manifest(manifest, limits=_LIMITS)
        workspace = materialize_snapshot(snapshot, self.workspace_root, prefix="bat_core_")
        observed = fingerprint_workspace(workspace, limits=_LIMITS)
        if observed != snapshot.content_sha256:
            raise CoreAgenticSandboxError("sandbox_reset_fingerprint_mismatch")
        rollout_id = uuid4().hex
        capability = self._inner.bind_workspace(
            workspace_path=workspace,
            session_id=f"core-{rollout_id}",
            state_id=str(contract["state_id"]),
            reset_fingerprint=observed,
        )
        return CoreSandboxLease(capability, workspace, rollout_id, contract)

    def create_kwargs(self, lease: CoreSandboxLease) -> dict[str, str]:
        return self._inner.create_kwargs(lease.capability)

    async def create(self, instance_id: str | None = None, **kwargs: Any) -> tuple[str, Any]:
        return await self._inner.create(instance_id, **kwargs)

    async def execute(self, instance_id: str, parameters: Mapping[str, Any], **kwargs: Any) -> tuple[Any, float, dict[str, Any]]:
        return await self._inner.execute(instance_id, parameters, **kwargs)

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        await self._inner.release(instance_id, **kwargs)

    async def abort_trajectory(self, lease: CoreSandboxLease) -> None:
        await self._inner.abort_workspace(lease.capability)
        self._cleanup(lease.workspace)

    async def finalize_trajectory(
        self,
        lease: CoreSandboxLease,
        *,
        transcript: str,
        response_tokens: int,
        num_turns: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if response_tokens > AGENTIC_MAX_RESPONSE_TOKENS:
            raise CoreAgenticSandboxError("response_budget_exceeded")
        calls = await self._inner.finalize_workspace(lease.capability, allow_no_calls=True)
        artifacts = artifact_evidence_for_workspace(lease.workspace, lease.contract)
        protocol = runtime_protocol_binding()
        evidence = {
            "schema_version": 1,
            "contract": {
                key: lease.contract[key]
                for key in ("task_definition_version", "contract_sha256", "state_id", "track", "target_stage")
            },
            "tool": calls,
            "artifacts": artifacts,
            "runtime_protocol": protocol,
            "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        }
        evidence["evidence_sha256"] = stable_sha256(evidence)
        summary = {
            "schema_version": 1,
            "contract_sha256": lease.contract["contract_sha256"],
            "target_stage": lease.contract["target_stage"],
            "track": lease.contract["track"],
            "tool_calls": calls["tool_calls"],
            "successful_tool_calls": calls["successful_tool_calls"],
            "workspace_changed": calls["workspace_changed"],
            "num_turns": int(num_turns),
            "response_tokens": int(response_tokens),
            "current_stage_complete": artifacts["current_stage_complete"],
            "stage_complete": artifacts["stage_complete"],
            "evidence_sha256": evidence["evidence_sha256"],
        }
        summary["summary_sha256"] = stable_sha256(summary)
        self._write_receipt(lease.rollout_id, evidence, summary)
        self._cleanup(lease.workspace)
        return evidence, summary

    def _write_receipt(self, rollout_id: str, evidence: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
        path = self.receipts_root / f"{rollout_id}.json"
        content = _json_bytes({"evidence": evidence, "summary": summary})
        with self._write_lock:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                pending = memoryview(content)
                while pending:
                    written = os.write(descriptor, pending)
                    if written <= 0:
                        raise CoreAgenticSandboxError("receipt_short_write")
                    pending = pending[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def _cleanup(self, workspace: Path) -> None:
        try:
            info = workspace.lstat()
            resolved = workspace.resolve(strict=True)
        except OSError as exc:
            raise CoreAgenticSandboxError("workspace_cleanup_refused") from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or resolved != workspace
            or resolved.parent != self.workspace_root
            or resolved == self.workspace_root
        ):
            raise CoreAgenticSandboxError("workspace_cleanup_refused")
        shutil.rmtree(resolved)


def safe_agentic_error_code(exc: BaseException) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", type(exc).__name__.lower()).strip("_")
    return name[:64] or "unknown"


__all__ = [
    "CoreAgenticSandboxError",
    "CoreSandboxExecuteCodeTool",
    "CoreSandboxLease",
    "artifact_evidence_for_workspace",
    "safe_agentic_error_code",
    "sandbox_snapshot_for_row",
]

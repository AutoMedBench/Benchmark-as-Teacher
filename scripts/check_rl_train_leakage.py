#!/usr/bin/env python3
"""Fail-closed scan for held-out evidence in SFT or RL JSONL artifacts."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


_ABSOLUTE_PATH = re.compile(r"(?:^|[\s\"'])/(?:home|root|localhome|mnt|workspace|tmp)/")
_SENSITIVE_FIELD_PARTS = (
    "heldout_answer",
    "expected_answer",
    "evaluation_report",
    "evaluator_trace",
    "rollout_transcript",
    "teacher_completion",
    "workspace_path",
    "correction_text",
)


@dataclass(frozen=True)
class Violation:
    line: int
    field: str
    code: str


def _markers(path: Path | None) -> set[str]:
    if path is None:
        return set()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, str) or len(item.strip()) < 4 for item in value):
        raise ValueError("heldout_markers_invalid")
    return {item.casefold() for item in value}


def _walk(value: Any, field: str = "$") -> Iterable[tuple[str, Any]]:
    yield field, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, f"{field}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{field}[{index}]")


def scan_file(path: Path, *, heldout_markers: Path | None = None) -> list[Violation]:
    markers = _markers(heldout_markers)
    violations: list[Violation] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                violations.append(Violation(line_number, "$", "invalid_json"))
                continue
            for field, value in _walk(row):
                lowered_field = field.casefold()
                declared_safe_flag = field.endswith(
                    (
                        "assistant_target_included",
                        "teacher_trajectory_included",
                        "rollout_transcript_reuse_allowed",
                    )
                ) and value is False
                if not declared_safe_flag and any(part in lowered_field for part in _SENSITIVE_FIELD_PARTS):
                    violations.append(Violation(line_number, field, "forbidden_evaluation_field"))
                if isinstance(value, str):
                    lowered = value.casefold()
                    if _ABSOLUTE_PATH.search(value):
                        violations.append(Violation(line_number, field, "machine_specific_path"))
                    if any(marker in lowered for marker in markers):
                        violations.append(Violation(line_number, field, "heldout_marker"))
                if field.endswith("evaluation_content_used") and value is not False:
                    violations.append(Violation(line_number, field, "evaluation_content_flag"))
                if field.endswith(("assistant_target_included", "teacher_trajectory_included", "rollout_transcript_reuse_allowed")) and value is not False:
                    violations.append(Violation(line_number, field, "training_target_leakage_flag"))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--heldout-markers", type=Path)
    args = parser.parse_args()
    violations = scan_file(args.path.resolve(strict=True), heldout_markers=args.heldout_markers)
    print(json.dumps({"status": "failed" if violations else "passed", "violations": [asdict(item) for item in violations]}, indent=2, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())

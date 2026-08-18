#!/usr/bin/env python3
"""Validate external evaluator output and emit BaT-safe aggregate scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from rl.reward.automedbench import STAGES, compute_agentic_score, finite_stage_scores, strict_overall, unit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_summary(raw_path: Path, *, checkpoint_sha256: str) -> dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1 or raw.get("status") != "complete":
        raise ValueError("evaluation_payload_invalid")
    cells = raw.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("evaluation_cells_missing")
    safe_cells: list[dict[str, Any]] = []
    keys: set[tuple[str, str, int]] = set()
    by_track: dict[str, list[float]] = defaultdict(list)
    by_stage: dict[str, list[float]] = defaultdict(list)
    tasks: list[float] = []
    agentic_values: list[float] = []
    overall_values: list[float] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping) or cell.get("valid") is not True:
            raise ValueError(f"evaluation_cell_{index}_invalid")
        track, task_id, repeat = cell.get("track"), cell.get("task_id"), cell.get("repeat")
        if not isinstance(track, str) or not track or not isinstance(task_id, str) or not task_id or isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
            raise ValueError(f"evaluation_cell_{index}_identity_invalid")
        key = (track, task_id, repeat)
        if key in keys:
            raise ValueError("evaluation_cell_duplicate")
        keys.add(key)
        task_score = unit(cell.get("task_score"), "task_score")
        stages = finite_stage_scores(cell.get("stages"))
        agentic = compute_agentic_score(stages)
        declared = cell.get("agentic_score")
        if declared is not None and not math.isclose(unit(declared, "agentic_score"), agentic, abs_tol=1e-8):
            raise ValueError(f"evaluation_cell_{index}_agentic_mismatch")
        overall = strict_overall(task_score, agentic)
        safe = {"track": track, "task_id": task_id, "repeat": repeat, "task_score": task_score, "agentic_score": agentic, "overall_score": overall, "stages": stages}
        safe_cells.append(safe)
        tasks.append(task_score)
        agentic_values.append(agentic)
        overall_values.append(overall)
        by_track[track].append(overall)
        for stage in STAGES:
            by_stage[stage].append(stages[stage])
    return {
        "schema_version": 1,
        "artifact_kind": "bat_eval_summary",
        "status": "valid",
        "checkpoint_sha256": checkpoint_sha256,
        "raw_evaluation_sha256": _sha256(raw_path),
        "cell_count": len(safe_cells),
        "cells": safe_cells,
        "means": {
            "task": statistics.fmean(tasks),
            "agentic": statistics.fmean(agentic_values),
            "overall": statistics.fmean(overall_values),
        },
        "stage_scores": {stage: statistics.fmean(by_stage[stage]) for stage in STAGES},
        "track_scores": {track: statistics.fmean(values) for track, values in sorted(by_track.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.raw.resolve(strict=True), checkpoint_sha256=args.checkpoint_sha256)
    content = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if args.output.exists() and args.output.read_bytes() != content:
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

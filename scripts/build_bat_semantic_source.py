#!/usr/bin/env python3
"""Create the held-out-safe 126-row abstract BaT source grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rl.grpo.core_task_contract import TRACKS


FOCUS_VALUES = ("E2E", "S1", "S2", "S3", "S4", "S5")
OUTCOMES = ("success", "recovery", "failure")


def _stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable(value).encode("utf-8")).hexdigest()


def build_source_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for focus in FOCUS_VALUES:
        for track in TRACKS:
            for outcome in OUTCOMES:
                identity = {"focus_stage": focus, "track": track, "outcome_bucket": outcome}
                source_state_id = "bat-safe:" + _digest(identity)[:24]
                row = {
                    "schema_version": 1,
                    "source_state_id": source_state_id,
                    **identity,
                    "abstract_condition": {
                        "success": "The preceding workflow evidence is internally consistent.",
                        "recovery": "A safe synthetic check exposed a correctable inconsistency.",
                        "failure": "The current workflow has no verified successful execution yet.",
                    }[outcome],
                    "training_contract": {
                        "prompt_only": True,
                        "assistant_target_included": False,
                        "teacher_trajectory_included": False,
                        "rollout_transcript_reuse_allowed": False,
                        "model_specific_serialization": False,
                    },
                    "provenance": {
                        "source": "heldout_safe_synthetic",
                        "evaluation_content_used": False,
                        "transformation": "cartesian_grid_v1",
                    },
                }
                row["row_sha256"] = _digest(row)
                rows.append(row)
    if len(rows) != 126 or len({row["source_state_id"] for row in rows}) != 126:
        raise RuntimeError("source_grid_invariant_failed")
    return rows


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise FileExistsError(f"refusing to overwrite: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_source(output_dir: Path) -> dict[str, Any]:
    rows = build_source_rows()
    content = b"".join((_stable(row) + "\n").encode("utf-8") for row in rows)
    source_path = output_dir / "source.jsonl"
    manifest_path = output_dir / "manifest.json"
    _write_immutable(source_path, content)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "bat_safe_source_grid",
        "rows": len(rows),
        "focus_values": list(FOCUS_VALUES),
        "tracks": list(TRACKS),
        "outcomes": list(OUTCOMES),
        "source_file": source_path.name,
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "heldout_safe": True,
        "evaluation_content_used": False,
    }
    _write_immutable(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_source(args.output_dir.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

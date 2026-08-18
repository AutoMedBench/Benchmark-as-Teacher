#!/usr/bin/env python3
"""Build one deterministic fixed three-pool BaT GRPO round."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rl.grpo.core_reward import e2e_reward_contract, stage_reward_contract
from rl.grpo.core_task_contract import STAGES, TASK_DEFINITION_VERSION, TRACKS, render_prompt
from scripts.build_bat_semantic_source import FOCUS_VALUES, OUTCOMES


TARGET_ROWS_PER_BATCH = 4
MIX_ROWS_PER_BATCH = 2
E2E_ROWS_PER_BATCH = 2
TRAIN_BATCH_SIZE = 8
DEFAULT_BATCHES = 378
SOURCE_POOLS = ("s_target", "s_mix", "e2e")


def _stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"source_line_{line_number}_invalid_json") from exc
            if not isinstance(value, dict):
                raise ValueError(f"source_line_{line_number}_not_object")
            rows.append(value)
    validate_source(rows)
    return rows


def validate_source(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = {(focus, track, outcome) for focus in FOCUS_VALUES for track in TRACKS for outcome in OUTCOMES}
    observed: set[tuple[str, str, str]] = set()
    ids: set[str] = set()
    for row in rows:
        copy = dict(row)
        declared = copy.pop("row_sha256", None)
        if declared != _digest(copy):
            raise ValueError("source_row_digest_invalid")
        cell = (str(row.get("focus_stage")), str(row.get("track")), str(row.get("outcome_bucket")))
        observed.add(cell)
        source_id = row.get("source_state_id")
        if not isinstance(source_id, str) or source_id in ids:
            raise ValueError("source_state_id_invalid")
        ids.add(source_id)
        contract = row.get("training_contract")
        if not isinstance(contract, Mapping) or contract.get("prompt_only") is not True:
            raise ValueError("source_training_contract_invalid")
        if any(contract.get(key) is not False for key in ("assistant_target_included", "teacher_trajectory_included", "rollout_transcript_reuse_allowed", "model_specific_serialization")):
            raise ValueError("source_training_contract_invalid")
        provenance = row.get("provenance")
        if not isinstance(provenance, Mapping) or provenance.get("evaluation_content_used") is not False:
            raise ValueError("source_provenance_invalid")
    if len(rows) != 126 or observed != expected or len(ids) != 126:
        raise ValueError("source_grid_incomplete")


def _cycle(rows: Sequence[Mapping[str, Any]], count: int, *, seed: str) -> list[tuple[Mapping[str, Any], int]]:
    if not rows or count < 0:
        raise ValueError("selection_invalid")
    output: list[tuple[Mapping[str, Any], int]] = []
    view = 0
    while len(output) < count:
        shuffled = list(rows)
        random.Random(f"{seed}:{view}").shuffle(shuffled)
        take = min(len(shuffled), count - len(output))
        output.extend((row, view) for row in shuffled[:take])
        view += 1
    return output


def _state_id(*, round_id: str, source_id: str, pool: str, stage: str, ordinal: int) -> str:
    return "bat:" + _digest({"round_id": round_id, "source_state_id": source_id, "source_pool": pool, "target_stage": stage, "ordinal": ordinal})[:32]


def _project(
    source: Mapping[str, Any],
    *,
    round_id: str,
    pool: str,
    stage: str,
    ordinal: int,
    view_index: int,
    source_sha256: str,
) -> dict[str, Any]:
    source_id = str(source["source_state_id"])
    extra = {
        "state_id": _state_id(round_id=round_id, source_id=source_id, pool=pool, stage=stage, ordinal=ordinal),
        "source_state_id": source_id,
        "target_stage": stage,
        "source_pool": pool,
        "focus_stage": source["focus_stage"],
        "outcome_bucket": source["outcome_bucket"],
        "track": source["track"],
        "view_index": view_index,
        "task_definition_version": TASK_DEFINITION_VERSION,
        "assistant_target_included": False,
        "teacher_trajectory_included": False,
        "rollout_transcript_reuse_allowed": False,
        "model_specific_serialization": False,
        "reward_contract": e2e_reward_contract() if stage == "E2E" else stage_reward_contract(stage),
        "provenance": {
            "source": "heldout_safe_synthetic",
            "source_file_sha256": source_sha256,
            "source_row_sha256": source["row_sha256"],
            "evaluation_content_used": False,
        },
    }
    return {
        "data_source": "automedbench_clean_prompt",
        "prompt": render_prompt(extra),
        "ability": f"automedbench:{stage}",
        "reward_model": {"style": "custom"},
        "extra_info": extra,
    }


def build_round(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    round_id: str,
    target_stage: str,
    batches: int = DEFAULT_BATCHES,
    seed: str | None = None,
    source_sha256: str = "0" * 64,
) -> list[dict[str, Any]]:
    validate_source(source_rows)
    if target_stage not in STAGES or batches <= 0 or not round_id.strip():
        raise ValueError("round_arguments_invalid")
    seed = seed or f"{round_id}:{target_stage}:bat-core-v1"
    target_source = [row for row in source_rows if row["focus_stage"] == target_stage]
    e2e_source = [row for row in source_rows if row["focus_stage"] == "E2E"]
    other_stages = [stage for stage in STAGES if stage != target_stage]
    mix_source = {stage: [row for row in source_rows if row["focus_stage"] == stage] for stage in other_stages}
    target_rows = _cycle(target_source, batches * TARGET_ROWS_PER_BATCH, seed=f"{seed}:target")
    e2e_rows = _cycle(e2e_source, batches * E2E_ROWS_PER_BATCH, seed=f"{seed}:e2e")
    mix_counts = Counter(other_stages[index % len(other_stages)] for index in range(batches * MIX_ROWS_PER_BATCH))
    mix_rows: dict[str, list[tuple[Mapping[str, Any], int]]] = {
        stage: _cycle(mix_source[stage], count, seed=f"{seed}:mix:{stage}")
        for stage, count in mix_counts.items()
    }
    mix_offsets = Counter()
    output: list[dict[str, Any]] = []
    target_offset = 0
    e2e_offset = 0
    ordinal = 0
    for batch_index in range(batches):
        batch: list[dict[str, Any]] = []
        for source, view in target_rows[target_offset : target_offset + TARGET_ROWS_PER_BATCH]:
            batch.append(_project(source, round_id=round_id, pool="s_target", stage=target_stage, ordinal=ordinal, view_index=view, source_sha256=source_sha256))
            ordinal += 1
        target_offset += TARGET_ROWS_PER_BATCH
        for mix_index in range(MIX_ROWS_PER_BATCH):
            stage = other_stages[(batch_index * MIX_ROWS_PER_BATCH + mix_index) % len(other_stages)]
            source, view = mix_rows[stage][mix_offsets[stage]]
            mix_offsets[stage] += 1
            batch.append(_project(source, round_id=round_id, pool="s_mix", stage=stage, ordinal=ordinal, view_index=view, source_sha256=source_sha256))
            ordinal += 1
        for source, view in e2e_rows[e2e_offset : e2e_offset + E2E_ROWS_PER_BATCH]:
            batch.append(_project(source, round_id=round_id, pool="e2e", stage="E2E", ordinal=ordinal, view_index=view, source_sha256=source_sha256))
            ordinal += 1
        e2e_offset += E2E_ROWS_PER_BATCH
        random.Random(f"{seed}:batch:{batch_index}").shuffle(batch)
        for slot, row in enumerate(batch):
            row["extra_info"]["core_batch_index"] = batch_index
            row["extra_info"]["core_batch_slot"] = slot
            output.append(row)
    return output


def verify_round_rows(rows: Sequence[Mapping[str, Any]], target_stage: str, batches: int) -> None:
    if len(rows) != batches * TRAIN_BATCH_SIZE:
        raise ValueError("round_row_count_invalid")
    state_ids: set[str] = set()
    for batch_index in range(batches):
        batch = rows[batch_index * TRAIN_BATCH_SIZE : (batch_index + 1) * TRAIN_BATCH_SIZE]
        pools = Counter(row["extra_info"]["source_pool"] for row in batch)
        if pools != Counter({"s_target": 4, "s_mix": 2, "e2e": 2}):
            raise ValueError("round_pool_composition_invalid")
        for row in batch:
            extra = row["extra_info"]
            if extra["source_pool"] == "s_target" and extra["target_stage"] != target_stage:
                raise ValueError("round_target_stage_invalid")
            if extra["source_pool"] == "s_mix" and extra["target_stage"] in {target_stage, "E2E"}:
                raise ValueError("round_mix_stage_invalid")
            if extra["source_pool"] == "e2e" and extra["target_stage"] != "E2E":
                raise ValueError("round_e2e_stage_invalid")
            state_id = extra["state_id"]
            if state_id in state_ids:
                raise ValueError("round_state_id_duplicate")
            state_ids.add(state_id)


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


def _prescribed_target(path: Path | None, target: str | None) -> str:
    value = target
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("prescription_invalid")
        prescribed = payload.get("target_stage")
        if value is not None and prescribed != value:
            raise ValueError("target_disagrees_with_prescription")
        value = prescribed
    if value not in STAGES:
        raise ValueError("target_stage_required")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--target-stage", choices=STAGES)
    parser.add_argument("--prescription", type=Path)
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--seed")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    source_sha256 = _file_sha256(source)
    if args.source_sha256 != source_sha256:
        raise ValueError("source_digest_mismatch")
    target = _prescribed_target(args.prescription, args.target_stage)
    source_rows = read_source(source)
    rows = build_round(source_rows, round_id=args.round_id, target_stage=target, batches=args.batches, seed=args.seed, source_sha256=source_sha256)
    verify_round_rows(rows, target, args.batches)
    content = b"".join((_stable(row) + "\n").encode("utf-8") for row in rows)
    output_dir = args.output_dir.resolve()
    train_path = output_dir / "train.jsonl"
    _write_immutable(train_path, content)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "bat_three_pool_round",
        "round_id": args.round_id,
        "target_stage": target,
        "batches": args.batches,
        "rows": len(rows),
        "rows_per_batch": 8,
        "continuations_per_state": 4,
        "pool_rows_per_batch": {"s_target": 4, "s_mix": 2, "e2e": 2},
        "source": {"file": source.name, "sha256": source_sha256, "rows": len(source_rows)},
        "train": {"file": train_path.name, "sha256": hashlib.sha256(content).hexdigest()},
        "model_specific_serialization": False,
        "assistant_targets_included": False,
    }
    _write_immutable(output_dir / "manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

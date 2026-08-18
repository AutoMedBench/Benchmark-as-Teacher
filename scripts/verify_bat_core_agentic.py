#!/usr/bin/env python3
"""Verify that every admitted row can produce a sealed sandbox snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl.grpo.core_agentic_sandbox import sandbox_snapshot_for_row
from rl.sandbox.state_package_runtime import validate_snapshot_manifest
from scripts.verify_bat_core_dataset import verify


def verify_agentic(train: Path, manifest: Path) -> dict[str, object]:
    base = verify(train, manifest)
    count = 0
    content_digests: set[str] = set()
    with train.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            snapshot, contract = sandbox_snapshot_for_row(row["prompt"], row["extra_info"])
            validated = validate_snapshot_manifest(snapshot)
            if contract["state_id"] != row["extra_info"]["state_id"]:
                raise ValueError("sandbox_state_binding_invalid")
            content_digests.add(validated.content_sha256)
            count += 1
    if count != base["rows"]:
        raise ValueError("sandbox_row_count_invalid")
    return {**base, "snapshots": count, "distinct_snapshots": len(content_digests)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_agentic(args.train.resolve(strict=True), args.manifest.resolve(strict=True)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

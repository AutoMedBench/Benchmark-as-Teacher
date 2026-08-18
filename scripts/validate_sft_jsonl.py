#!/usr/bin/env python3
"""Validate model-neutral semantic SFT source rows without rendering them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.sft_tool_protocol import ensure_no_model_specific_tool_tags, validate_assistant_target, validate_messages


def validate(path: Path) -> dict[str, int]:
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line_{line_number}_not_object")
            validate_messages(value.get("messages"))
            validate_assistant_target(value.get("target"))
            ensure_no_model_specific_tool_tags(value)
            mask = value.get("loss_mask")
            if not isinstance(mask, dict) or mask.get("messages") != 0 or mask.get("target_action") != 1:
                raise ValueError(f"line_{line_number}_loss_mask_invalid")
            rows += 1
    if not rows:
        raise ValueError("sft_source_empty")
    return {"rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.path.resolve(strict=True)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

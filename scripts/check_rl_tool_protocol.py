#!/usr/bin/env python3
"""Prove that an RL view is prompt-only and protocol-neutral."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from rl.grpo.tool_protocol_contract import validate_runtime_protocol


_TOOL_TAG = re.compile(r"<\s*/?\s*(?:tool|function)[^>]*>", re.IGNORECASE)
_FORBIDDEN_FIELDS = {"assistant", "completion", "response", "target", "teacher", "trajectory", "solution"}


class RLToolProtocolError(ValueError):
    pass


def validate_row(row: Any, *, line_number: int) -> None:
    if not isinstance(row, Mapping):
        raise RLToolProtocolError(f"line_{line_number}_not_object")
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise RLToolProtocolError(f"line_{line_number}_prompt_invalid")
    for message in prompt:
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise RLToolProtocolError(f"line_{line_number}_prompt_message_invalid")
        if message.get("role") not in {"system", "user"} or not isinstance(message.get("content"), str):
            raise RLToolProtocolError(f"line_{line_number}_assistant_content_present")
    rendered = json.dumps(row, ensure_ascii=False, sort_keys=True)
    if _TOOL_TAG.search(rendered):
        raise RLToolProtocolError(f"line_{line_number}_serialized_tool_syntax_present")
    for field in _FORBIDDEN_FIELDS:
        if field in row:
            raise RLToolProtocolError(f"line_{line_number}_forbidden_top_level_field_{field}")
    extra = row.get("extra_info")
    if not isinstance(extra, Mapping):
        raise RLToolProtocolError(f"line_{line_number}_extra_info_invalid")
    expected_false = (
        "assistant_target_included",
        "teacher_trajectory_included",
        "rollout_transcript_reuse_allowed",
        "model_specific_serialization",
    )
    if any(extra.get(field) is not False for field in expected_false):
        raise RLToolProtocolError(f"line_{line_number}_neutrality_contract_invalid")


def audit(path: Path, *, runtime_tool_format: str) -> dict[str, Any]:
    protocol = validate_runtime_protocol(runtime_tool_format)
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RLToolProtocolError(f"line_{line_number}_invalid_json") from exc
            validate_row(value, line_number=line_number)
            count += 1
    if not count:
        raise RLToolProtocolError("rl_dataset_empty")
    return {
        "schema_version": 1,
        "status": "passed",
        "rows": count,
        "dataset_sha256": digest.hexdigest(),
        "runtime_tool_format": protocol,
        "model_specific_serialization_in_dataset": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--runtime-tool-format", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.path.resolve(strict=True), runtime_tool_format=args.runtime_tool_format), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

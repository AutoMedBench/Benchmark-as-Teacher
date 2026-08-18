"""Runtime-only binding for a model's native tool-call protocol.

Shared SFT and RL rows contain structured actions or prompts, never serialized
tool tags.  The launcher records an adapter identifier and the serving runtime
performs the actual rendering and parsing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Mapping


RUNTIME_TOOL_FORMAT_ENV = "BAT_RUNTIME_TOOL_FORMAT"
_SAFE_PROTOCOL = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


class ToolProtocolContractError(ValueError):
    """The runtime protocol binding is missing or malformed."""


def validate_runtime_protocol(value: Any) -> str:
    if not isinstance(value, str):
        raise ToolProtocolContractError("runtime_tool_format_missing")
    protocol = value.strip().lower()
    if _SAFE_PROTOCOL.fullmatch(protocol) is None:
        raise ToolProtocolContractError("runtime_tool_format_invalid")
    return protocol


def runtime_protocol_binding(value: str | None = None) -> dict[str, Any]:
    """Return a stable receipt for the explicitly selected runtime adapter."""

    raw = value if value is not None else os.environ.get(RUNTIME_TOOL_FORMAT_ENV)
    protocol = validate_runtime_protocol(raw)
    payload = {
        "schema_version": 1,
        "runtime_tool_format": protocol,
        "binding_scope": "runtime_only",
        "model_specific_serialization_in_training_data": False,
        "injected_first_action": False,
    }
    payload["binding_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def validate_protocol_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ToolProtocolContractError("runtime_tool_receipt_invalid")
    observed = dict(value)
    expected = runtime_protocol_binding(observed.get("runtime_tool_format"))
    if observed != expected:
        raise ToolProtocolContractError("runtime_tool_receipt_mismatch")
    return observed


__all__ = [
    "RUNTIME_TOOL_FORMAT_ENV",
    "ToolProtocolContractError",
    "runtime_protocol_binding",
    "validate_protocol_receipt",
    "validate_runtime_protocol",
]

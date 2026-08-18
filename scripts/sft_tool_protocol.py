"""Semantic SFT action validation and tokenizer-native rendering."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_TAG_LIKE_TOOL_SYNTAX = re.compile(r"<\s*/?\s*(?:tool|function)[^>]*>", re.IGNORECASE)


class ToolProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class TokenProjection:
    prompt_ids: tuple[int, ...]
    full_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    semantic_action_sha256: str


def stable_semantic_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def ensure_no_model_specific_tool_tags(value: Any) -> None:
    rendered = stable_semantic_json(value)
    if _TAG_LIKE_TOOL_SYNTAX.search(rendered):
        raise ToolProtocolError("model_specific_tool_syntax_in_shared_source")


def validate_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ToolProtocolError("messages_invalid")
    rows: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ToolProtocolError("message_invalid")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
            raise ToolProtocolError("message_invalid")
        rows.append({"role": role, "content": content})
    return rows


def validate_assistant_target(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("role") != "assistant":
        raise ToolProtocolError("assistant_target_invalid")
    reasoning = value.get("reasoning_summary", "")
    if not isinstance(reasoning, str):
        raise ToolProtocolError("assistant_reasoning_invalid")
    action = value.get("action")
    if not isinstance(action, Mapping) or action.get("type") != "tool":
        raise ToolProtocolError("assistant_action_invalid")
    if action.get("name") != "execute_code":
        raise ToolProtocolError("assistant_tool_name_invalid")
    arguments = action.get("arguments")
    if not isinstance(arguments, Mapping) or set(arguments) != {"language", "code"}:
        raise ToolProtocolError("assistant_tool_arguments_invalid")
    if arguments.get("language") not in {"python", "bash"}:
        raise ToolProtocolError("assistant_tool_language_invalid")
    if not isinstance(arguments.get("code"), str) or not arguments["code"].strip():
        raise ToolProtocolError("assistant_tool_code_invalid")
    normalized = {
        "role": "assistant",
        "reasoning_summary": reasoning,
        "action": {
            "type": "tool",
            "name": "execute_code",
            "arguments": {"language": arguments["language"], "code": arguments["code"]},
        },
    }
    ensure_no_model_specific_tool_tags(normalized)
    return normalized


def target_message(target: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_assistant_target(target)
    action = value["action"]
    return {
        "role": "assistant",
        "content": value["reasoning_summary"],
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": action["name"],
                    "arguments": dict(action["arguments"]),
                },
            }
        ],
    }


def execute_code_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": "Run Python or bash in the current isolated workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "language": {"type": "string", "enum": ["python", "bash"]},
                        "code": {"type": "string"},
                    },
                    "required": ["language", "code"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _ids(tokenizer: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
    try:
        value = tokenizer.apply_chat_template(
            messages,
            tools=execute_code_schema(),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception as exc:
        raise ToolProtocolError("tokenizer_chat_template_tool_render_failed") from exc
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not value or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ToolProtocolError("tokenizer_chat_template_ids_invalid")
    return value


def render_semantic_example(tokenizer: Any, messages: Any, target: Any) -> TokenProjection:
    context = validate_messages(messages)
    normalized_target = validate_assistant_target(target)
    ensure_no_model_specific_tool_tags({"messages": context, "target": normalized_target})
    prompt_ids = _ids(tokenizer, context, add_generation_prompt=True)
    full_ids = _ids(tokenizer, [*context, target_message(normalized_target)], add_generation_prompt=False)
    prefix = 0
    for left, right in zip(prompt_ids, full_ids):
        if left != right:
            break
        prefix += 1
    if prefix == 0 or prefix >= len(full_ids):
        raise ToolProtocolError("target_token_boundary_unavailable")
    target_ids = full_ids[prefix:]
    return TokenProjection(
        tuple(prompt_ids[:prefix]),
        tuple(full_ids),
        tuple(target_ids),
        hashlib.sha256(stable_semantic_json(normalized_target["action"]).encode("utf-8")).hexdigest(),
    )


__all__ = [
    "TokenProjection",
    "ToolProtocolError",
    "ensure_no_model_specific_tool_tags",
    "execute_code_schema",
    "render_semantic_example",
    "stable_semantic_json",
    "target_message",
    "validate_assistant_target",
    "validate_messages",
]

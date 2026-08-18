"""Optional fail-closed JSON rubric judge using an OpenAI-compatible endpoint."""
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping


class RubricJudgeConfigurationError(ValueError):
    pass


class RubricJudgeResponseError(ValueError):
    pass


@dataclass(frozen=True)
class JudgeConfig:
    endpoint: str
    model: str
    api_key_env: str = "BAT_JUDGE_API_KEY"
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlparse(self.endpoint)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise RubricJudgeConfigurationError("judge_endpoint_must_be_https")
        if parsed.query or parsed.fragment:
            raise RubricJudgeConfigurationError("judge_endpoint_query_forbidden")
        if not self.model.strip() or len(self.model) > 256:
            raise RubricJudgeConfigurationError("judge_model_invalid")
        if not self.api_key_env.startswith("BAT_"):
            raise RubricJudgeConfigurationError("judge_api_key_env_invalid")
        if not 1 <= self.timeout_seconds <= 600:
            raise RubricJudgeConfigurationError("judge_timeout_invalid")


def _unit(value: Any) -> float:
    if isinstance(value, bool):
        raise RubricJudgeResponseError("judge_score_invalid")
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RubricJudgeResponseError("judge_score_invalid") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RubricJudgeResponseError("judge_score_invalid")
    return score


def validate_rubric_judge_payload(
    value: Any,
    reward_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RubricJudgeResponseError("judge_payload_not_object")
    items = reward_contract.get("items")
    if not isinstance(items, list):
        raise RubricJudgeResponseError("reward_contract_items_invalid")
    expected = [item.get("id") for item in items if isinstance(item, Mapping)]
    observed = value.get("item_scores")
    if not isinstance(observed, Mapping) or set(observed) != set(expected):
        raise RubricJudgeResponseError("judge_items_mismatch")
    scores = {item: _unit(observed[item]) for item in expected}
    rationales = value.get("rationales")
    if rationales is not None and (
        not isinstance(rationales, Mapping)
        or set(rationales) != set(expected)
        or any(not isinstance(text, str) or len(text) > 1000 for text in rationales.values())
    ):
        raise RubricJudgeResponseError("judge_rationales_invalid")
    return {"item_scores": scores, "rationales": dict(rationales or {})}


class OpenAICompatibleRubricJudge:
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    def judge(
        self,
        *,
        conversation: Mapping[str, Any],
        task: str,
        reward_contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RubricJudgeConfigurationError("judge_api_key_missing")
        item_ids = [item["id"] for item in reward_contract.get("items", [])]
        system = (
            "Score only retained observable evidence. Return one JSON object with "
            "item_scores in [0,1] and short rationales for exactly the requested ids."
        )
        user = json.dumps(
            {
                "task": task,
                "rubric_item_ids": item_ids,
                "conversation": conversation,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        body = json.dumps(
            {
                "model": self.config.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.config.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RubricJudgeResponseError("judge_request_failed") from exc
        try:
            text = payload["choices"][0]["message"]["content"]
            value = json.loads(text)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RubricJudgeResponseError("judge_response_invalid") from exc
        return validate_rubric_judge_payload(value, reward_contract)


class FakeJudge:
    """Explicit test double; production builders never select it."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = dict(payload)

    def judge(self, **kwargs: Any) -> dict[str, Any]:
        return validate_rubric_judge_payload(self.payload, kwargs["reward_contract"])


def build_rubric_judge() -> OpenAICompatibleRubricJudge:
    endpoint = os.environ.get("BAT_JUDGE_ENDPOINT")
    model = os.environ.get("BAT_JUDGE_MODEL")
    if not endpoint or not model:
        raise RubricJudgeConfigurationError("judge_configuration_missing")
    raw_timeout = os.environ.get("BAT_JUDGE_TIMEOUT_SECONDS", "60")
    try:
        timeout = int(raw_timeout)
    except ValueError as exc:
        raise RubricJudgeConfigurationError("judge_timeout_invalid") from exc
    return OpenAICompatibleRubricJudge(JudgeConfig(endpoint, model, timeout_seconds=timeout))


def judge_rubric_contract(
    judge: Any,
    conversation: Mapping[str, Any],
    task: str,
    reward_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not hasattr(judge, "judge"):
        raise RubricJudgeConfigurationError("judge_invalid")
    return judge.judge(
        conversation=conversation,
        task=task,
        reward_contract=reward_contract,
    )


__all__ = [
    "FakeJudge",
    "JudgeConfig",
    "OpenAICompatibleRubricJudge",
    "RubricJudgeConfigurationError",
    "RubricJudgeResponseError",
    "build_rubric_judge",
    "judge_rubric_contract",
    "validate_rubric_judge_payload",
]

"""VERL reward-manager bridge for contract-bound BaT rewards."""
from __future__ import annotations

import inspect
import math
from typing import Any, Mapping

from rl.grpo.core_reward import compute_score_with_breakdown, is_verified_reward_payload

try:  # Supplied by the training environment.
    from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
except ModuleNotFoundError:  # pragma: no cover
    RewardManagerBase = object  # type: ignore[assignment,misc]


_SOURCE_POOLS = {"s_target", "s_mix", "e2e"}


def _plain(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (dict, list, tuple, str, bytes)):
        try:
            return item()
        except Exception:
            pass
    return value


def _mapping(value: Any) -> dict[str, Any]:
    value = _plain(value)
    return dict(value) if isinstance(value, Mapping) else {}


def merge_runtime_evidence(extra_info: Any, tool_extra_fields: Any) -> dict[str, Any]:
    merged = _mapping(extra_info)
    for key, value in _mapping(tool_extra_fields).items():
        if key in merged and merged[key] != value:
            raise RuntimeError("reward_extra_field_collision")
        merged[key] = value
    return merged


class CoreRewardManager(RewardManagerBase):  # type: ignore[misc]
    """Decode one rollout and require a verified finite reward payload."""

    def __init__(
        self,
        config: Any,
        tokenizer: Any,
        compute_score: Any = None,
        reward_router_address: Any = None,
        reward_model_tokenizer: Any = None,
        **_: Any,
    ) -> None:
        if RewardManagerBase is object:
            raise RuntimeError("verl_not_installed")
        scorer = compute_score or compute_score_with_breakdown
        super().__init__(config, tokenizer, scorer)
        self.compute_score = scorer
        self.is_async_reward_score = inspect.iscoroutinefunction(scorer)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: Any) -> dict[str, Any]:
        data_item = data[-1:][0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())
        valid_ids = response_ids[:valid_length]
        response = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_ids, skip_special_tokens=True),
        )
        extra = merge_runtime_evidence(
            data_item.non_tensor_batch.get("extra_info"),
            data_item.non_tensor_batch.get("tool_extra_fields"),
        )
        source_pool = extra.get("source_pool")
        if source_pool not in _SOURCE_POOLS:
            raise RuntimeError("source_pool_invalid")
        kwargs = {
            "data_source": _plain(data_item.non_tensor_batch.get("data_source")),
            "solution_str": response,
            "ground_truth": _mapping(data_item.non_tensor_batch.get("reward_model")).get("ground_truth"),
            "extra_info": extra,
        }
        if self.is_async_reward_score:
            payload = await self.compute_score(**kwargs)
        else:
            payload = await self.loop.run_in_executor(None, lambda: self.compute_score(**kwargs))
        if not is_verified_reward_payload(payload):
            raise RuntimeError("unverified_bat_core_reward")
        score = float(payload["score"])
        if not math.isfinite(score):
            raise RuntimeError("nonfinite_bat_core_reward")
        return {
            "reward_score": score,
            "reward_extra_info": {
                "source_pool": source_pool,
                "score": score,
                "acc": score,
                "target_stage": payload.get("target_stage"),
                "reward_version": payload.get("reward_version"),
                "raw_score": payload.get("raw_score"),
                "gate_cap": payload.get("gate_cap"),
                "gate_reason": payload.get("gate_reason"),
                "item_scores": payload.get("item_scores"),
                "stage_scores": payload.get("stage_scores"),
                "tool_calls": payload.get("agentic_tool_calls"),
                "successful_tool_calls": payload.get("agentic_successful_tool_calls"),
                "workspace_changed": payload.get("agentic_workspace_changed"),
                "current_stage_complete": payload.get("agentic_current_stage_complete"),
                "contract_sha256": payload.get("contract_sha256"),
                "evidence_sha256": payload.get("evidence_sha256"),
            },
        }


__all__ = ["CoreRewardManager", "merge_runtime_evidence"]

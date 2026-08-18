"""Run stock verl sync GRPO with fail-closed rollout endpoint admission."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from typing import Any


EXPECTED_REPLICAS_ENV = "BAT_EXPECTED_ROLLOUT_REPLICAS"
POOL_REWARD_AVERAGES_ENV = "BAT_LOG_POOL_REWARD_AVERAGES"
EXPECTED_POOL_ROLLOUT_COUNTS = {
    "s_target": 16,
    "s_mix": 8,
    "e2e": 8,
}
_POOL_LABELS: ContextVar[tuple[str, ...] | None] = ContextVar(
    "bat_pool_reward_labels", default=None
)


def _pool_reward_logging_enabled() -> bool:
    raw = os.environ.get(POOL_REWARD_AVERAGES_ENV, "").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    raise RuntimeError(f"{POOL_REWARD_AVERAGES_ENV} must be a boolean")


def extract_source_pool_labels(
    extra_fields: Sequence[Any], *, non_padding_mask: Sequence[Any]
) -> tuple[str, ...]:
    """Extract source-pool labels in the balanced metric-batch order."""
    if len(extra_fields) != len(non_padding_mask):
        raise RuntimeError("source-pool metadata length does not match metric batch")
    labels: list[str] = []
    for extra, keep in zip(extra_fields, non_padding_mask, strict=True):
        if not bool(keep):
            continue
        if not isinstance(extra, Mapping):
            raise RuntimeError("source-pool metric metadata is missing")
        reward_info = extra.get("reward_extra_info")
        if not isinstance(reward_info, Mapping):
            raise RuntimeError("source-pool reward metadata is missing")
        label = reward_info.get("source_pool")
        if not isinstance(label, str) or label not in EXPECTED_POOL_ROLLOUT_COUNTS:
            raise RuntimeError("source-pool reward label is invalid")
        labels.append(label)
    counts = Counter(labels)
    if counts != Counter(EXPECTED_POOL_ROLLOUT_COUNTS):
        raise RuntimeError(
            "source-pool rollout composition changed: "
            f"expected={EXPECTED_POOL_ROLLOUT_COUNTS} observed={dict(counts)}"
        )
    return tuple(labels)


def compute_source_pool_reward_averages(
    batch: Any, source_pools: Sequence[str]
) -> dict[str, float]:
    """Average existing per-rollout rewards for the three fixed source pools."""
    import torch

    sequence_rewards = batch.batch["token_level_rewards"].sum(-1)
    if sequence_rewards.ndim != 1 or sequence_rewards.shape[0] != len(source_pools):
        raise RuntimeError("source-pool labels do not align with sequence rewards")
    counts = Counter(source_pools)
    if counts != Counter(EXPECTED_POOL_ROLLOUT_COUNTS):
        raise RuntimeError(
            "source-pool rollout composition changed: "
            f"expected={EXPECTED_POOL_ROLLOUT_COUNTS} observed={dict(counts)}"
        )
    response_length = batch.batch.get("response_length")
    if response_length is None or response_length.shape != sequence_rewards.shape:
        raise RuntimeError("source-pool reward logging requires aligned response lengths")
    non_aborted = response_length != 0
    metrics: dict[str, float] = {}
    for pool in EXPECTED_POOL_ROLLOUT_COUNTS:
        pool_mask = torch.tensor(
            [label == pool for label in source_pools],
            dtype=torch.bool,
            device=sequence_rewards.device,
        )
        selected = sequence_rewards[pool_mask & non_aborted]
        if selected.numel() == 0:
            raise RuntimeError(f"source-pool reward average has no valid rows: {pool}")
        average = selected.mean().detach().item()
        if not math.isfinite(average):
            raise RuntimeError(f"source-pool reward average is non-finite: {pool}")
        metrics[f"critic/rewards/{pool}/mean"] = float(average)
    return metrics


def validate_rollout_endpoints(
    addresses: Sequence[Any], *, expected: int
) -> dict[str, int | str]:
    """Validate the post-Ray endpoint topology without logging raw addresses."""
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise ValueError("expected rollout replicas must be a positive integer")
    if isinstance(addresses, (str, bytes)) or not isinstance(addresses, Sequence):
        raise RuntimeError("rollout server addresses must be a sequence")
    rendered = [str(address).strip() for address in addresses]
    observed = len(rendered)
    distinct = len(set(rendered))
    if any(not address for address in rendered):
        raise RuntimeError("rollout endpoint admission found an empty address")
    if observed != expected or distinct != expected:
        raise RuntimeError(
            "rollout endpoint admission failed: "
            f"expected={expected} observed={observed} distinct={distinct}"
        )
    digest = hashlib.sha256(
        json.dumps(sorted(rendered), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "expected": expected,
        "observed": observed,
        "distinct": distinct,
        "endpoints_sha256": digest,
    }


def install_rollout_endpoint_admission() -> None:
    """Guard Verl after Ray creates servers and before its load balancer starts."""
    raw_expected = os.environ.get(EXPECTED_REPLICAS_ENV, "")
    try:
        expected = int(raw_expected)
    except ValueError as exc:
        raise RuntimeError(f"{EXPECTED_REPLICAS_ENV} must be an integer") from exc
    if expected < 1:
        raise RuntimeError(f"{EXPECTED_REPLICAS_ENV} must be positive")

    from verl.workers.rollout.llm_server import LLMServerManager

    original = LLMServerManager._initialize_llm_servers
    if getattr(original, "_bat_endpoint_admission", False):
        return

    async def guarded_initialize(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await original(self, *args, **kwargs)
        runtime_tp = int(self.rollout_config.tensor_model_parallel_size)
        if runtime_tp != 1:
            raise RuntimeError(
                "rollout endpoint admission requires runtime tensor parallel size 1; "
                f"observed={runtime_tp}"
            )
        receipt = validate_rollout_endpoints(self.server_addresses, expected=expected)
        print(
            "[BaT core] rollout_endpoint_admission=PASS "
            f"tp={runtime_tp} expected={receipt['expected']} "
            f"observed={receipt['observed']} distinct={receipt['distinct']} "
            f"endpoints_sha256={receipt['endpoints_sha256']}",
            flush=True,
        )
        return result

    guarded_initialize._bat_endpoint_admission = True  # type: ignore[attr-defined]
    LLMServerManager._initialize_llm_servers = guarded_initialize


def install_source_pool_reward_average_logging() -> None:
    """Add three CPU-only averages to stock VERL metrics when explicitly enabled."""
    if not _pool_reward_logging_enabled():
        return

    from verl.trainer import main_ppo_sync

    original_compute_metrics = main_ppo_sync.PPOTrainer._compute_metrics
    if getattr(original_compute_metrics, "_bat_pool_reward_averages", False):
        return
    original_compute_data_metrics = main_ppo_sync.compute_data_metrics

    def compute_data_metrics_with_pool_averages(
        batch: Any, use_critic: bool = True
    ) -> dict[str, Any]:
        metrics = original_compute_data_metrics(batch=batch, use_critic=use_critic)
        labels = _POOL_LABELS.get()
        if labels is None:
            raise RuntimeError("source-pool labels were not installed for metric computation")
        metrics.update(compute_source_pool_reward_averages(batch, labels))
        return metrics

    def compute_metrics_with_pool_labels(
        self: Any,
        batch: Any,
        metrics: dict[str, Any],
        timing_raw: dict[str, Any],
        global_steps: int,
        epoch: int,
    ) -> Any:
        pool_data = main_ppo_sync.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["extra_fields"],
        )
        raw_extra_fields = pool_data["extra_fields"]
        extra_fields = (
            raw_extra_fields.tolist()
            if hasattr(raw_extra_fields, "tolist")
            else list(raw_extra_fields)
        )
        labels = extract_source_pool_labels(
            extra_fields,
            non_padding_mask=[not tag.get("is_padding", False) for tag in batch.tags],
        )
        token = _POOL_LABELS.set(labels)
        try:
            return original_compute_metrics(
                self,
                batch,
                metrics,
                timing_raw,
                global_steps,
                epoch,
            )
        finally:
            _POOL_LABELS.reset(token)

    compute_data_metrics_with_pool_averages._bat_pool_reward_averages = True  # type: ignore[attr-defined]
    compute_metrics_with_pool_labels._bat_pool_reward_averages = True  # type: ignore[attr-defined]
    main_ppo_sync.compute_data_metrics = compute_data_metrics_with_pool_averages
    main_ppo_sync.PPOTrainer._compute_metrics = compute_metrics_with_pool_labels


if __name__ == "__main__":
    install_rollout_endpoint_admission()
    install_source_pool_reward_average_logging()
    from verl.trainer import main_ppo_sync

    main_ppo_sync.main()

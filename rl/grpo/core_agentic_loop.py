"""Model-neutral multi-turn loop that attaches BaT sandbox evidence."""
from __future__ import annotations

from typing import Any

from rl.grpo.core_agentic_sandbox import CoreSandboxExecuteCodeTool, safe_agentic_error_code

try:  # VERL is supplied by the training environment, not this source bundle.
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
except ModuleNotFoundError:  # pragma: no cover - exercised only without the RL extra
    ToolAgentLoop = object  # type: ignore[assignment,misc]


def align_agent_loop_response_logprobs(output: Any) -> bool:
    """Expand packed assistant log-probabilities onto the transcript axis."""

    response_ids = list(output.response_ids)
    response_mask = list(output.response_mask)
    if len(response_mask) != len(response_ids):
        raise RuntimeError("bat_response_axis_mismatch")
    if any(value not in (0, 1, False, True) for value in response_mask):
        raise RuntimeError("bat_response_mask_not_binary")
    if output.response_logprobs is None:
        return False
    packed = list(output.response_logprobs)
    if len(packed) == len(response_ids):
        return False
    selected = sum(bool(value) for value in response_mask)
    if len(packed) != selected:
        raise RuntimeError("bat_rollout_logprob_axis_ambiguous")
    values = iter(packed)
    output.response_logprobs = [
        next(values) if bool(keep) else 0.0 for keep in response_mask
    ]
    return True


class CoreAgenticToolLoop(ToolAgentLoop):  # type: ignore[misc]
    """Lease one persistent workspace, run stock VERL, and retain evidence."""

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any):
        if ToolAgentLoop is object:
            raise RuntimeError("verl_not_installed")
        tool = self.tools.get("execute_code")
        if not isinstance(tool, CoreSandboxExecuteCodeTool):
            raise RuntimeError("bat_execute_code_tool_missing")
        if kwargs.get("tools_kwargs") not in (None, {}):
            raise RuntimeError("bat_tools_kwargs_must_be_runtime_owned")
        lease = tool.prepare_trajectory(
            prompt=kwargs.get("raw_prompt"),
            extra_info=kwargs.get("extra_info"),
        )
        run_kwargs = dict(kwargs)
        run_kwargs["tools_kwargs"] = {
            "execute_code": {"create_kwargs": tool.create_kwargs(lease)}
        }
        try:
            output = await super().run(sampling_params, **run_kwargs)
            align_agent_loop_response_logprobs(output)
            transcript = self.tokenizer.decode(
                output.response_ids,
                skip_special_tokens=False,
            )
            evidence, summary = await tool.finalize_trajectory(
                lease,
                transcript=transcript,
                response_tokens=len(output.response_ids),
                num_turns=int(output.num_turns),
            )
        except BaseException as exc:
            try:
                await tool.abort_trajectory(lease)
            except Exception:
                pass
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RuntimeError(
                f"bat_agentic_rollout_failed:{safe_agentic_error_code(exc)}"
            ) from exc
        fields = getattr(output, "extra_fields", None)
        if not isinstance(fields, dict):
            raise RuntimeError("bat_rollout_extra_fields_missing")
        for key in ("bat_sandbox_evidence", "bat_agentic_summary"):
            if key in fields:
                raise RuntimeError("bat_extra_field_collision")
        fields["bat_sandbox_evidence"] = evidence
        fields["bat_agentic_summary"] = summary
        return output


__all__ = ["CoreAgenticToolLoop", "align_agent_loop_response_logprobs"]

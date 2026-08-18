"""Capability-scoped execute-code tool used by BaT multi-turn rollouts."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from rl.sandbox.container_runtime import PinnedContainerRuntime, stable_sha256


class CoreWorkspaceToolError(RuntimeError):
    """A workspace capability or tool request is invalid."""


@dataclass(frozen=True)
class WorkspaceCapability:
    token: str
    workspace: Path
    session_id: str
    state_id: str
    reset_fingerprint: str
    max_steps: int


@dataclass
class _Session:
    capability: WorkspaceCapability
    instance_id: str
    calls: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False
    execute_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class CoreWorkspaceExecuteCodeTool:
    """Bind opaque capabilities to one persistent isolated workspace each."""

    def __init__(
        self,
        *,
        runtime: PinnedContainerRuntime,
        tool_schema: Any,
        max_steps: int = 23,
    ) -> None:
        if not isinstance(runtime, PinnedContainerRuntime):
            raise CoreWorkspaceToolError("runtime_invalid")
        if not 1 <= max_steps <= 128:
            raise CoreWorkspaceToolError("max_steps_invalid")
        self.runtime = runtime
        self.tool_schema = tool_schema
        self.name = "execute_code"
        self.max_steps = max_steps
        self._capabilities: dict[str, WorkspaceCapability] = {}
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()

    def bind_workspace(
        self,
        *,
        workspace_path: Path,
        session_id: str,
        state_id: str,
        reset_fingerprint: str,
        max_steps: int | None = None,
    ) -> WorkspaceCapability:
        workspace = workspace_path.resolve(strict=True)
        if self.runtime.allowed_root not in workspace.parents:
            raise CoreWorkspaceToolError("workspace_outside_runtime_root")
        for label, value in (
            ("session_id", session_id),
            ("state_id", state_id),
            ("reset_fingerprint", reset_fingerprint),
        ):
            if not isinstance(value, str) or not value or len(value) > 256:
                raise CoreWorkspaceToolError(f"{label}_invalid")
        budget = max_steps if max_steps is not None else self.max_steps
        if not isinstance(budget, int) or isinstance(budget, bool) or not 1 <= budget <= 128:
            raise CoreWorkspaceToolError("max_steps_invalid")
        capability = WorkspaceCapability(
            token=secrets.token_urlsafe(32),
            workspace=workspace,
            session_id=session_id,
            state_id=state_id,
            reset_fingerprint=reset_fingerprint,
            max_steps=budget,
        )
        with self._lock:
            self._capabilities[capability.token] = capability
        return capability

    @staticmethod
    def create_kwargs(capability: WorkspaceCapability) -> dict[str, str]:
        if not isinstance(capability, WorkspaceCapability):
            raise CoreWorkspaceToolError("capability_invalid")
        return {"capability_token": capability.token}

    async def create(
        self,
        instance_id: str | None = None,
        *,
        capability_token: str,
        **_: Any,
    ) -> tuple[str, dict[str, Any]]:
        with self._lock:
            capability = self._capabilities.pop(capability_token, None)
            if capability is None:
                raise CoreWorkspaceToolError("capability_missing_or_reused")
            selected = instance_id or secrets.token_hex(16)
            if selected in self._sessions:
                raise CoreWorkspaceToolError("instance_id_collision")
            self._sessions[selected] = _Session(capability=capability, instance_id=selected)
        return selected, {
            "schema_version": 1,
            "tool": self.name,
            "state_id_sha256": hashlib.sha256(
                capability.state_id.encode("utf-8")
            ).hexdigest(),
            "max_steps": capability.max_steps,
        }

    def _session(self, instance_id: str) -> _Session:
        with self._lock:
            session = self._sessions.get(instance_id)
            if session is None or session.closed:
                raise CoreWorkspaceToolError("tool_session_missing")
            return session

    async def execute(
        self,
        instance_id: str,
        parameters: Mapping[str, Any],
        **_: Any,
    ) -> tuple[str, float, dict[str, Any]]:
        if not isinstance(parameters, Mapping) or set(parameters) != {"language", "code"}:
            raise CoreWorkspaceToolError("execute_code_arguments_invalid")
        language, code = parameters.get("language"), parameters.get("code")
        if not isinstance(language, str) or not isinstance(code, str):
            raise CoreWorkspaceToolError("execute_code_arguments_invalid")
        session = self._session(instance_id)
        async with session.execute_lock:
            with self._lock:
                if session.closed:
                    raise CoreWorkspaceToolError("tool_session_closed")
                if len(session.calls) >= session.capability.max_steps:
                    raise CoreWorkspaceToolError("tool_call_budget_exhausted")
            receipt = await asyncio.to_thread(
                self.runtime.execute,
                session.capability.workspace,
                language,
                code,
            )
            public_receipt = {
                "exit_code": receipt["exit_code"],
                "timed_out": receipt["timed_out"],
                "stdout": receipt["stdout"],
                "stderr": receipt["stderr"],
                "workspace_delta": receipt["workspace_delta"],
            }
            call = {
                "index": len(session.calls),
                "tool": self.name,
                "language": language,
                "code_sha256": receipt["code_sha256"],
                "exit_code": receipt["exit_code"],
                "timed_out": receipt["timed_out"],
                "pre_workspace_sha256": receipt["pre_workspace_sha256"],
                "post_workspace_sha256": receipt["post_workspace_sha256"],
                "receipt_sha256": receipt["receipt_sha256"],
            }
            with self._lock:
                session.calls.append(call)
        text = (
            f"exit_code={receipt['exit_code']}\n"
            f"stdout:\n{receipt['stdout']}\n"
            f"stderr:\n{receipt['stderr']}"
        )
        return text, 0.0, {"receipt": public_receipt}

    async def release(self, instance_id: str, **_: Any) -> None:
        with self._lock:
            session = self._sessions.get(instance_id)
            if session is not None:
                session.closed = True

    async def abort_workspace(self, capability: WorkspaceCapability) -> None:
        with self._lock:
            self._capabilities.pop(capability.token, None)
            for session in self._sessions.values():
                if session.capability.token == capability.token:
                    session.closed = True

    async def finalize_workspace(
        self,
        capability: WorkspaceCapability,
        *,
        allow_no_calls: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        with self._lock:
            self._capabilities.pop(capability.token, None)
            matches = [
                session
                for session in self._sessions.values()
                if session.capability.token == capability.token
            ]
            calls = [copy.deepcopy(call) for item in matches for call in item.calls]
            for session in matches:
                session.closed = True
        if not calls and not allow_no_calls:
            raise CoreWorkspaceToolError("trajectory_has_no_tool_calls")
        evidence = {
            "schema_version": 1,
            "state_id_sha256": hashlib.sha256(
                capability.state_id.encode("utf-8")
            ).hexdigest(),
            "reset_fingerprint": capability.reset_fingerprint,
            "calls": calls,
            "tool_calls": len(calls),
            "successful_tool_calls": sum(call["exit_code"] == 0 for call in calls),
            "workspace_changed": any(
                call["pre_workspace_sha256"] != call["post_workspace_sha256"]
                for call in calls
            ),
        }
        evidence["evidence_sha256"] = stable_sha256(evidence)
        return evidence


__all__ = [
    "CoreWorkspaceExecuteCodeTool",
    "CoreWorkspaceToolError",
    "WorkspaceCapability",
]

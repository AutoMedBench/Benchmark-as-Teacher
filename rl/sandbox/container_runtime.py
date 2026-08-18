"""Small fail-closed container executor for BaT rollout workspaces.

The image is an input, not a repository default.  It must be pinned by digest;
networking, privilege escalation, GPUs, and host paths outside the leased
workspace are unavailable to executed code.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SANDBOX_IMAGE_ENV = "BAT_SANDBOX_IMAGE"
_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9./:_-]*@sha256:[0-9a-f]{64}$")
_LANGUAGES = {"python", "bash"}
_MAX_CODE_BYTES = 256 * 1024
_MAX_CAPTURE_BYTES = 128 * 1024


class ContainerRuntimeRefusal(RuntimeError):
    """Execution was rejected before untrusted code ran."""


class ContainerCleanupRefusal(ContainerRuntimeRefusal):
    """Workspace cleanup could not be proved safe."""


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def validate_image_reference(value: Any) -> str:
    if not isinstance(value, str) or _IMAGE.fullmatch(value.strip()) is None:
        raise ContainerRuntimeRefusal("sandbox_image_must_be_digest_pinned")
    return value.strip()


def runtime_attestation_sha256(image: str | None = None) -> str:
    selected = validate_image_reference(image or os.environ.get(SANDBOX_IMAGE_ENV))
    return stable_sha256(
        {
            "schema_version": 1,
            "image": selected,
            "network": "none",
            "capabilities": "drop_all",
            "no_new_privileges": True,
            "workspace_mount": "single_rw_bind",
        }
    )


def _regular_binary(value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        located = shutil.which(str(candidate))
        if located is None:
            raise ContainerRuntimeRefusal("container_runtime_not_found")
        candidate = Path(located)
    try:
        resolved = candidate.resolve(strict=True)
        info = resolved.stat()
    except OSError as exc:
        raise ContainerRuntimeRefusal("container_runtime_not_found") from exc
    if not stat.S_ISREG(info.st_mode) or not os.access(resolved, os.X_OK):
        raise ContainerRuntimeRefusal("container_runtime_not_executable")
    return resolved


def _real_workspace(path: Path, *, allowed_root: Path) -> Path:
    try:
        root = allowed_root.resolve(strict=True)
        workspace = path.resolve(strict=True)
        root_info = allowed_root.lstat()
        workspace_info = path.lstat()
    except OSError as exc:
        raise ContainerRuntimeRefusal("workspace_invalid") from exc
    if (
        root.is_symlink()
        or allowed_root != root
        or path.is_symlink()
        or path != workspace
        or not root.is_dir()
        or not workspace.is_dir()
        or workspace == root
        or root not in workspace.parents
        or workspace_info.st_uid != os.getuid()
        or root_info.st_uid != os.getuid()
    ):
        raise ContainerRuntimeRefusal("workspace_outside_allowed_root")
    return workspace


def _read_regular(path: Path, root: Path, limit: int) -> bytes:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ContainerRuntimeRefusal("workspace_scan_escape") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ContainerRuntimeRefusal("workspace_scan_escape")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ContainerRuntimeRefusal("workspace_contains_unsupported_entry")
    if info.st_size > limit:
        raise ContainerRuntimeRefusal("workspace_file_too_large")
    return path.read_bytes()


def workspace_manifest(
    workspace: Path,
    *,
    max_files: int = 4096,
    max_file_bytes: int = 16 * 1024 * 1024,
    max_total_bytes: int = 128 * 1024 * 1024,
) -> dict[str, Any]:
    """Fingerprint regular files without following links or special entries."""

    root = workspace.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    total = 0
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted(directories):
            if (current_path / name).is_symlink():
                raise ContainerRuntimeRefusal("workspace_contains_symlink")
        for name in sorted(files):
            path = current_path / name
            raw = _read_regular(path, root, max_file_bytes)
            total += len(raw)
            if len(rows) >= max_files or total > max_total_bytes:
                raise ContainerRuntimeRefusal("workspace_limits_exceeded")
            relative = path.relative_to(root).as_posix()
            rows.append(
                {
                    "path": relative,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "mode": stat.S_IMODE(path.stat().st_mode),
                }
            )
    payload = {"schema_version": 1, "files": rows, "total_bytes": total}
    payload["content_sha256"] = stable_sha256(payload)
    return payload


def workspace_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    left = {row["path"]: row for row in before.get("files", [])}
    right = {row["path"]: row for row in after.get("files", [])}
    return {
        "created": sorted(set(right) - set(left)),
        "deleted": sorted(set(left) - set(right)),
        "modified": sorted(
            path
            for path in set(left) & set(right)
            if left[path].get("sha256") != right[path].get("sha256")
            or left[path].get("mode") != right[path].get("mode")
        ),
    }


def _bounded_text(value: bytes, workspace: Path) -> tuple[str, bool]:
    clipped = len(value) > _MAX_CAPTURE_BYTES
    text = value[:_MAX_CAPTURE_BYTES].decode("utf-8", errors="replace")
    text = text.replace(str(workspace), "<workspace>")
    return text, clipped


def _command(language: str, code: str) -> list[str]:
    if language not in _LANGUAGES:
        raise ContainerRuntimeRefusal("language_not_allowed")
    if not isinstance(code, str) or not code.strip():
        raise ContainerRuntimeRefusal("code_missing")
    encoded = code.encode("utf-8")
    if len(encoded) > _MAX_CODE_BYTES or "\x00" in code:
        raise ContainerRuntimeRefusal("code_invalid")
    if language == "python":
        return ["python", "-I", "-c", code]
    return ["/bin/bash", "--noprofile", "--norc", "-c", code]


@dataclass(frozen=True)
class PinnedContainerRuntime:
    allowed_root: Path
    image: str
    binary: Path = Path("docker")
    timeout_seconds: int = 120
    memory: str = "4g"
    cpus: str = "2"
    pids_limit: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(self, "image", validate_image_reference(self.image))
        object.__setattr__(self, "binary", _regular_binary(self.binary))
        if not 1 <= self.timeout_seconds <= 600:
            raise ContainerRuntimeRefusal("timeout_invalid")
        if not 16 <= self.pids_limit <= 4096:
            raise ContainerRuntimeRefusal("pids_limit_invalid")
        original_root = self.allowed_root
        root = original_root.resolve(strict=True)
        if (
            original_root != root
            or original_root.is_symlink()
            or not root.is_dir()
            or root == Path("/")
        ):
            raise ContainerRuntimeRefusal("allowed_root_invalid")
        object.__setattr__(self, "allowed_root", root)

    @classmethod
    def from_environment(
        cls,
        allowed_root: Path,
        *,
        timeout_seconds: int = 120,
        binary: str | Path = "docker",
    ) -> "PinnedContainerRuntime":
        return cls(
            allowed_root=allowed_root,
            image=validate_image_reference(os.environ.get(SANDBOX_IMAGE_ENV)),
            binary=Path(binary),
            timeout_seconds=timeout_seconds,
        )

    def execute(self, workspace: Path, language: str, code: str) -> dict[str, Any]:
        workspace = _real_workspace(workspace, allowed_root=self.allowed_root)
        before = workspace_manifest(workspace)
        inner = _command(language, code)
        uid, gid = os.getuid(), os.getgid()
        command: Sequence[str] = [
            str(self.binary),
            "run",
            "--rm",
            "--network=none",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--read-only",
            f"--pids-limit={self.pids_limit}",
            f"--memory={self.memory}",
            f"--cpus={self.cpus}",
            f"--user={uid}:{gid}",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=256m",
            "--mount",
            f"type=bind,src={workspace},dst=/workspace,rw",
            "--workdir=/workspace",
            "--env=HOME=/tmp",
            "--env=PYTHONDONTWRITEBYTECODE=1",
            self.image,
            *inner,
        ]
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=False,
            )
            exit_code = int(completed.returncode)
            stdout_raw, stderr_raw = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout_raw = exc.stdout or b""
            stderr_raw = exc.stderr or b""
        after = workspace_manifest(workspace)
        stdout, stdout_clipped = _bounded_text(stdout_raw, workspace)
        stderr, stderr_clipped = _bounded_text(stderr_raw, workspace)
        receipt = {
            "schema_version": 1,
            "language": language,
            "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "image_sha256": self.image.rsplit("@sha256:", 1)[1],
            "runtime_attestation_sha256": runtime_attestation_sha256(self.image),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_clipped": stdout_clipped,
            "stderr_clipped": stderr_clipped,
            "pre_workspace_sha256": before["content_sha256"],
            "post_workspace_sha256": after["content_sha256"],
            "workspace_delta": workspace_delta(before, after),
        }
        receipt["receipt_sha256"] = stable_sha256(receipt)
        return receipt


def execute_container_lightweight(
    *,
    workspace: Path,
    allowed_root: Path,
    language: str,
    code: str,
    image: str | None = None,
    timeout_seconds: int = 120,
    binary: str | Path = "docker",
) -> dict[str, Any]:
    runtime = PinnedContainerRuntime(
        allowed_root=allowed_root,
        image=validate_image_reference(image or os.environ.get(SANDBOX_IMAGE_ENV)),
        binary=Path(binary),
        timeout_seconds=timeout_seconds,
    )
    return runtime.execute(workspace, language, code)


__all__ = [
    "ContainerCleanupRefusal",
    "ContainerRuntimeRefusal",
    "PinnedContainerRuntime",
    "SANDBOX_IMAGE_ENV",
    "execute_container_lightweight",
    "runtime_attestation_sha256",
    "stable_sha256",
    "validate_image_reference",
    "workspace_delta",
    "workspace_manifest",
]

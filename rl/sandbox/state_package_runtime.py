"""Sealed inline snapshot validation and safe workspace materialization."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MODES = {0o600, 0o640, 0o644, 0o660, 0o664, 0o700, 0o750, 0o755, 0o770, 0o775}


class SnapshotRefusal(RuntimeError):
    """A snapshot or destination does not meet the safe subset."""


@dataclass(frozen=True)
class SnapshotLimits:
    max_entries: int = 10_000
    max_files: int = 8_000
    max_directories: int = 2_000
    max_file_bytes: int = 64 * 1024 * 1024
    max_total_bytes: int = 512 * 1024 * 1024
    max_manifest_bytes: int = 768 * 1024 * 1024
    max_path_bytes: int = 512
    max_path_depth: int = 32

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in vars(self).values()):
            raise ValueError("snapshot_limits_invalid")
        if self.max_files > self.max_entries or self.max_directories > self.max_entries or self.max_file_bytes > self.max_total_bytes:
            raise ValueError("snapshot_limits_inconsistent")


@dataclass(frozen=True)
class SnapshotEntry:
    path: str
    kind: str
    mode: int
    content: bytes = b""

    @property
    def sha256(self) -> str | None:
        return hashlib.sha256(self.content).hexdigest() if self.kind == "file" else None

    def inventory(self) -> dict[str, Any]:
        value: dict[str, Any] = {"path": self.path, "type": self.kind, "mode": f"0{self.mode:o}"}
        if self.kind == "file":
            value.update({"size_bytes": len(self.content), "sha256": self.sha256})
        return value

    def manifest(self) -> dict[str, Any]:
        value = self.inventory()
        if self.kind == "file":
            value["content_base64"] = base64.b64encode(self.content).decode("ascii")
        return value


@dataclass(frozen=True)
class ValidatedSnapshot:
    entries: tuple[SnapshotEntry, ...]
    content_sha256: str
    manifest_sha256: str
    file_count: int
    directory_count: int
    size_bytes: int

    @property
    def manifest(self) -> dict[str, Any]:
        return _manifest(self.entries, self.content_sha256)


def _stable(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise SnapshotRefusal("snapshot_not_stable_json") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable(value).encode("utf-8")).hexdigest()


def _safe_relative(value: Any, limits: SnapshotLimits) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value or "\\" in value:
        raise SnapshotRefusal("snapshot_path_invalid")
    path = PurePosixPath(value)
    parts = value.split("/")
    if path.is_absolute() or value.endswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise SnapshotRefusal("snapshot_path_invalid")
    if len(parts) > limits.max_path_depth or len(value.encode("utf-8")) > limits.max_path_bytes:
        raise SnapshotRefusal("snapshot_path_too_large")
    return value


def _mode(value: Any) -> int:
    if not isinstance(value, str) or re.fullmatch(r"0[0-7]{3}", value) is None:
        raise SnapshotRefusal("snapshot_mode_invalid")
    result = int(value, 8)
    if result not in _MODES:
        raise SnapshotRefusal("snapshot_mode_unsupported")
    return result


def _entries(rows: Any, limits: SnapshotLimits) -> tuple[SnapshotEntry, ...]:
    if not isinstance(rows, list) or len(rows) > limits.max_entries:
        raise SnapshotRefusal("snapshot_entries_invalid")
    entries: list[SnapshotEntry] = []
    seen: set[str] = set()
    folded: set[str] = set()
    total = files = directories = 0
    for row in rows:
        if not isinstance(row, Mapping) or row.get("type") not in {"file", "directory"}:
            raise SnapshotRefusal("snapshot_entry_invalid")
        kind = str(row["type"])
        expected = {"path", "type", "mode"} | ({"size_bytes", "sha256", "content_base64"} if kind == "file" else set())
        if set(row) != expected:
            raise SnapshotRefusal("snapshot_entry_fields_invalid")
        path = _safe_relative(row.get("path"), limits)
        if path in seen or path.casefold() in folded:
            raise SnapshotRefusal("snapshot_path_duplicate")
        seen.add(path)
        folded.add(path.casefold())
        mode = _mode(row.get("mode"))
        if kind == "directory":
            directories += 1
            entry = SnapshotEntry(path, kind, mode)
        else:
            files += 1
            encoded = row.get("content_base64")
            try:
                content = base64.b64decode(encoded, validate=True)
            except Exception as exc:
                raise SnapshotRefusal("snapshot_file_base64_invalid") from exc
            if not isinstance(encoded, str) or base64.b64encode(content).decode("ascii") != encoded:
                raise SnapshotRefusal("snapshot_file_base64_not_normalized")
            if row.get("size_bytes") != len(content) or row.get("sha256") != hashlib.sha256(content).hexdigest():
                raise SnapshotRefusal("snapshot_file_digest_invalid")
            if len(content) > limits.max_file_bytes:
                raise SnapshotRefusal("snapshot_file_too_large")
            total += len(content)
            entry = SnapshotEntry(path, kind, mode, content)
        entries.append(entry)
    if files > limits.max_files or directories > limits.max_directories or total > limits.max_total_bytes:
        raise SnapshotRefusal("snapshot_limits_exceeded")
    by_path = {entry.path: entry for entry in entries}
    for entry in entries:
        parts = entry.path.split("/")
        for index in range(1, len(parts)):
            parent = "/".join(parts[:index])
            if parent not in by_path or by_path[parent].kind != "directory":
                raise SnapshotRefusal("snapshot_parent_directory_missing")
    return tuple(sorted(entries, key=lambda item: item.path.encode("utf-8")))


def _basis(entries: Sequence[SnapshotEntry]) -> dict[str, Any]:
    return {"schema_version": 1, "entries": [entry.inventory() for entry in entries]}


def _manifest(entries: Sequence[SnapshotEntry], digest: str) -> dict[str, Any]:
    return {"schema_version": 1, "immutable": True, "snapshot_sha256": digest, "entries": [entry.manifest() for entry in entries]}


def seal_snapshot_manifest(entries: Sequence[Mapping[str, Any]], *, limits: SnapshotLimits = SnapshotLimits()) -> dict[str, Any]:
    parsed = _entries(list(entries), limits)
    digest = _digest(_basis(parsed))
    manifest = _manifest(parsed, digest)
    if len(_stable(manifest).encode("utf-8")) > limits.max_manifest_bytes:
        raise SnapshotRefusal("snapshot_manifest_too_large")
    return manifest


def validate_snapshot_manifest(manifest: Mapping[str, Any], *, limits: SnapshotLimits = SnapshotLimits()) -> ValidatedSnapshot:
    if not isinstance(manifest, Mapping) or set(manifest) != {"schema_version", "immutable", "snapshot_sha256", "entries"}:
        raise SnapshotRefusal("snapshot_manifest_invalid")
    if manifest.get("schema_version") != 1 or manifest.get("immutable") is not True:
        raise SnapshotRefusal("snapshot_manifest_invalid")
    parsed = _entries(manifest.get("entries"), limits)
    digest = _digest(_basis(parsed))
    if manifest.get("snapshot_sha256") != digest or _SHA256.fullmatch(digest) is None:
        raise SnapshotRefusal("snapshot_digest_invalid")
    normalized = _stable(_manifest(parsed, digest))
    if len(normalized.encode("utf-8")) > limits.max_manifest_bytes:
        raise SnapshotRefusal("snapshot_manifest_too_large")
    return ValidatedSnapshot(
        entries=parsed,
        content_sha256=digest,
        manifest_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        file_count=sum(item.kind == "file" for item in parsed),
        directory_count=sum(item.kind == "directory" for item in parsed),
        size_bytes=sum(len(item.content) for item in parsed),
    )


def _safe_root(root: Path) -> Path:
    if not root.is_absolute():
        raise SnapshotRefusal("workspace_root_not_absolute")
    try:
        resolved = root.resolve(strict=True)
        info = root.lstat()
    except OSError as exc:
        raise SnapshotRefusal("workspace_root_invalid") from exc
    if resolved != root or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or root == Path("/") or not os.access(root, os.W_OK | os.X_OK):
        raise SnapshotRefusal("workspace_root_invalid")
    return root


def materialize_snapshot(snapshot: ValidatedSnapshot, safe_workspace_root: Path, *, prefix: str = "bat_reset_") -> Path:
    if not isinstance(snapshot, ValidatedSnapshot) or re.fullmatch(r"[A-Za-z0-9_.-]{1,48}", prefix) is None:
        raise SnapshotRefusal("snapshot_materialization_invalid")
    root = _safe_root(Path(safe_workspace_root))
    workspace = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    os.chmod(workspace, 0o700)
    try:
        for entry in sorted((item for item in snapshot.entries if item.kind == "directory"), key=lambda item: (item.path.count("/"), item.path)):
            destination = workspace.joinpath(*entry.path.split("/"))
            os.mkdir(destination, entry.mode)
        for entry in (item for item in snapshot.entries if item.kind == "file"):
            destination = workspace.joinpath(*entry.path.split("/"))
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(destination, flags, entry.mode)
            try:
                pending = memoryview(entry.content)
                while pending:
                    written = os.write(descriptor, pending)
                    if written <= 0:
                        raise SnapshotRefusal("snapshot_file_short_write")
                    pending = pending[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return workspace
    except BaseException:
        import shutil

        shutil.rmtree(workspace, ignore_errors=True)
        raise


def _scan_entries(workspace: Path, limits: SnapshotLimits) -> tuple[SnapshotEntry, ...]:
    rows: list[dict[str, Any]] = []
    root = workspace.resolve(strict=True)
    for current, directories, files in os.walk(root, followlinks=False):
        base = Path(current)
        for name in sorted(directories):
            path = base / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise SnapshotRefusal("workspace_symlink_forbidden")
            rows.append({"path": path.relative_to(root).as_posix(), "type": "directory", "mode": f"0{stat.S_IMODE(info.st_mode):o}"})
        for name in sorted(files):
            path = base / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limits.max_file_bytes:
                raise SnapshotRefusal("workspace_file_invalid")
            content = path.read_bytes()
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "type": "file",
                "mode": f"0{stat.S_IMODE(info.st_mode):o}",
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "content_base64": base64.b64encode(content).decode("ascii"),
            })
    return _entries(rows, limits)


def fingerprint_workspace(workspace: Path, *, limits: SnapshotLimits = SnapshotLimits()) -> str:
    path = Path(workspace)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SnapshotRefusal("workspace_invalid")
    return _digest(_basis(_scan_entries(path, limits)))


__all__ = [
    "SnapshotEntry",
    "SnapshotLimits",
    "SnapshotRefusal",
    "ValidatedSnapshot",
    "fingerprint_workspace",
    "materialize_snapshot",
    "seal_snapshot_manifest",
    "validate_snapshot_manifest",
]

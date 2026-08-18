#!/usr/bin/env python3
"""Write an immutable identity binding for one BaT gate evaluation arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    fingerprint_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    fingerprint_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if fingerprint_before != fingerprint_after:
        raise ValueError(f"model file changed while hashing: {path}")
    return digest.hexdigest()


def _model_files(model: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(model.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"model export must not contain symlinks: {path}")
        if path.is_file():
            files.append(path.resolve(strict=True))
    if not files:
        raise ValueError(f"model export contains no files: {model}")
    return files


def _weight_paths(model: Path) -> tuple[list[Path], Path | None]:
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = model / name
        if index.is_file():
            payload = json.loads(index.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"model weight index is malformed: {index}")
            files = sorted({model / str(value) for value in weight_map.values()})
            return files, index
    files = sorted(model.glob("model*.safetensors"))
    if not files:
        files = sorted(model.glob("pytorch_model*.bin"))
    return files, None


def build_binding(
    model_path: Path,
    *,
    experiment_name: str,
    gate_step: int,
    arm: str,
) -> dict[str, Any]:
    if model_path.is_symlink():
        raise ValueError(f"model path must not be a symlink: {model_path}")
    model = model_path.resolve(strict=True)
    if not model.is_dir():
        raise ValueError(f"model path is not a directory: {model}")
    config = model / "config.json"
    if not config.is_file():
        raise ValueError(f"model config is missing: {config}")
    weights, index = _weight_paths(model)
    resolved_weights: list[Path] = []
    for path in weights:
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(model)
        except ValueError as exc:
            raise ValueError(f"model weight escapes model directory: {path}") from exc
        if path.is_symlink() or not resolved.is_file() or resolved.stat().st_size <= 0:
            raise ValueError(f"model weight is not a non-empty regular file: {path}")
        resolved_weights.append(resolved)
    weights = sorted(set(resolved_weights))
    if not weights:
        raise ValueError(f"model weights are missing or incomplete: {model}")
    all_files = _model_files(model)
    file_rows = []
    file_sha: dict[Path, str] = {}
    for path in all_files:
        digest = _sha256(path)
        file_sha[path] = digest
        file_rows.append(
            {
                "path": str(path.relative_to(model)),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    weight_rows = [
        {
            "name": str(path.relative_to(model)),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha[path],
        }
        for path in weights
    ]
    content_sha = hashlib.sha256(
        json.dumps(file_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    identity_basis = {
        "model_content_sha256": content_sha,
        "config_sha256": file_sha[config.resolve(strict=True)],
        "weight_index_sha256": (
            file_sha[index.resolve(strict=True)] if index is not None else None
        ),
        "files": file_rows,
        "weights": weight_rows,
    }
    identity_sha = hashlib.sha256(
        json.dumps(identity_basis, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "artifact_kind": "bat_gate_model_binding",
        "schema_version": 2,
        "experiment_name": experiment_name,
        "gate_step": gate_step,
        "arm": arm,
        **identity_basis,
        "model_identity_sha256": identity_sha,
    }


def write_immutable(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == rendered:
            return
        raise FileExistsError(f"gate model binding differs from requested model: {path}")
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--gate-step", type=int, required=True)
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.gate_step < 0:
        parser.error("--gate-step must be non-negative")
    binding = build_binding(
        args.model_path,
        experiment_name=args.experiment_name,
        gate_step=args.gate_step,
        arm=args.arm,
    )
    write_immutable(args.output, binding)
    print(json.dumps({"output": str(args.output), "model_identity_sha256": binding["model_identity_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

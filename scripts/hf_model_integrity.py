#!/usr/bin/env python3
"""Model-neutral integrity checks for an exported Hugging Face checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class ModelIntegrityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weight_files(model: Path) -> list[Path]:
    indexes = [model / "model.safetensors.index.json", model / "pytorch_model.bin.index.json"]
    for index in indexes:
        if index.is_file():
            payload = json.loads(index.read_text(encoding="utf-8"))
            mapping = payload.get("weight_map") if isinstance(payload, dict) else None
            if not isinstance(mapping, dict) or not mapping:
                raise ModelIntegrityError("weight_index_invalid")
            files = sorted({model / str(name) for name in mapping.values()})
            if any(path.parent != model or not path.is_file() or path.stat().st_size <= 0 for path in files):
                raise ModelIntegrityError("weight_shard_missing")
            return files
    files = sorted(model.glob("model*.safetensors")) or sorted(model.glob("pytorch_model*.bin"))
    if not files or any(path.stat().st_size <= 0 for path in files):
        raise ModelIntegrityError("model_weights_missing")
    return files


def validate_model_directory(model_dir: Path) -> dict[str, Any]:
    model = model_dir.resolve(strict=True)
    if not model.is_dir() or model_dir.is_symlink():
        raise ModelIntegrityError("model_directory_invalid")
    config_path = model / "config.json"
    if not config_path.is_file():
        raise ModelIntegrityError("model_config_missing")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("model_type"), str):
        raise ModelIntegrityError("model_config_invalid")
    weights = _weight_files(model)
    for path in model.rglob("*"):
        if path.is_symlink():
            raise ModelIntegrityError("model_symlink_forbidden")
    rows = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in weights
    ]
    content_sha256 = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "status": "passed",
        "model_type": config["model_type"],
        "config_sha256": _sha256(config_path),
        "weight_files": rows,
        "weight_content_sha256": content_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_model_directory(args.model), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

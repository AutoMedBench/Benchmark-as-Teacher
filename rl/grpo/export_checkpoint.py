#!/usr/bin/env python3
"""Export a VERL actor checkpoint into an evaluation-ready HF directory."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from scripts.hf_model_integrity import validate_model_directory


def resolve_actor(checkpoint: Path) -> Path:
    root = checkpoint.resolve(strict=True)
    actor = root / "actor" if (root / "actor").is_dir() else root
    if actor.is_symlink() or not actor.is_dir():
        raise ValueError("actor_checkpoint_invalid")
    return actor


def merge_command(actor: Path, target: Path, *, python: str = sys.executable) -> list[str]:
    return [python, "-m", "verl.model_merger", "merge", "--backend", "fsdp", "--local_dir", str(actor), "--target_dir", str(target)]


def _copy_model_metadata(base: Path, target: Path) -> None:
    names = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    }
    for path in base.iterdir():
        if path.is_file() and (path.name in names or path.suffix == ".py") and not (target / path.name).exists():
            shutil.copy2(path, target / path.name)


def export_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    base_model: Path | None = None,
    python: str = sys.executable,
    runner: Any = subprocess.run,
) -> dict[str, object]:
    actor = resolve_actor(checkpoint)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        has_hf = (actor / "config.json").is_file() and bool(list(actor.glob("model*.safetensors")) or list(actor.glob("pytorch_model*.bin")))
        if has_hf:
            shutil.rmtree(temporary)
            shutil.copytree(actor, temporary, symlinks=False)
        else:
            completed = runner(merge_command(actor, temporary, python=python), check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"checkpoint_merge_failed:{completed.returncode}")
        if base_model is not None:
            _copy_model_metadata(base_model.resolve(strict=True), temporary)
        receipt = validate_model_directory(temporary)
        (temporary / "bat-export.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output, base_model=args.base_model, python=args.python), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

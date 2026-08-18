"""Load a tokenizer with an explicit, usable chat template."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_chat_tokenizer(model: str | Path, *, trust_remote_code: bool = False) -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError("transformers_not_installed") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=trust_remote_code)
    if not isinstance(getattr(tokenizer, "chat_template", None), str) or not tokenizer.chat_template.strip():
        raise RuntimeError("tokenizer_chat_template_missing")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer_padding_token_missing")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


__all__ = ["load_chat_tokenizer"]

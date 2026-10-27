"""Staging helpers for persistent MNN LLM worker configurations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def stage_model_config(
    base_config: Path,
    model_dir: Path,
    stage_dir: Path,
    *,
    kvcache_mmap: bool | None = True,
) -> dict[str, Any]:
    """Stage model symlinks beside a copied config for MNN path resolution."""
    data = json.loads(base_config.read_text(encoding="utf-8"))
    stage_dir.mkdir(parents=True, exist_ok=True)
    for key in ("llm_model", "llm_weight", "tokenizer_file"):
        value = data.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"model config is missing required path {key!r}")
        source = Path(value) if os.path.isabs(value) else model_dir / value
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"model config path does not exist: {source}")
        link = stage_dir / source.name
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != source:
                raise FileExistsError(f"staged model path collision: {link}")
        else:
            link.symlink_to(source)
        data[key] = link.name
    companion = model_dir / "llm_config.json"
    if companion.is_file():
        link = stage_dir / companion.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(companion.resolve())
    if kvcache_mmap is not None:
        data["kvcache_mmap"] = bool(kvcache_mmap)
    return data


def write_backend_config(
    base: dict[str, Any],
    output_path: Path,
    *,
    backend_type: str,
    thread_num: int,
) -> Path:
    """Write one explicit CPU or OpenCL MNN worker config."""
    backend = backend_type.strip().lower()
    if backend not in {"cpu", "opencl"}:
        raise ValueError(f"unsupported MNN backend_type: {backend_type!r}")
    if thread_num < 1:
        raise ValueError("thread_num must be positive")
    config = dict(base)
    config["backend_type"] = backend
    config["thread_num"] = int(thread_num)
    config["sampler_type"] = "greedy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path

"""Narrow model-family contract shared by adapters and native dispatch."""

from __future__ import annotations

from enum import IntEnum


class ModelFamily(IntEnum):
    """Native runtime model-family ids used by the EdgeFlow C ABI."""

    QWEN25VL = 0
    FASTVLM = 1
    LLAVA15 = 2


def infer_model_family_from_id(model_id: str | None) -> ModelFamily:
    """Compatibility inference for callers that have not set model_family yet."""
    mid = (model_id or "").lower()
    if "llava-1.5" in mid or "llava15" in mid:
        return ModelFamily.LLAVA15
    if "fastvlm" in mid or "llava-fastvithd" in mid:
        return ModelFamily.FASTVLM
    return ModelFamily.QWEN25VL

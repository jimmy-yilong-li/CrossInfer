"""Typed visual payloads shared by RK3588 visual and text backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VisualPayload:
    """Visual embedding plus enough metadata to route it safely."""

    family: str
    embedding: np.ndarray
    model_path: str
    image_path: str
    latency_ms: float
    backend: str = "rknn_lite"

    @property
    def embedding_dim(self) -> int:
        arr = np.asarray(self.embedding)
        if arr.ndim == 0:
            raise ValueError("visual embedding must have at least one dimension")
        return int(arr.shape[-1])

    @property
    def token_count(self) -> int | None:
        arr = np.asarray(self.embedding)
        if arr.ndim == 3:
            return int(arr.shape[1])
        return None

    def to_json_dict(self) -> dict[str, Any]:
        arr = np.asarray(self.embedding)
        finite = bool(np.isfinite(arr).all())
        arr32 = arr.astype(np.float32, copy=False)
        finite_arr = arr32[np.isfinite(arr32)]
        if finite_arr.size:
            min_value = float(finite_arr.min())
            max_value = float(finite_arr.max())
            mean_value = float(finite_arr.mean())
            l2_norm = float(np.linalg.norm(finite_arr))
        else:
            min_value = None
            max_value = None
            mean_value = None
            l2_norm = None
        return {
            "family": self.family,
            "backend": self.backend,
            "model_path": self.model_path,
            "image_path": self.image_path,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite": finite,
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
            "l2_norm": l2_norm,
            "latency_ms": float(self.latency_ms),
            "embedding_dim": self.embedding_dim,
            "token_count": self.token_count,
        }

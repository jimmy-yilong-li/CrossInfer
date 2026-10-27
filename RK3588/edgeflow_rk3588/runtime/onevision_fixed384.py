"""Fixed-384 LLaVA-OneVision preprocessing and trained projector."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image


_MATH_ERF_UFUNC = np.frompyfunc(math.erf, 1, 1)


def _erf_float32(array: np.ndarray) -> np.ndarray:
    """Vectorized erf without making SciPy a board-runtime dependency."""
    try:
        from scipy.special import erf

        return np.asarray(erf(array), dtype=np.float32)
    except ImportError:
        return np.asarray(_MATH_ERF_UFUNC(array), dtype=np.float32)


def preprocess_siglip_fixed384(image_path: str | Path) -> tuple[np.ndarray, float]:
    """Return the existing single-crop SigLIP NCHW input and elapsed time."""
    start = perf_counter()
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize((384, 384), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = ((array - 0.5) / 0.5).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(array, dtype=np.float32), (perf_counter() - start) * 1e3


def pool_onevision_video_features(features: np.ndarray) -> np.ndarray:
    """Bilinearly resize a square projected grid to ceil-half, without newline."""
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("expected (tokens, dim) projected video features")
    side = math.isqrt(array.shape[0])
    if side == 0 or side * side != array.shape[0]:
        raise ValueError("video_pool2 requires a non-empty square token grid")

    pooled_side = (side + 1) // 2
    grid = array.reshape(side, side, array.shape[1])
    # Half-pixel coordinates match bilinear interpolation with align_corners=False.
    positions = (
        (np.arange(pooled_side, dtype=np.float32) + 0.5)
        * np.float32(side / pooled_side)
        - 0.5
    )
    lower = np.floor(positions).astype(np.intp)
    upper = np.minimum(lower + 1, side - 1)
    fraction = positions - lower.astype(np.float32)
    x_weight = fraction[None, :, None]
    y_weight = fraction[:, None, None]
    top = (
        grid[lower[:, None], lower[None, :]] * (1.0 - x_weight)
        + grid[lower[:, None], upper[None, :]] * x_weight
    )
    bottom = (
        grid[upper[:, None], lower[None, :]] * (1.0 - x_weight)
        + grid[upper[:, None], upper[None, :]] * x_weight
    )
    pooled = top * (1.0 - y_weight) + bottom * y_weight
    return np.ascontiguousarray(
        pooled.reshape(pooled_side * pooled_side, array.shape[1]), dtype=np.float32
    )


@dataclass(frozen=True)
class OneVisionProjector:
    """The trained OneVision two-layer GELU projector and newline token."""

    linear_1_weight: np.ndarray
    linear_1_bias: np.ndarray
    linear_2_weight: np.ndarray
    linear_2_bias: np.ndarray
    image_newline: np.ndarray | None

    @classmethod
    def from_npz(cls, path: str | Path) -> "OneVisionProjector":
        with np.load(path) as data:
            required = (
                "linear_1_weight",
                "linear_1_bias",
                "linear_2_weight",
                "linear_2_bias",
            )
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"projector archive is missing keys: {missing}")
            newline = (
                np.asarray(data["image_newline"], dtype=np.float32).reshape(1, -1)
                if "image_newline" in data
                else None
            )
            return cls(
                linear_1_weight=np.asarray(data["linear_1_weight"], dtype=np.float32),
                linear_1_bias=np.asarray(data["linear_1_bias"], dtype=np.float32),
                linear_2_weight=np.asarray(data["linear_2_weight"], dtype=np.float32),
                linear_2_bias=np.asarray(data["linear_2_bias"], dtype=np.float32),
                image_newline=newline,
            )

    def project(
        self, features: np.ndarray, *, visual_token_mode: str = "image"
    ) -> tuple[np.ndarray, float]:
        """Project `(1, tokens, dim)` or `(tokens, dim)` SigLIP features."""
        start = perf_counter()
        if visual_token_mode not in ("image", "video_pool2"):
            raise ValueError(f"unsupported visual_token_mode: {visual_token_mode!r}")
        array = np.asarray(features, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2 or array.shape[1] != self.linear_1_weight.shape[1]:
            raise ValueError(
                "unexpected SigLIP feature shape: "
                f"got {array.shape}, expected (*, {self.linear_1_weight.shape[1]})"
            )
        hidden = array @ self.linear_1_weight.T + self.linear_1_bias
        hidden = 0.5 * hidden * (1.0 + _erf_float32(hidden / np.sqrt(2.0)))
        projected = hidden @ self.linear_2_weight.T + self.linear_2_bias
        projected = np.asarray(projected, dtype=np.float32)
        if visual_token_mode == "video_pool2":
            projected = pool_onevision_video_features(projected)
        if self.image_newline is not None:
            if self.image_newline.shape[1] != projected.shape[1]:
                raise ValueError("image_newline width does not match projector output")
            projected = np.vstack((projected, self.image_newline))
        if not np.isfinite(projected).all():
            raise ValueError("projector produced non-finite values")
        return np.ascontiguousarray(projected), (perf_counter() - start) * 1e3

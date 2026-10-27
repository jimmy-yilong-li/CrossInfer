from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from edgeflow_split.contracts import ImageEmbedsBatch


def _looks_like_mlx_array(value) -> bool:
    try:
        tensor_module = getattr(type(value), "__module__", "")
        return isinstance(tensor_module, str) and tensor_module.startswith("mlx.core")
    except Exception:
        return False


def _require_perfect_square(value: int, *, name: str) -> int:
    side = math.isqrt(int(value))
    if side * side != int(value):
        raise ValueError(f"{name} must be a perfect square, got {value}")
    return side


def _infer_token_grid(batch: ImageEmbedsBatch) -> tuple[int, int, int, int]:
    t, h, w = (int(v) for v in batch.image_grid_thw)
    if t <= 0 or h <= 0 or w <= 0:
        raise ValueError("image_grid_thw must be positive")
    if batch.num_visual_tokens <= 0:
        raise ValueError("num_visual_tokens must be positive")
    if batch.num_visual_tokens % t != 0:
        raise ValueError("cannot infer per-frame token grid from num_visual_tokens and image_grid_thw")

    tokens_per_frame = batch.num_visual_tokens // t
    merge_unit = (h * w) // tokens_per_frame if tokens_per_frame else 0
    if tokens_per_frame == 0 or h * w % tokens_per_frame != 0:
        raise ValueError("cannot infer current token grid from image_grid_thw and num_visual_tokens")

    spatial_merge_size = _require_perfect_square(merge_unit, name="inferred spatial merge unit")
    if h % spatial_merge_size != 0 or w % spatial_merge_size != 0:
        raise ValueError("cannot infer token grid dimensions from image_grid_thw")

    token_h = h // spatial_merge_size
    token_w = w // spatial_merge_size
    if token_h * token_w != tokens_per_frame:
        raise ValueError("cannot infer token grid dimensions from image_grid_thw and num_visual_tokens")
    return t, token_h, token_w, spatial_merge_size


def _pool_numpy(array, *, t: int, token_h: int, token_w: int, reduce_side: int):
    hidden = int(array.shape[1])
    reshaped = np.asarray(array).reshape(t, token_h, token_w, hidden)
    new_h = token_h // reduce_side
    new_w = token_w // reduce_side
    pooled = reshaped.reshape(t, new_h, reduce_side, new_w, reduce_side, hidden).mean(axis=(2, 4))
    return pooled.reshape(t * new_h * new_w, hidden).astype(array.dtype, copy=False)


def _pool_torch(array, *, t: int, token_h: int, token_w: int, reduce_side: int):
    import torch

    hidden = int(array.shape[1])
    reshaped = array.reshape(t, token_h, token_w, hidden)
    new_h = token_h // reduce_side
    new_w = token_w // reduce_side
    pooled = reshaped.reshape(t, new_h, reduce_side, new_w, reduce_side, hidden).to(torch.float32).mean(dim=(2, 4))
    return pooled.reshape(t * new_h * new_w, hidden).to(array.dtype)


def _pool_mlx(array, *, t: int, token_h: int, token_w: int, reduce_side: int):
    import mlx.core as mx

    hidden = int(array.shape[1])
    reshaped = mx.reshape(array, (t, token_h, token_w, hidden))
    new_h = token_h // reduce_side
    new_w = token_w // reduce_side
    reshaped_6d = mx.reshape(reshaped, (t, new_h, reduce_side, new_w, reduce_side, hidden))
    pooled = mx.mean(reshaped_6d.astype(mx.float32), axis=(2, 4))
    return mx.reshape(pooled, (t * new_h * new_w, hidden)).astype(array.dtype)


def spatial_reduce_batch(batch: ImageEmbedsBatch, *, factor: int = 4) -> ImageEmbedsBatch:
    """Reduce per-frame visual tokens by square average pooling.

    `factor` is the token-count reduction ratio and must be a perfect square.
    Example: factor=4 performs 2x2 average pooling on the current token grid.
    """
    factor = int(factor)
    if factor <= 0:
        raise ValueError("factor must be positive")
    if factor == 1:
        return batch

    reduce_side = _require_perfect_square(factor, name="factor")
    t, token_h, token_w, _spatial_merge_size = _infer_token_grid(batch)
    if token_h % reduce_side != 0 or token_w % reduce_side != 0:
        raise ValueError("reduction factor does not evenly divide the current token grid")

    image_embeds = batch.image_embeds
    if _looks_like_mlx_array(image_embeds):
        import mlx.core as mx

        if not isinstance(image_embeds, mx.array):
            raise TypeError("Expected mlx array when module indicates mlx.core")

        reduced_embeds = _pool_mlx(
            image_embeds,
            t=t,
            token_h=token_h,
            token_w=token_w,
            reduce_side=reduce_side,
        )
    else:
        try:
            import torch

            if torch.is_tensor(image_embeds):
                reduced_embeds = _pool_torch(
                    image_embeds,
                    t=t,
                    token_h=token_h,
                    token_w=token_w,
                    reduce_side=reduce_side,
                )
            else:
                reduced_embeds = _pool_numpy(
                    image_embeds,
                    t=t,
                    token_h=token_h,
                    token_w=token_w,
                    reduce_side=reduce_side,
                )
        except ImportError:
            reduced_embeds = _pool_numpy(
                image_embeds,
                t=t,
                token_h=token_h,
                token_w=token_w,
                reduce_side=reduce_side,
            )

    original_t, original_h, original_w = batch.image_grid_thw
    reduced_grid = (int(original_t), int(original_h) // reduce_side, int(original_w) // reduce_side)
    return ImageEmbedsBatch(
        image_embeds=reduced_embeds,
        image_grid_thw=reduced_grid,
        num_visual_tokens=int(reduced_embeds.shape[0]),
        hidden_size=int(reduced_embeds.shape[1]),
        dtype=batch.dtype,
        source_image_size=batch.source_image_size,
        model_revision=batch.model_revision,
    )


def spatial_reduce_batch_to_max_tokens_per_frame(
    batch: ImageEmbedsBatch,
    *,
    max_tokens_per_frame: int,
) -> ImageEmbedsBatch:
    """Reduce a batch to the largest supported per-frame token count <= target.

    The current contract requires one shared `(h, w)` token grid across all frames
    in the batch. This helper therefore picks a single square pooling factor whose
    reduced token grid evenly divides the current token grid and stays within the
    requested per-frame budget.
    """
    max_tokens_per_frame = int(max_tokens_per_frame)
    if max_tokens_per_frame <= 0:
        raise ValueError("max_tokens_per_frame must be positive")

    t, token_h, token_w, _spatial_merge_size = _infer_token_grid(batch)
    current_tokens_per_frame = (batch.num_visual_tokens // t) if t else 0
    if current_tokens_per_frame <= max_tokens_per_frame:
        return batch

    best_reduce_side = None
    best_tokens_per_frame = 0
    for reduce_side in range(1, min(token_h, token_w) + 1):
        if token_h % reduce_side != 0 or token_w % reduce_side != 0:
            continue
        candidate_tokens = (token_h // reduce_side) * (token_w // reduce_side)
        if candidate_tokens <= max_tokens_per_frame and candidate_tokens > best_tokens_per_frame:
            best_reduce_side = reduce_side
            best_tokens_per_frame = candidate_tokens

    if best_reduce_side is None:
        raise ValueError(
            "cannot satisfy max_tokens_per_frame with square spatial reduction "
            f"(current grid={token_h}x{token_w}, target={max_tokens_per_frame})"
        )

    factor = best_reduce_side * best_reduce_side
    return spatial_reduce_batch(batch, factor=factor)


def make_spatial_reduction_hook(*, factor: int = 4) -> Callable[[ImageEmbedsBatch], ImageEmbedsBatch]:
    return lambda batch: spatial_reduce_batch(batch, factor=factor)

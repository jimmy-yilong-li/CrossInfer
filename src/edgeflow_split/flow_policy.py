from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from edgeflow_split.contracts import ImageEmbedsBatch, TabmAdmissionDecision
from edgeflow_split.shared_tensor import HandleRingBuffer, SharedTensorHandle
from edgeflow_split.token_reduction import spatial_reduce_batch_to_max_tokens_per_frame


AllocationStrategy = Literal["uniform_reduce"]
DropPolicy = Literal["oldest_first"]


@dataclass(frozen=True)
class TokenFlowPolicy:
    """Token-budget policy for consuming queued visual frames.

    The current `ImageEmbedsBatch` contract only supports one shared `(t, h, w)`
    token grid across all frames in a batch. The policy therefore uses a
    uniform per-frame spatial budget plus oldest-first drop.
    """

    max_visual_tokens: int
    allocation_strategy: AllocationStrategy = "uniform_reduce"
    drop_policy: DropPolicy = "oldest_first"
    min_tokens_per_frame: int = 1

    def __post_init__(self) -> None:
        if int(self.max_visual_tokens) <= 0:
            raise ValueError("max_visual_tokens must be positive")
        if int(self.min_tokens_per_frame) <= 0:
            raise ValueError("min_tokens_per_frame must be positive")


def _concatenate_batches(batches: list[ImageEmbedsBatch]) -> ImageEmbedsBatch:
    if not batches:
        raise ValueError("batches must be non-empty")
    if len(batches) == 1:
        return batches[0]

    first = batches[0]
    hidden_size = first.hidden_size
    dtype = first.dtype
    grid_h = int(first.image_grid_thw[1])
    grid_w = int(first.image_grid_thw[2])

    for batch in batches[1:]:
        if batch.hidden_size != hidden_size:
            raise ValueError("all batches must share hidden_size")
        if batch.dtype != dtype:
            raise ValueError("all batches must share dtype")
        if tuple(batch.image_grid_thw[1:]) != (grid_h, grid_w):
            raise ValueError("all batches must share reduced spatial grid")

    embeds = [batch.image_embeds for batch in batches]
    first_module = getattr(type(first.image_embeds), "__module__", "")
    if isinstance(first_module, str) and first_module.startswith("mlx.core"):
        import mlx.core as mx

        if isinstance(first.image_embeds, mx.array):
            merged_embeds = mx.concatenate(embeds, axis=0)
        else:
            raise TypeError("Expected mlx array when module indicates mlx.core")
    else:
        try:
            import torch

            if torch.is_tensor(first.image_embeds):
                merged_embeds = torch.cat(embeds, dim=0)
            else:
                raise TypeError
        except (ImportError, TypeError):
            import numpy as np

            merged_embeds = np.concatenate(embeds, axis=0)

    total_t = sum(int(batch.image_grid_thw[0]) for batch in batches)
    return ImageEmbedsBatch(
        image_embeds=merged_embeds,
        image_grid_thw=(total_t, grid_h, grid_w),
        num_visual_tokens=int(merged_embeds.shape[0]),
        hidden_size=hidden_size,
        dtype=dtype,
        source_image_size=first.source_image_size,
        model_revision=first.model_revision,
    )


def pop_batch_with_budget(
    buffer: HandleRingBuffer,
    *,
    policy: TokenFlowPolicy,
) -> ImageEmbedsBatch | None:
    """Consume queued handles into one batch under a fixed visual-token budget."""
    batch, _ = pop_batch_with_budget_with_metrics(buffer, policy=policy)
    return batch


def pop_batch_with_budget_with_metrics(
    buffer: HandleRingBuffer,
    *,
    policy: TokenFlowPolicy,
) -> tuple[ImageEmbedsBatch | None, TabmAdmissionDecision]:
    """Consume queued handles into one batch with an explicit admission decision."""
    handles = buffer.drain()
    if not handles:
        return (
            None,
            TabmAdmissionDecision(
                policy_applied=True,
                policy="drop_only",
                reason="insufficient_budget",
                input_handles=0,
                output_handles=0,
                input_tokens=0,
                output_tokens=0,
            ),
        )

    input_handles = len(handles)
    input_tokens = sum(int(handle.token_count) for handle in handles)
    retained = list(handles)
    dropped_before_reduction: list[SharedTensorHandle] = []

    if policy.drop_policy != "oldest_first":
        raise ValueError(f"unsupported drop_policy: {policy.drop_policy}")
    if policy.allocation_strategy != "uniform_reduce":
        raise ValueError(f"unsupported allocation_strategy: {policy.allocation_strategy}")

    while retained and len(retained) * policy.min_tokens_per_frame > policy.max_visual_tokens:
        dropped_before_reduction.append(retained.pop(0))

    if not retained:
        for handle in dropped_before_reduction:
            handle.release()
        return (
            None,
            TabmAdmissionDecision(
                policy_applied=True,
                policy="drop_only",
                reason="insufficient_budget",
                input_handles=input_handles,
                output_handles=0,
                input_tokens=input_tokens,
                output_tokens=0,
                dropped_handle_ids=tuple(
                    int(handle.handle_id) for handle in dropped_before_reduction
                ),
            ),
        )

    per_frame_budget = max(policy.min_tokens_per_frame, policy.max_visual_tokens // len(retained))
    reduced_batches: list[ImageEmbedsBatch] = []
    reduced_handle_tokens: list[tuple[int, int]] = []

    for handle in retained:
        try:
            batch = handle.to_image_embeds_batch()
            reduced_batch = spatial_reduce_batch_to_max_tokens_per_frame(
                batch,
                max_tokens_per_frame=per_frame_budget,
            )
            reduced_batches.append(reduced_batch)
            if reduced_batch.num_visual_tokens != batch.num_visual_tokens:
                reduced_handle_tokens.append(
                    (int(handle.handle_id), int(reduced_batch.num_visual_tokens))
                )
        finally:
            handle.release()

    while reduced_batches and sum(batch.num_visual_tokens for batch in reduced_batches) > policy.max_visual_tokens:
        reduced_batches.pop(0)

    for handle in dropped_before_reduction:
        handle.release()

    output_handles = len(reduced_batches)
    output_tokens = sum(int(batch.num_visual_tokens) for batch in reduced_batches)
    dropped_handle_ids = tuple(int(handle.handle_id) for handle in dropped_before_reduction)

    if not reduced_batches:
        return (
            None,
            TabmAdmissionDecision(
                policy_applied=True,
                policy="merge_only",
                reason="insufficient_budget",
                input_handles=input_handles,
                output_handles=0,
                input_tokens=input_tokens,
                output_tokens=0,
                dropped_handle_ids=dropped_handle_ids,
                reduced_handle_tokens=tuple(reduced_handle_tokens),
            ),
        )

    has_dropped_handle_ids = bool(dropped_handle_ids)
    has_reduced_handles = bool(reduced_handle_tokens)
    policy_name = (
        "uniform_reduce" if has_reduced_handles else
        "drop_only" if has_dropped_handle_ids else
        "merge_only"
    )
    reason = (
        "single_handle" if (
            input_handles == 1
            and not has_dropped_handle_ids
            and not has_reduced_handles
        )
        else "policy_budget"
    )

    return (
        _concatenate_batches(reduced_batches),
        TabmAdmissionDecision(
            policy_applied=True,
            policy=policy_name,
            reason=reason,
            input_handles=input_handles,
            output_handles=output_handles,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            dropped_handle_ids=dropped_handle_ids,
            reduced_handle_tokens=tuple(reduced_handle_tokens),
        ),
    )

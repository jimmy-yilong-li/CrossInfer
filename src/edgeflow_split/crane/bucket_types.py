# src/edgeflow_split/crane/bucket_types.py
"""Bucket data types, validation and exceptions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BucketSignature:
    """A single bucket in the bucket set.

    grid_h, grid_w: token-grid dimensions (after patch + spatial-merge).
    patch_size: source-image patch size used by the visual encoder; baked
        into the signature so the cache key is stable per encoder.
    hidden_size: model hidden width, baked for the same reason.
    """
    grid_h: int
    grid_w: int
    patch_size: int
    hidden_size: int

    def __post_init__(self) -> None:
        for name in ("grid_h", "grid_w", "patch_size", "hidden_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive; got {getattr(self, name)}")

    @property
    def token_count(self) -> int:
        return self.grid_h * self.grid_w

    def dominates(self, other: "BucketSignature") -> bool:
        """True iff every grid dim of self is >= the corresponding dim of other.

        patch_size and hidden_size must match (a bucket from a different
        encoder family cannot dominate). The dispatcher will only ever
        compare signatures that share patch_size and hidden_size because
        BucketSet enforces homogeneity, but we still assert it here so
        callers cannot abuse this method.
        """
        if (self.patch_size, self.hidden_size) != (other.patch_size, other.hidden_size):
            raise ValueError(
                "dominates() compares only signatures with the same patch_size "
                f"and hidden_size; got self=({self.patch_size},{self.hidden_size}) "
                f"other=({other.patch_size},{other.hidden_size})"
            )
        return self.grid_h >= other.grid_h and self.grid_w >= other.grid_w


@dataclass(frozen=True)
class BucketSet:
    """A homogeneous set of buckets for one encoder.

    All buckets share patch_size and hidden_size so the cache key is
    well-defined per (model_id, bucket_signature).
    """
    buckets: Tuple[BucketSignature, ...]

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("BucketSet must be non-empty")
        head = self.buckets[0]
        for b in self.buckets[1:]:
            if b.patch_size != head.patch_size:
                raise ValueError(
                    f"BucketSet patch_size must be homogeneous; got "
                    f"{head.patch_size} and {b.patch_size}"
                )
            if b.hidden_size != head.hidden_size:
                raise ValueError(
                    f"BucketSet hidden_size must be homogeneous; got "
                    f"{head.hidden_size} and {b.hidden_size}"
                )


class OutOfBucketMaxError(Exception):
    """Raised when no bucket in BucketSet dominates the real input shape.

    The bucket set's effective maximum is computed (per-dim max across the
    set); if real_grid exceeds that, the input is unsupported.
    """
    def __init__(self, real_grid: tuple[int, int], bucket_max: BucketSignature):
        self.real_grid = real_grid
        self.bucket_max = bucket_max
        super().__init__(
            f"real_grid={real_grid} not dominated by any bucket; "
            f"bucket_max=(grid_h={bucket_max.grid_h}, grid_w={bucket_max.grid_w})"
        )


class BoundaryContractViolation(Exception):
    """Raised at SliceBoundary when any of C4 a/b/c contracts fails.

    contract: "C4a", "C4b", or "C4c".
    detail: human-readable summary including offending values.
    """
    def __init__(self, contract: str, detail: str):
        self.contract = contract
        self.detail = detail
        super().__init__(f"{contract}: {detail}")

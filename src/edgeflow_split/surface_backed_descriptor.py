"""Surface-backed visual descriptor for TABM transport.

Ownership and release contract:
- Producer allocates an ANESurfaceLease, runs the final ANE kernel bound to
  the leased surface, and constructs a SurfaceBackedVisualDescriptor.
- Producer pushes the descriptor into TABM. Ownership transfers to TABM.
- Consumer pops the descriptor from TABM. Ownership transfers to consumer.
- Consumer MUST call either release() or materialize_to_batch() exactly once.
  Both paths end in the released state and free the underlying lease.
- If TABM or flow policy drops a descriptor, the dropping code MUST call
  release() on the dropped descriptor before discarding it.
- release() is idempotent; double release is a safe no-op.
- materialize_to_batch() is a one-way downgrade: after success, the descriptor
  is released and cannot be consumed again.

This type is intentionally not frozen. It owns a lease resource and has
mutable lifecycle state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading

import numpy as np

from edgeflow_split.ane_surface import ANESurfaceLease
from edgeflow_split.contracts import ImageEmbedsBatch


SURFACE_DATA_PLANE_STAGED_CPU_SURFACE = "staged_cpu_surface"
SURFACE_DATA_PLANE_ANE_OUTPUT_SURFACE = "ane_output_surface"


@dataclass
class SurfaceBackedVisualDescriptor:
    descriptor_id: int
    surface_lease: ANESurfaceLease | None
    token_count: int
    hidden_size: int
    dtype: str
    image_grid_thw: tuple[int, int, int]
    frame_id: int
    timestamp_ms: float
    source_image_size: tuple[int, int] | None = None
    model_revision: str | None = None
    data_plane: str = SURFACE_DATA_PLANE_STAGED_CPU_SURFACE
    handoff_receipt: dict = field(default_factory=dict)
    text_prefix_tokens: int = 0
    text_suffix_tokens: int = 0
    prepared_prompt: str | None = None
    _released: bool = field(default=False, init=False, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    @property
    def requires_native_surface_consumer(self) -> bool:
        return self.data_plane == SURFACE_DATA_PLANE_ANE_OUTPUT_SURFACE

    @property
    def released(self) -> bool:
        with self._lock:
            return self._released

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
            lease = self.surface_lease
            self.surface_lease = None
        if lease is not None:
            lease.release()

    def materialize_to_batch(self) -> ImageEmbedsBatch:
        if self.text_prefix_tokens or self.text_suffix_tokens:
            raise RuntimeError("packed prompt surfaces require the native packed consumer")
        with self._lock:
            if self._released:
                raise RuntimeError("descriptor already released")
            lease = self.surface_lease

        if lease is None:
            raise RuntimeError("descriptor has no active surface lease")

        np_dtype = np.dtype(self.dtype)
        buf = np.empty((self.token_count, self.hidden_size), dtype=np_dtype)
        lease_bytes = getattr(lease, "surface_bytes", None)
        if lease_bytes is not None and int(buf.nbytes) > int(lease_bytes):
            raise ValueError("descriptor payload exceeds leased surface size")
        lease.read_into(buf)
        batch = ImageEmbedsBatch(
            image_embeds=buf,
            image_grid_thw=self.image_grid_thw,
            num_visual_tokens=self.token_count,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
            source_image_size=self.source_image_size,
            model_revision=self.model_revision,
        )

        self.release()
        return batch

    def __enter__(self) -> SurfaceBackedVisualDescriptor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

"""Shared tensor handle for zero-copy transport on unified memory.

The control plane (ring buffer) passes lightweight descriptors.
The data plane (tensor data) stays in unified memory — Metal buffer,
IOSurface, or numpy array — without CPU-side copies.

Lifecycle:
  1. Producer creates handle via SharedTensorHandle.from_batch() — refcount=1
  2. Producer pushes handle to HandleRingBuffer — refcount stays 1 (ownership transfer)
  3. Consumer pops handle — owns the handle
  4. Consumer calls handle.release() when done — refcount drops, tensor eligible for reuse
  5. If ring buffer drops a handle (push_with_drop), dropped handle is returned
     to caller for explicit release
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from edgeflow_split.contracts import _DTYPE_TO_BYTES


# TABM zero-copy visibility constants.
TABM_ZERO_COPY_CONTROL_PLANE = "control-plane"
TABM_ZERO_COPY_DATA_PLANE_VISUAL_READBACK = "control-plane-plus-readback"
TABM_ZERO_COPY_DATA_PLANE_METAL_LINK = "control-plane-plus-readback-plus-metal"


# ---------------------------------------------------------------------------
# Module-level handle ID counter (thread-safe)
# ---------------------------------------------------------------------------

_HANDLE_ID_LOCK = threading.Lock()
_HANDLE_NEXT_ID = [0]


def _next_handle_id() -> int:
    with _HANDLE_ID_LOCK:
        hid = _HANDLE_NEXT_ID[0]
        _HANDLE_NEXT_ID[0] += 1
        return hid


# ---------------------------------------------------------------------------
# Canonical dtype validation
# ---------------------------------------------------------------------------

_VALID_DTYPES = frozenset(_DTYPE_TO_BYTES.keys())


def _validate_dtype(dtype: str) -> str:
    """Normalize and validate dtype against the canonical vocabulary."""
    try:
        normalized = dtype.lower()
    except AttributeError as exc:
        raise TypeError("dtype must be a string") from exc
    if normalized not in _VALID_DTYPES:
        raise ValueError(
            f"unsupported dtype: {normalized!r}; "
            f"must be one of {sorted(_VALID_DTYPES)}"
        )
    return normalized


def _infer_dtype(tensor: Any) -> str:
    """Infer canonical dtype string from a tensor object."""
    dtype_str = str(getattr(tensor, "dtype", ""))

    # Strip framework prefixes: "torch.bfloat16" -> "bfloat16", "mlx.core.bfloat16" -> "bfloat16"
    for prefix in ("torch.", "mlx.core.", "numpy."):
        if dtype_str.startswith(prefix):
            dtype_str = dtype_str[len(prefix):]

    # Map common aliases
    _ALIASES = {"float16": "float16", "float32": "float32", "float64": "float64"}
    dtype_str = _ALIASES.get(dtype_str, dtype_str)

    if dtype_str in _VALID_DTYPES:
        return dtype_str
    raise ValueError(f"cannot infer canonical dtype from tensor dtype {getattr(tensor, 'dtype', '?')!r}")


# ---------------------------------------------------------------------------
# Buffer identity
# ---------------------------------------------------------------------------


def _extract_buffer_id(tensor: Any) -> int:
    """Extract a best-effort identity for the underlying storage.

    For mx.array: Python object identity (MLX does not expose a public Metal-buffer
    pointer here yet).
    For np.ndarray: data pointer (ctypes.data).
    For torch.Tensor: data_ptr().
    Fallback: Python object id.
    """
    try:
        tensor_module = getattr(type(tensor), "__module__", "")
        if isinstance(tensor_module, str) and tensor_module.startswith("mlx.core"):
            return id(tensor)
    except Exception:
        # Keep this helper robust against unusual tensor wrappers.
        pass

    try:
        import numpy as np
        if isinstance(tensor, np.ndarray):
            return tensor.ctypes.data
    except (ImportError, AttributeError):
        pass

    try:
        import torch
        if torch.is_tensor(tensor):
            return tensor.data_ptr()
    except ImportError:
        pass

    return id(tensor)


# ---------------------------------------------------------------------------
# SharedTensorHandle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedTensorHandle:
    """Descriptor for a tensor living in shared/unified memory.

    The ring buffer transports these handles, NOT the tensor bytes.
    The underlying tensor stays wherever the producer put it (Metal buffer,
    IOSurface, host memory) and the consumer accesses it directly.

    Lifecycle: acquire via from_batch(), release via release(). The handle
    pins the underlying tensor (prevents Python GC) until released.
    """

    handle_id: int                              # unique per-handle, monotonic
    buffer_id: int                              # identity of underlying memory buffer
    tensor: Any                                 # mx.array, np.ndarray, torch.Tensor
    shape: tuple[int, ...]                      # (num_tokens, hidden_size)
    dtype: str                                  # canonical: "bfloat16", "float32", etc.
    frame_id: int                               # which frame produced this
    timestamp_ms: float                         # production time (monotonic)
    token_count: int                            # number of visual tokens
    image_grid_thw: tuple[int, int, int]        # spatial grid metadata
    source_image_size: tuple[int, int] | None = None
    model_revision: str | None = None
    _released: bool = field(default=False, repr=False, compare=False)

    @classmethod
    def from_batch(
        cls,
        image_embeds: Any,
        *,
        frame_id: int,
        image_grid_thw: tuple[int, int, int],
        dtype: str | None = None,
        source_image_size: tuple[int, int] | None = None,
        model_revision: str | None = None,
    ) -> SharedTensorHandle:
        """Wrap a tensor as a shared handle.

        If dtype is None, it is inferred from the tensor. If provided,
        it is validated against the canonical dtype vocabulary.
        """
        shape = tuple(int(d) for d in image_embeds.shape)
        if len(shape) != 2:
            raise ValueError("image_embeds must be 2D (num_tokens, hidden_size)")

        inferred_dtype = _infer_dtype(image_embeds)
        if dtype is None:
            resolved_dtype = inferred_dtype
        else:
            resolved_dtype = _validate_dtype(dtype)
            if resolved_dtype != inferred_dtype:
                raise ValueError(
                    f"explicit dtype {resolved_dtype!r} does not match tensor dtype {inferred_dtype!r}"
                )

        return cls(
            handle_id=_next_handle_id(),
            buffer_id=_extract_buffer_id(image_embeds),
            tensor=image_embeds,
            shape=shape,
            dtype=resolved_dtype,
            frame_id=frame_id,
            timestamp_ms=time.monotonic() * 1000.0,
            token_count=shape[0],
            image_grid_thw=image_grid_thw,
            source_image_size=source_image_size,
            model_revision=model_revision,
        )

    @property
    def num_visual_tokens(self) -> int:
        return self.token_count

    @property
    def hidden_size(self) -> int:
        return self.shape[1]

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        """Mark this handle as released. The underlying tensor reference is cleared.

        After release, accessing .tensor raises. The ring buffer returns dropped
        handles so the caller can release them explicitly.
        """
        if self._released:
            return
        # frozen dataclass: use object.__setattr__
        object.__setattr__(self, "_released", True)
        object.__setattr__(self, "tensor", None)

    def to_image_embeds_batch(self) -> Any:
        """Convert back to ImageEmbedsBatch for consumers that expect it."""
        if self._released:
            raise RuntimeError("cannot convert a released handle to ImageEmbedsBatch")
        from edgeflow_split.contracts import ImageEmbedsBatch

        return ImageEmbedsBatch(
            image_embeds=self.tensor,
            image_grid_thw=self.image_grid_thw,
            num_visual_tokens=self.token_count,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
            source_image_size=self.source_image_size,
            model_revision=self.model_revision,
        )


# ---------------------------------------------------------------------------
# HandleRingBuffer
# ---------------------------------------------------------------------------


@dataclass
class HandleRingBuffer:
    """Control-plane ring buffer that transports SharedTensorHandles.

    This is the TABM descriptor queue. Tensor data stays in unified memory;
    only lightweight handles (shape, dtype, buffer identity) flow through
    the buffer. Thread-safe for single-producer / single-consumer.
    """

    _capacity: int
    _buffer: list[SharedTensorHandle | None] = field(init=False, repr=False)
    _read_index: int = field(init=False, default=0)
    _write_index: int = field(init=False, default=0)
    _occupancy: int = field(init=False, default=0)
    _max_occupancy: int = field(init=False, default=0)
    _blocked_push_count: int = field(init=False, default=0)
    _blocked_pop_count: int = field(init=False, default=0)
    _total_tokens_pushed: int = field(init=False, default=0)
    _total_tokens_dropped: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not isinstance(self._capacity, int) or self._capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._buffer = [None] * self._capacity
        self._lock = threading.Lock()

    def capacity(self) -> int:
        return self._capacity

    def occupancy(self) -> int:
        with self._lock:
            return self._occupancy

    def max_occupancy(self) -> int:
        with self._lock:
            return self._max_occupancy

    def blocked_push_count(self) -> int:
        with self._lock:
            return self._blocked_push_count

    def blocked_pop_count(self) -> int:
        with self._lock:
            return self._blocked_pop_count

    def total_tokens_pushed(self) -> int:
        with self._lock:
            return self._total_tokens_pushed

    def total_tokens_dropped(self) -> int:
        with self._lock:
            return self._total_tokens_dropped

    def is_full(self) -> bool:
        with self._lock:
            return self._occupancy == self._capacity

    def is_empty(self) -> bool:
        with self._lock:
            return self._occupancy == 0

    def push(self, handle: SharedTensorHandle) -> bool:
        """Push a handle into the buffer. Returns False if full (backpressure)."""
        if handle.released:
            raise ValueError("cannot push a released handle")
        with self._lock:
            if self._occupancy == self._capacity:
                self._blocked_push_count += 1
                return False
            self._buffer[self._write_index] = handle
            self._write_index = (self._write_index + 1) % self._capacity
            self._occupancy += 1
            self._total_tokens_pushed += handle.token_count
            if self._occupancy > self._max_occupancy:
                self._max_occupancy = self._occupancy
            return True

    def push_with_drop(self, handle: SharedTensorHandle) -> SharedTensorHandle | None:
        """Push a handle, dropping the oldest entry if full.

        Returns the dropped handle (caller should call .release() on it),
        or None if no drop was needed.
        """
        if handle.released:
            raise ValueError("cannot push a released handle")
        with self._lock:
            dropped = None
            if self._occupancy == self._capacity:
                dropped = self._buffer[self._read_index]
                if dropped is not None:
                    self._total_tokens_dropped += dropped.token_count
                self._buffer[self._read_index] = None
                self._read_index = (self._read_index + 1) % self._capacity
                self._occupancy -= 1

            self._buffer[self._write_index] = handle
            self._write_index = (self._write_index + 1) % self._capacity
            self._occupancy += 1
            self._total_tokens_pushed += handle.token_count
            if self._occupancy > self._max_occupancy:
                self._max_occupancy = self._occupancy
            return dropped

    def pop(self) -> SharedTensorHandle | None:
        """Pop the oldest handle. Returns None if empty."""
        with self._lock:
            if self._occupancy == 0:
                self._blocked_pop_count += 1
                return None
            handle = self._buffer[self._read_index]
            self._buffer[self._read_index] = None
            self._read_index = (self._read_index + 1) % self._capacity
            self._occupancy -= 1
            return handle

    def peek_all(self) -> list[SharedTensorHandle]:
        """Non-destructively view all queued handles (oldest first).

        Useful for token budget decisions before popping.
        """
        with self._lock:
            result = []
            for i in range(self._occupancy):
                idx = (self._read_index + i) % self._capacity
                handle = self._buffer[idx]
                if handle is not None:
                    result.append(handle)
            return result

    def drain(self) -> list[SharedTensorHandle]:
        """Pop all handles at once (oldest first)."""
        with self._lock:
            result = []
            while self._occupancy > 0:
                handle = self._buffer[self._read_index]
                self._buffer[self._read_index] = None
                self._read_index = (self._read_index + 1) % self._capacity
                self._occupancy -= 1
                if handle is not None:
                    result.append(handle)
            return result

    def total_queued_tokens(self) -> int:
        """Sum of token_count across all queued handles."""
        with self._lock:
            total = 0
            for i in range(self._occupancy):
                idx = (self._read_index + i) % self._capacity
                handle = self._buffer[idx]
                if handle is not None:
                    total += handle.token_count
            return total

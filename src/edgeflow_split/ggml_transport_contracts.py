from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from edgeflow_split.contracts import _DTYPE_TO_BYTES
from edgeflow_split.contracts import ImageEmbedsBatch
from edgeflow_split.shared_tensor import SharedTensorHandle
from edgeflow_split.surface_backed_descriptor import SurfaceBackedVisualDescriptor


GGML_TRANSPORT_DTYPES = frozenset({"bfloat16", "float16", "float32"})


@dataclass(frozen=True)
class GGMLGenerateResult:
    text: str
    first_token_ms: float | None = None
    prompt_eval_ms: float | None = None
    eval_ms: float | None = None
    eval_count: int | None = None
    ingress_copy_ms: float | None = None
    ingress_payload_bytes: int | None = None


def as_numpy_and_bytes_dtype(value, dtype: str) -> tuple[np.ndarray, str]:
    canonical = str(dtype).lower()
    if canonical not in _DTYPE_TO_BYTES or canonical not in GGML_TRANSPORT_DTYPES:
        raise ValueError(f"unsupported dtype for ggml transport: {dtype}")

    try:
        import torch

        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            if canonical == "bfloat16":
                return tensor.view(torch.uint16).numpy(), canonical
            if canonical == "float16":
                return tensor.to(torch.float16).numpy(), canonical
            if canonical == "float32":
                return tensor.to(torch.float32).numpy(), canonical
            raise ValueError(f"unsupported torch ggml transport dtype: {dtype}")
    except (ImportError, AttributeError, TypeError):
        pass

    try:
        value_type = type(value)
        if value_type.__module__.startswith("mlx.") or value_type.__module__.startswith("mlx_"):
            import mlx.core as mx

            if canonical == "bfloat16":
                return np.asarray(mx.view(value, mx.uint16)), canonical
            if canonical == "float16":
                return np.asarray(value.astype(mx.float16)), canonical
            if canonical == "float32":
                return np.asarray(value.astype(mx.float32)), canonical
            raise ValueError(f"unsupported mlx ggml transport dtype: {dtype}")
    except (ImportError, AttributeError, TypeError):
        pass

    array = np.asarray(value)
    if canonical == "float16":
        return np.asarray(array, dtype=np.float16), canonical
    if canonical == "float32":
        return np.asarray(array, dtype=np.float32), canonical
    raise ValueError(f"unsupported numpy ggml transport dtype: {dtype}")


@dataclass(frozen=True)
class VisualPayloadView:
    image_grid_thw: tuple[int, int, int]
    num_visual_tokens: int
    hidden_size: int
    dtype: str
    source_image_size: tuple[int, int] | None = None
    model_revision: str | None = None
    _source: object | None = field(default=None, repr=False, compare=False)
    _shape: tuple[int, ...] | None = None
    _handle_id: int | None = None
    _buffer_id: int | None = None
    _frame_id: int | None = None
    _timestamp_ms: float | None = None
    _surface_lease: object | None = field(default=None, repr=False, compare=False)
    _descriptor: SurfaceBackedVisualDescriptor | None = field(default=None, repr=False, compare=False)
    _array_cache: dict[str, tuple[np.ndarray, str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_batch(cls, batch: ImageEmbedsBatch) -> VisualPayloadView:
        return cls(
            image_grid_thw=tuple(int(v) for v in batch.image_grid_thw),
            num_visual_tokens=int(batch.num_visual_tokens),
            hidden_size=int(batch.hidden_size),
            dtype=str(batch.dtype),
            source_image_size=batch.source_image_size,
            model_revision=batch.model_revision,
            _source=batch.image_embeds,
        )

    @classmethod
    def from_handle(cls, handle: SharedTensorHandle) -> VisualPayloadView:
        if handle.released:
            raise RuntimeError("cannot prepare a released handle for ggml transport")
        return cls(
            image_grid_thw=tuple(int(v) for v in handle.image_grid_thw),
            num_visual_tokens=int(handle.token_count),
            hidden_size=int(handle.hidden_size),
            dtype=str(handle.dtype),
            source_image_size=handle.source_image_size,
            model_revision=handle.model_revision,
            _source=handle.tensor,
            _shape=tuple(int(v) for v in handle.shape),
            _handle_id=int(handle.handle_id),
            _buffer_id=int(handle.buffer_id),
            _frame_id=int(handle.frame_id),
            _timestamp_ms=float(handle.timestamp_ms),
        )

    @classmethod
    def from_descriptor(cls, descriptor: SurfaceBackedVisualDescriptor) -> VisualPayloadView:
        lease = descriptor.surface_lease
        if lease is None or descriptor.released:
            raise RuntimeError("cannot prepare a released surface-backed descriptor for native transport")
        return cls(
            image_grid_thw=tuple(int(v) for v in descriptor.image_grid_thw),
            num_visual_tokens=int(descriptor.token_count),
            hidden_size=int(descriptor.hidden_size),
            dtype=str(descriptor.dtype),
            source_image_size=descriptor.source_image_size,
            model_revision=descriptor.model_revision,
            _shape=(int(descriptor.token_count), int(descriptor.hidden_size)),
            _frame_id=int(descriptor.frame_id),
            _timestamp_ms=float(descriptor.timestamp_ms),
            _surface_lease=lease,
            _descriptor=descriptor,
        )

    def metadata_for_ggml(self) -> dict[str, object]:
        contiguous, payload_dtype = self._payload_array_and_dtype()
        metadata = {
            "image_grid_thw": [int(v) for v in self.image_grid_thw],
            "num_visual_tokens": int(self.num_visual_tokens),
            "hidden_size": int(self.hidden_size),
            "dtype": str(self.dtype),
            "payload_dtype": payload_dtype,
            "payload_bytes": int(contiguous.nbytes),
            "source_image_size": list(self.source_image_size) if self.source_image_size is not None else None,
            "model_revision": self.model_revision,
        }
        self._add_handle_metadata(metadata, shape_as=list)
        return metadata

    def payload_array(self, *, dtype: str | None = None) -> np.ndarray:
        contiguous, _payload_dtype = self._payload_array_and_dtype(dtype=dtype)
        return contiguous

    def payload_bytes(self) -> bytes:
        return self.payload_array().tobytes()

    def native_array_request_fields(self, *, dtype: str = "float32") -> dict[str, object]:
        visual = self.payload_array(dtype=dtype)
        visual_ptr = 0 if int(self.num_visual_tokens) == 0 or int(visual.nbytes) == 0 else int(visual.ctypes.data)
        request = {
            "image_grid_thw": tuple(int(v) for v in self.image_grid_thw),
            "num_visual_tokens": int(self.num_visual_tokens),
            "hidden_size": int(self.hidden_size),
            "visual_ptr": visual_ptr,
            "visual_nbytes": int(visual.nbytes),
            "_visual_array": visual,
            "_visual_payload": self,
        }
        self._add_handle_metadata(request, shape_as=tuple)
        return request

    def native_surface_request_fields(self) -> dict[str, object]:
        if self._surface_lease is None or self._descriptor is None:
            raise RuntimeError("visual payload has no active surface lease")
        return {
            "image_grid_thw": tuple(int(v) for v in self.image_grid_thw),
            "num_visual_tokens": int(self.num_visual_tokens),
            "hidden_size": int(self.hidden_size),
            "surface_lease": self._surface_lease,
            "_descriptor": self._descriptor,
            "_visual_payload": self,
        }

    def _payload_array_and_dtype(self, *, dtype: str | None = None) -> tuple[np.ndarray, str]:
        if self._source is None:
            raise RuntimeError("visual payload has no CPU-addressable source")

        requested = str(dtype or self.dtype).lower()
        cached = self._array_cache.get(requested)
        if cached is not None:
            return cached

        if (
            isinstance(self._source, np.ndarray)
            and self._source.ndim == 2
            and self._source.flags.c_contiguous
        ):
            expected = {
                "float16": np.dtype(np.float16),
                "float32": np.dtype(np.float32),
            }.get(requested)
            if expected is not None and self._source.dtype == expected:
                result = (self._source, requested)
                self._array_cache[requested] = result
                return result

        array, payload_dtype = as_numpy_and_bytes_dtype(self._source, requested)
        if array.ndim != 2:
            raise ValueError("image_embeds must be a 2D tensor")
        if dtype is None:
            contiguous = np.ascontiguousarray(array)
        else:
            contiguous = np.ascontiguousarray(array, dtype=np.dtype(requested))
        result = (contiguous, str(payload_dtype))
        self._array_cache[requested] = result
        return result

    def _add_handle_metadata(self, metadata: dict[str, object], *, shape_as) -> None:
        if self._shape is not None:
            metadata["shape"] = shape_as(int(v) for v in self._shape)
        if self._handle_id is not None:
            metadata["handle_id"] = int(self._handle_id)
        if self._buffer_id is not None:
            metadata["buffer_id"] = int(self._buffer_id)
        if self._frame_id is not None:
            metadata["frame_id"] = int(self._frame_id)
        if self._timestamp_ms is not None:
            metadata["timestamp_ms"] = float(self._timestamp_ms)


def prepare_batch_request_for_ggml(batch: ImageEmbedsBatch) -> tuple[dict[str, object], np.ndarray]:
    view = VisualPayloadView.from_batch(batch)
    return view.metadata_for_ggml(), view.payload_array()


def serialize_batch_request_for_ggml(batch: ImageEmbedsBatch) -> tuple[dict[str, object], bytes]:
    view = VisualPayloadView.from_batch(batch)
    return view.metadata_for_ggml(), view.payload_bytes()


def prepare_handle_request_for_ggml(handle: SharedTensorHandle) -> tuple[dict[str, object], np.ndarray]:
    view = VisualPayloadView.from_handle(handle)
    return view.metadata_for_ggml(), view.payload_array()

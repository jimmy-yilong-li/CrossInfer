"""CoreML visual output backed directly by an existing TABM surface lease."""

from __future__ import annotations

import ctypes
import threading
import time

import numpy as np

from edgeflow_split.surface_backed_descriptor import (
    SURFACE_DATA_PLANE_ANE_OUTPUT_SURFACE,
    SurfaceBackedVisualDescriptor,
)


class CoreMLOutput(ctypes.Structure):
    _fields_ = [
        ("lease_handle", ctypes.c_void_p), ("pointer", ctypes.c_void_p),
        ("generation", ctypes.c_uint64), ("nbytes", ctypes.c_size_t),
        ("shape", ctypes.c_int32 * 3), ("strides", ctypes.c_int32 * 3),
        ("surface_id", ctypes.c_uint32), ("slot", ctypes.c_uint32),
        ("object_matches", ctypes.c_int32), ("pointer_matches", ctypes.c_int32),
    ]


class CoreMLSurfaceModel:
    def __init__(self, package, library, *, width, tokens, input_shape):
        self.width, self.tokens = int(width), int(tokens)
        self.input_shape = tuple(input_shape)
        self._lock = threading.Lock()
        self.lib = ctypes.CDLL(str(library))
        self.lib.ef_coreml_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
        self.lib.ef_coreml_init.restype = ctypes.c_void_p
        self.lib.ef_coreml_predict_into.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.c_void_p, ctypes.POINTER(CoreMLOutput),
        ]
        self.lib.ef_coreml_predict_into.restype = ctypes.c_int
        packed = getattr(self.lib, "ef_coreml_predict_packed", None)
        if packed is not None:
            packed.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
                ctypes.c_int32, ctypes.POINTER(CoreMLOutput),
            ]
            packed.restype = ctypes.c_int
        self.lib.fv_coreml_error.restype = ctypes.c_char_p
        self.lib.fv_coreml_free.argtypes = [ctypes.c_void_p]
        self.context = self.lib.ef_coreml_init(str(package).encode(), self.width)
        if not self.context:
            raise RuntimeError(self.lib.fv_coreml_error().decode())

    def predict_to_descriptor(self, pixels, *, pool, frame_id, grid_thw,
                              model_revision=None, acquire_timeout=None,
                              text_embeddings=None, prefix_tokens=0, prompt=None):
        if pixels.dtype != np.float32 or not pixels.flags.c_contiguous:
            raise ValueError("CoreML inputs must already be contiguous float32")
        if pixels.shape != self.input_shape:
            raise ValueError(f"expected input shape {self.input_shape}, got {pixels.shape}")
        suffix_tokens = 0
        if text_embeddings is not None:
            if (text_embeddings.dtype != np.float32 or not text_embeddings.flags.c_contiguous
                    or text_embeddings.ndim != 2 or text_embeddings.shape[1] != self.width
                    or not 0 < prefix_tokens < text_embeddings.shape[0] or not prompt):
                raise ValueError("packed text requires contiguous FP32 rows and a prompt boundary")
            if not hasattr(self.lib, "ef_coreml_predict_packed"):
                raise RuntimeError("CoreML library lacks packed prompt output support")
            suffix_tokens = text_embeddings.shape[0] - prefix_tokens
        elif prefix_tokens or prompt is not None:
            raise ValueError("a packed prompt requires text embeddings")
        lease = pool.acquire(timeout=acquire_timeout)
        try:
            with self._lock:
                if not self.context:
                    raise RuntimeError("CoreML surface model is closed")
                output = CoreMLOutput()
                prediction_started = time.monotonic()
                arguments = (self.context,
                             pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                             pixels.size, lease.lease_handle)
                if text_embeddings is None:
                    status = self.lib.ef_coreml_predict_into(*arguments, ctypes.byref(output))
                else:
                    status = self.lib.ef_coreml_predict_packed(
                        *arguments, text_embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        prefix_tokens, suffix_tokens, ctypes.byref(output),
                    )
                prediction_finished = time.monotonic()
                if status:
                    raise RuntimeError(self.lib.fv_coreml_error().decode())
            if (not output.pointer or not output.surface_id or not output.pointer_matches
                    or not output.object_matches
                    or tuple(output.shape) != (1, self.tokens, self.width)
                    or output.nbytes != self.tokens * self.width * 4):
                raise RuntimeError("CoreML shared-output backing contract failed")
            return SurfaceBackedVisualDescriptor(
                descriptor_id=int(output.generation), surface_lease=lease,
                token_count=self.tokens, hidden_size=self.width, dtype="float32",
                image_grid_thw=tuple(grid_thw), frame_id=frame_id, timestamp_ms=0,
                model_revision=model_revision, data_plane=SURFACE_DATA_PLANE_ANE_OUTPUT_SURFACE,
                text_prefix_tokens=prefix_tokens, text_suffix_tokens=suffix_tokens,
                prepared_prompt=prompt,
                handoff_receipt={
                    "data_plane": "coreml_output_backing_iosurface",
                    "compute_units": "CPU_AND_NE",
                    "producer_pointer": int(output.pointer),
                    "surface_id": int(output.surface_id),
                    "generation": int(output.generation),
                    "object_identity": bool(output.object_matches),
                    "pointer_identity": bool(output.pointer_matches),
                    "visual_offset_bytes": prefix_tokens * self.width * 4,
                    "lease_base_pointer": int(output.pointer) - prefix_tokens * self.width * 4,
                    "application_visual_embedding_copy_bytes": 0,
                    "application_text_embedding_copy_bytes": (prefix_tokens + suffix_tokens) * self.width * 4,
                    "application_embedding_copy_bytes": (prefix_tokens + suffix_tokens) * self.width * 4,
                    "prompt_tokens": self.tokens + prefix_tokens + suffix_tokens if prefix_tokens else None,
                    "zero_copy_scope": "CoreML-visible output to native embedding input",
                    "coreml_internal_copy_status": "not_measured",
                    "prediction_started_s": prediction_started,
                    "prediction_finished_s": prediction_finished,
                },
            )
        except BaseException:
            lease.release()
            raise

    def close(self):
        with self._lock:
            if self.context:
                self.lib.fv_coreml_free(self.context)
                self.context = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CoreMLSurfaceChain:
    """Execute partitioned vision without materializing intermediate embeddings."""

    def __init__(self, models, *, intermediate_pool):
        if not models:
            raise ValueError("a visual chain needs at least one model")
        self.models = list(models)
        self.intermediate_pool = intermediate_pool

    def predict_to_descriptor(self, pixels, *, pool, **metadata):
        previous = None
        receipts = []
        started = time.monotonic()
        try:
            for index, model in enumerate(self.models):
                target = pool if index == len(self.models) - 1 else self.intermediate_pool
                result = model.predict_to_descriptor(pixels, pool=target, **metadata)
                if previous is not None:
                    previous.release()
                previous = result
                receipts.append(dict(result.handoff_receipt))
                if index < len(self.models) - 1:
                    # Borrow while the descriptor owns the lease. The next
                    # synchronous prediction completes before it is released.
                    count = result.token_count * result.hidden_size
                    pointer = result.handoff_receipt["producer_pointer"]
                    buffer = (ctypes.c_float * count).from_address(pointer)
                    pixels = np.ctypeslib.as_array(buffer).reshape(self.models[index + 1].input_shape)
            result.handoff_receipt = dict(
                result.handoff_receipt, parts=receipts, intermediate_copy_bytes=0,
                prediction_started_s=started, prediction_finished_s=time.monotonic(),
            )
            return result
        except BaseException:
            if previous is not None:
                previous.release()
            raise

    def close(self):
        for model in reversed(self.models):
            model.close()

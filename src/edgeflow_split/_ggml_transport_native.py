from __future__ import annotations

import atexit
import ctypes
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Protocol

from edgeflow_split.config import SplitConfig
from edgeflow_split.contracts import ImageEmbedsBatch
from edgeflow_split.ggml_transport_contracts import GGMLGenerateResult
from edgeflow_split.ggml_transport_contracts import VisualPayloadView
from edgeflow_split.shared_tensor import SharedTensorHandle
from edgeflow_split.surface_backed_descriptor import SurfaceBackedVisualDescriptor

_EDGEFLOW_LLM_FULL_GPU_OFFLOAD_LAYERS = 999
EDGEFLOW_LLM_ABI_VERSION = 1
_TokenCallback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                 ctypes.c_uint64, ctypes.c_void_p)


class _NativeRuntimeLeaseContext(Protocol):
    def __enter__(self) -> object:
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        ...


class _NativeRuntimeHandle:
    def __init__(self, lib: ctypes.CDLL, ctx: ctypes.c_void_p) -> None:
        self.lib = lib
        self.ctx = ctx
        self._lock = threading.Lock()
        self._execution_lock = threading.Lock()
        self._in_use = 0
        self._close_requested = False

    def acquire(self) -> "_NativeRuntimeLease":
        with self._lock:
            if self.ctx is None or self._close_requested:
                raise RuntimeError("native llama.cpp runtime handle is closed")
            self._in_use += 1
        return _NativeRuntimeLease(self)

    def _release(self) -> None:
        ctx_to_free = None
        with self._lock:
            self._in_use -= 1
            if self._in_use < 0:
                raise RuntimeError("native llama.cpp runtime handle released too many times")
            if self._in_use == 0 and self._close_requested and self.ctx is not None:
                ctx_to_free = self.ctx
                self.ctx = None
        if ctx_to_free is not None:
            self.lib.edgeflow_llm_free(ctx_to_free)

    def close(self) -> None:
        ctx_to_free = None
        with self._lock:
            if self.ctx is None:
                return
            if self._in_use > 0:
                self._close_requested = True
                return
            ctx_to_free = self.ctx
            self.ctx = None
        self.lib.edgeflow_llm_free(ctx_to_free)


class _NativeRuntimeLease:
    def __init__(self, runtime: _NativeRuntimeHandle) -> None:
        self._runtime = runtime
        self._entered = False
        self._released = False

    def __enter__(self) -> _NativeRuntimeHandle:
        if self._entered or self._released:
            raise RuntimeError("native runtime lease is single-use")
        # Reserve lifetime before waiting, but never wait under the cache lock.
        acquired = False
        try:
            acquired = self._runtime._execution_lock.acquire()
            self._entered = True
            return self._runtime
        except BaseException:
            self._released = True
            try:
                if acquired:
                    self._runtime._execution_lock.release()
            finally:
                self._runtime._release()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        if not self._entered or self._released:
            raise RuntimeError("native runtime lease is not active")
        self._released = True
        try:
            self._runtime._execution_lock.release()
        finally:
            self._runtime._release()


class _BorrowedRuntimeLease:
    """Context manager for tests that monkeypatch `_load_native_llm_runtime`.

    Production handles implement `acquire()` and get in-use accounting. Some unit
    tests replace the loader with a plain object to focus on request-shaping
    behavior; keep that seam working without adding lifecycle behavior to fakes.
    """

    def __init__(self, runtime: object) -> None:
        self._runtime = runtime

    def __enter__(self) -> object:
        return self._runtime

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb


class _EdgeFlowLLMInitParams(ctypes.Structure):
    _fields_ = [
        ("n_gpu_layers", ctypes.c_int),
        ("n_ctx", ctypes.c_int),
        ("n_batch", ctypes.c_int),
        ("n_threads", ctypes.c_int),
    ]


class _EdgeFlowLLMRequest(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("visual_embd", ctypes.POINTER(ctypes.c_float)),
        ("num_visual_tokens", ctypes.c_int),
        ("hidden_size", ctypes.c_int),
        ("image_grid_thw", ctypes.c_int * 3),
        ("max_new_tokens", ctypes.c_int),
    ]


# Mirrors tools/edgeflow_llm_runtime/include/edgeflow_llm/edgeflow_llm.h:74-82
EDGEFLOW_MODEL_FAMILY_QWEN25VL = 0
EDGEFLOW_MODEL_FAMILY_FASTVLM = 1
EDGEFLOW_MODEL_FAMILY_LLAVA15 = 2
EDGEFLOW_DTYPE_FP32 = 0
EDGEFLOW_DTYPE_FP16 = 1


class _EdgeFlowLLMRequestV2(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("visual_embd", ctypes.c_void_p),
        ("num_visual_tokens", ctypes.c_int),
        ("hidden_size", ctypes.c_int),
        ("image_grid_thw", ctypes.c_int * 3),
        ("max_new_tokens", ctypes.c_int),
        ("model_family", ctypes.c_int),
        ("visual_dtype", ctypes.c_int),
    ]


class _EdgeFlowLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("first_token_ms", ctypes.c_float),
        ("prompt_eval_ms", ctypes.c_float),
        ("eval_ms", ctypes.c_float),
        ("eval_count", ctypes.c_int),
    ]


_MAX_NATIVE_LLM_RUNTIME_CACHE_SIZE = 8
_NATIVE_LLM_RUNTIME_CACHE: "OrderedDict[tuple[str, str], _NativeRuntimeHandle]" = OrderedDict()
_NATIVE_LLM_RUNTIME_CACHE_LOCK = threading.RLock()


def _c_uint32_function(lib: ctypes.CDLL, name: str):
    fn = getattr(lib, name, None)
    if fn is None:
        raise RuntimeError(f"native llama.cpp ABI guard missing symbol: {name}")
    fn.argtypes = []
    fn.restype = ctypes.c_uint32
    return fn


def _require_native_llm_abi_compatible(lib: ctypes.CDLL) -> None:
    version = int(_c_uint32_function(lib, "edgeflow_llm_abi_version")())
    if version != EDGEFLOW_LLM_ABI_VERSION:
        raise RuntimeError(
            f"native llama.cpp ABI mismatch: expected version {EDGEFLOW_LLM_ABI_VERSION}, got {version}"
        )

    expected_sizes = {
        "edgeflow_llm_init_params_sizeof": ctypes.sizeof(_EdgeFlowLLMInitParams),
        "edgeflow_llm_request_sizeof": ctypes.sizeof(_EdgeFlowLLMRequest),
        "edgeflow_llm_request_v2_sizeof": ctypes.sizeof(_EdgeFlowLLMRequestV2),
        "edgeflow_llm_result_sizeof": ctypes.sizeof(_EdgeFlowLLMResult),
    }
    for symbol, expected in expected_sizes.items():
        actual = int(_c_uint32_function(lib, symbol)())
        if actual != expected:
            raise RuntimeError(
                f"native llama.cpp ABI mismatch: {symbol} expected {expected}, got {actual}"
            )


def _close_native_llm_runtimes() -> None:
    with _NATIVE_LLM_RUNTIME_CACHE_LOCK:
        for runtime in list(_NATIVE_LLM_RUNTIME_CACHE.values()):
            runtime.close()
        _NATIVE_LLM_RUNTIME_CACHE.clear()


atexit.register(_close_native_llm_runtimes)


def _load_native_llm_runtime_locked(config: SplitConfig) -> _NativeRuntimeHandle:
    model_path = config.ggml_model_path or ""
    if not model_path:
        raise RuntimeError("native llama.cpp transport requires ggml_model_path")

    library_path = _resolve_native_library_path(config)
    key = (library_path, model_path)
    cached = _NATIVE_LLM_RUNTIME_CACHE.get(key)
    if cached is not None:
        _NATIVE_LLM_RUNTIME_CACHE.move_to_end(key)
        return cached

    lib = ctypes.CDLL(library_path)
    _require_native_llm_abi_compatible(lib)
    lib.edgeflow_llm_init.argtypes = [ctypes.c_char_p, _EdgeFlowLLMInitParams]
    lib.edgeflow_llm_init.restype = ctypes.c_void_p
    lib.edgeflow_llm_free.argtypes = [ctypes.c_void_p]
    lib.edgeflow_llm_free.restype = None
    lib.edgeflow_llm_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(_EdgeFlowLLMRequest), ctypes.POINTER(_EdgeFlowLLMResult)]
    lib.edgeflow_llm_generate.restype = ctypes.c_int
    generate_v2 = getattr(lib, "edgeflow_llm_generate_v2", None)
    if generate_v2 is not None:
        generate_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_EdgeFlowLLMRequestV2),
            ctypes.POINTER(_EdgeFlowLLMResult),
        ]
        generate_v2.restype = ctypes.c_int
    generate_from_surface = getattr(lib, "edgeflow_llm_generate_from_surface_lease", None)
    if generate_from_surface is not None:
        generate_from_surface.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_EdgeFlowLLMRequest),
            ctypes.c_void_p,
            ctypes.POINTER(_EdgeFlowLLMResult),
        ]
        generate_from_surface.restype = ctypes.c_int
    generate_from_surface_lease_v2 = getattr(lib, "edgeflow_llm_generate_from_surface_lease_v2", None)
    if generate_from_surface_lease_v2 is not None:
        generate_from_surface_lease_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_EdgeFlowLLMRequestV2),
            ctypes.c_void_p,
            ctypes.POINTER(_EdgeFlowLLMResult),
        ]
        generate_from_surface_lease_v2.restype = ctypes.c_int
    generate_from_ane_surface = getattr(lib, "edgeflow_llm_generate_from_ane_surface", None)
    if generate_from_ane_surface is not None:
        generate_from_ane_surface.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_EdgeFlowLLMRequest),
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(_EdgeFlowLLMResult),
        ]
        generate_from_ane_surface.restype = ctypes.c_int
    lib.edgeflow_llm_free_result.argtypes = [ctypes.POINTER(_EdgeFlowLLMResult)]
    lib.edgeflow_llm_free_result.restype = None
    generation_options = getattr(lib, "edgeflow_llm_set_generation_options", None)
    if generation_options is not None:
        generation_options.argtypes = [ctypes.c_void_p, ctypes.c_int, _TokenCallback, ctypes.c_void_p]
        generation_options.restype = ctypes.c_int
    packed = getattr(lib, "edgeflow_llm_generate_from_surface_lease_packed", None)
    if packed is not None:
        packed.argtypes = [ctypes.c_void_p, ctypes.POINTER(_EdgeFlowLLMRequest),
                           ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                           ctypes.POINTER(_EdgeFlowLLMResult)]
        packed.restype = ctypes.c_int
    text_embeddings = getattr(lib, "edgeflow_llm_qwen25_prompt_embeddings", None)
    if text_embeddings is not None:
        text_embeddings.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
                                    ctypes.POINTER(ctypes.c_int32)]
        text_embeddings.restype = ctypes.c_int

    ctx = lib.edgeflow_llm_init(
        os.fsencode(model_path),
        # This vendored llama.cpp build treats -1 as CPU placement; 999 is the full-offload sentinel.
        _EdgeFlowLLMInitParams(
            n_gpu_layers=_EDGEFLOW_LLM_FULL_GPU_OFFLOAD_LAYERS,
            n_ctx=4096,
            n_batch=4096,
            n_threads=4,
        ),
    )
    if not ctx:
        raise RuntimeError(f"native llama.cpp runtime failed to initialize for {model_path}")

    runtime = _NativeRuntimeHandle(lib=lib, ctx=ctypes.c_void_p(ctx))
    _NATIVE_LLM_RUNTIME_CACHE[key] = runtime
    while len(_NATIVE_LLM_RUNTIME_CACHE) > _MAX_NATIVE_LLM_RUNTIME_CACHE_SIZE:
        _, evicted = _NATIVE_LLM_RUNTIME_CACHE.popitem(last=False)
        evicted.close()
    return runtime


def _load_native_llm_runtime(config: SplitConfig) -> _NativeRuntimeHandle:
    with _NATIVE_LLM_RUNTIME_CACHE_LOCK:
        return _load_native_llm_runtime_locked(config)


def _acquire_native_llm_runtime(config: SplitConfig) -> _NativeRuntimeLeaseContext:
    with _NATIVE_LLM_RUNTIME_CACHE_LOCK:
        runtime: object = _load_native_llm_runtime(config)
        acquire = getattr(runtime, "acquire", None)
        if callable(acquire):
            return acquire()
        return _BorrowedRuntimeLease(runtime)


load_native_llm_runtime = _load_native_llm_runtime


def qwen25_prompt_embeddings(prompt: str, config: SplitConfig, *, include_embeddings=True):
    """Tokenize with the loaded GGUF and read only the needed text rows."""
    import numpy as np

    with _acquire_native_llm_runtime(config) as runtime:
        prepare = getattr(runtime.lib, "edgeflow_llm_qwen25_prompt_embeddings", None)
        if prepare is None:
            raise RuntimeError("native library lacks packed Qwen prompt support")
        sizes = (ctypes.c_int32 * 3)()
        encoded = prompt.encode("utf-8")
        if prepare(runtime.ctx, encoded, None, 0, sizes) != 0:
            raise RuntimeError("Qwen prompt layout failed")
        prefix, suffix, width = tuple(sizes)
        if min(prefix, suffix, width) <= 0:
            raise RuntimeError("invalid native Qwen prompt layout")
        embeddings = None
        if include_embeddings:
            embeddings = np.empty((prefix + suffix, width), dtype=np.float32)
            pointer = embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if prepare(runtime.ctx, encoded, pointer, embeddings.size, sizes) != 0:
                raise RuntimeError("Qwen text embedding extraction failed")
        return embeddings, (prefix, suffix, width)


def _probe_native_surface_capability(config: SplitConfig) -> bool:
    """Probe the native runtime dylib at the config-resolved path for
    surface-lease generate support.

    This probe runs per-call rather than caching at import time because
    SplitConfig allows ggml_library_path override. An import-time cache
    would silently drift from the dylib that the current config resolves
    to at runtime. Returns False on any failure (path unresolvable, dylib
    missing, dlopen failure, symbol missing).
    """
    try:
        library_path = _resolve_native_library_path(config)
        lib = ctypes.CDLL(library_path)
    except Exception:
        return False
    return hasattr(lib, "edgeflow_llm_generate_from_surface_lease")


def _surface_path_capability_available(config: SplitConfig) -> bool:
    return _probe_native_surface_capability(config)


def edgeflow_llm_generate_from_surface_lease(runtime: _NativeRuntimeHandle, req_ptr, lease_ptr, out_ptr):
    generate_from_surface = getattr(runtime.lib, "edgeflow_llm_generate_from_surface_lease", None)
    if generate_from_surface is None:
        raise RuntimeError("native llama.cpp runtime is missing edgeflow_llm_generate_from_surface_lease")
    return int(generate_from_surface(runtime.ctx, req_ptr, ctypes.c_void_p(lease_ptr), out_ptr))


def _prepare_batch_request_for_native(batch: ImageEmbedsBatch) -> dict[str, object]:
    request = VisualPayloadView.from_batch(batch).native_array_request_fields(dtype="float32")
    request["model_path"] = None
    request["prompt"] = None
    request["max_new_tokens"] = None
    return request


def _prepare_descriptor_request_for_native(descriptor: SurfaceBackedVisualDescriptor) -> dict[str, object]:
    request = VisualPayloadView.from_descriptor(descriptor).native_surface_request_fields()
    request["model_path"] = None
    request["prompt"] = None
    request["max_new_tokens"] = None
    return request


def _prepare_handle_request_for_native(handle: SharedTensorHandle) -> dict[str, object]:
    if handle.released:
        raise RuntimeError("cannot prepare a released handle for ggml transport")

    canonical = str(handle.dtype).lower()
    if canonical != "float32":
        raise ValueError(
            "native ggml handle transport supports only float32 handles; "
            f"got dtype={handle.dtype!r}. Convert before creating the handle "
            "or use a transport path with an explicit non-float32 payload contract."
        )

    request = VisualPayloadView.from_handle(handle).native_array_request_fields(dtype="float32")
    request["model_path"] = None
    request["prompt"] = None
    request["max_new_tokens"] = None
    # Retain the handle until the synchronous ctypes call returns: it keeps the
    # source tensor/descriptor alive for the same window as the raw pointer.
    request["_handle"] = handle
    return request


def _invoke_native_llm(runtime: _NativeRuntimeHandle, request: dict[str, object]) -> dict[str, object]:
    visual_ptr_value = int(request["visual_ptr"])
    visual_ptr = (
        ctypes.cast(visual_ptr_value, ctypes.POINTER(ctypes.c_float))
        if visual_ptr_value != 0
        else ctypes.POINTER(ctypes.c_float)()
    )
    req = _EdgeFlowLLMRequest(
        prompt=os.fsencode(str(request["prompt"])),
        visual_embd=visual_ptr,
        num_visual_tokens=int(request["num_visual_tokens"]),
        hidden_size=int(request["hidden_size"]),
        image_grid_thw=(ctypes.c_int * 3)(*tuple(int(v) for v in request["image_grid_thw"])),
        max_new_tokens=int(request["max_new_tokens"]),
    )
    out = _EdgeFlowLLMResult()
    rc = runtime.lib.edgeflow_llm_generate(runtime.ctx, ctypes.byref(req), ctypes.byref(out))
    if rc != 0:
        raise RuntimeError(f"native llama.cpp runtime returned error code {rc}")
    try:
        text = ctypes.string_at(out.text).decode("utf-8") if out.text else ""
        return {
            "text": text,
            "first_token_ms": float(out.first_token_ms),
            "prompt_eval_ms": float(out.prompt_eval_ms),
            "eval_ms": float(out.eval_ms),
            "eval_count": int(out.eval_count),
        }
    finally:
        runtime.lib.edgeflow_llm_free_result(ctypes.byref(out))


def _parse_native_out(runtime: _NativeRuntimeHandle, out: _EdgeFlowLLMResult) -> dict[str, object]:
    try:
        text = ctypes.string_at(out.text).decode("utf-8") if out.text else ""
        return {
            "text": text,
            "first_token_ms": float(out.first_token_ms),
            "prompt_eval_ms": float(out.prompt_eval_ms),
            "eval_ms": float(out.eval_ms),
            "eval_count": int(out.eval_count),
        }
    finally:
        runtime.lib.edgeflow_llm_free_result(ctypes.byref(out))


def _invoke_native_llm_v2(runtime: _NativeRuntimeHandle, request: dict[str, object]) -> dict[str, object]:
    generate_fn = getattr(runtime.lib, "edgeflow_llm_generate_v2", None)
    if generate_fn is None:
        raise RuntimeError("loaded ggml library does not export edgeflow_llm_generate_v2")
    # Array/handle preparation owns the contiguous FP32 storage until return.
    req = _EdgeFlowLLMRequestV2(
        prompt=os.fsencode(str(request["prompt"])),
        visual_embd=ctypes.c_void_p(int(request["visual_ptr"])),
        num_visual_tokens=int(request["num_visual_tokens"]),
        hidden_size=int(request["hidden_size"]),
        image_grid_thw=(ctypes.c_int * 3)(*tuple(int(v) for v in request["image_grid_thw"])),
        max_new_tokens=int(request["max_new_tokens"]),
        model_family=int(request["model_family"]),
        visual_dtype=EDGEFLOW_DTYPE_FP32,
    )
    out = _EdgeFlowLLMResult()
    rc = generate_fn(runtime.ctx, ctypes.byref(req), ctypes.byref(out))
    if rc != 0:
        raise RuntimeError(f"edgeflow_llm_generate_v2 failed rc={rc}")
    return _parse_native_out(runtime, out)


def _invoke_native_surface_llm(runtime: _NativeRuntimeHandle, request: dict[str, object]) -> dict[str, object]:
    lease = request["surface_lease"]
    req = _EdgeFlowLLMRequest(
        prompt=os.fsencode(str(request["prompt"])),
        visual_embd=ctypes.POINTER(ctypes.c_float)(),
        num_visual_tokens=int(request["num_visual_tokens"]),
        hidden_size=int(request["hidden_size"]),
        image_grid_thw=(ctypes.c_int * 3)(*tuple(int(v) for v in request["image_grid_thw"])),
        max_new_tokens=int(request["max_new_tokens"]),
    )
    out = _EdgeFlowLLMResult()
    descriptor = request.get("_descriptor")
    if descriptor is not None and (descriptor.text_prefix_tokens or descriptor.text_suffix_tokens):
        if descriptor.prepared_prompt != request["prompt"]:
            raise ValueError("request differs from the prepared prompt in its surface")
        packed = getattr(runtime.lib, "edgeflow_llm_generate_from_surface_lease_packed", None)
        if packed is None:
            raise RuntimeError("native library lacks packed surface consumption")
        rc = packed(runtime.ctx, ctypes.byref(req), ctypes.c_void_p(int(lease._handle_ptr())),
                    descriptor.text_prefix_tokens, descriptor.text_suffix_tokens, ctypes.byref(out))
    else:
        rc = edgeflow_llm_generate_from_surface_lease(
            runtime,
            ctypes.byref(req),
            int(lease._handle_ptr()),
            ctypes.byref(out),
        )
    if rc != 0:
        raise RuntimeError(f"edgeflow_llm_generate_from_surface_lease failed rc={rc}")
    return _parse_native_out(runtime, out)


def _invoke_native_surface_llm_v2(
    runtime: _NativeRuntimeHandle,
    request: dict[str, object],
) -> dict[str, object]:
    """v2 surface-lease invocation, family-agnostic.

    Used by FastVLM (model_family=1) and LLaVA-1.5 (model_family=2). The lease
    holds raw embedding bytes (fp16 or fp32 per visual_dtype). The native
    runtime's family-specific ingest (run_fastvlm_multimodal_ingest /
    run_llava15_multimodal_ingest) handles fp16->fp32 conversion internally
    via ggml_fp16_to_fp32_row, so this wrapper passes the bytes through
    unchanged.
    """
    lease = request["surface_lease"]
    req = _EdgeFlowLLMRequestV2(
        prompt=os.fsencode(str(request["prompt"])),
        visual_embd=ctypes.c_void_p(0),  # filled by scoped consume callback in C
        num_visual_tokens=int(request["num_visual_tokens"]),
        hidden_size=int(request["hidden_size"]),
        image_grid_thw=(ctypes.c_int * 3)(*tuple(int(v) for v in request["image_grid_thw"])),
        max_new_tokens=int(request["max_new_tokens"]),
        model_family=int(request["model_family"]),
        visual_dtype=int(request["visual_dtype"]),
    )
    out = _EdgeFlowLLMResult()
    generate_fn = getattr(runtime.lib, "edgeflow_llm_generate_from_surface_lease_v2", None)
    if generate_fn is None:
        raise RuntimeError(
            "loaded ggml library does not export edgeflow_llm_generate_from_surface_lease_v2"
        )
    rc = generate_fn(
        runtime.ctx,
        ctypes.byref(req),
        ctypes.c_void_p(int(lease._handle_ptr())),
        ctypes.byref(out),
    )
    if rc != 0:
        raise RuntimeError(f"edgeflow_llm_generate_from_surface_lease_v2 failed rc={rc}")
    return _parse_native_out(runtime, out)


@contextmanager
def _generation_options(runtime, on_token=None, ignore_eog=False):
    """Scope optional delivery callbacks to an already execution-locked request."""
    if on_token is None and not ignore_eog:
        yield
        return
    setter = getattr(runtime.lib, "edgeflow_llm_set_generation_options", None)
    if setter is None:
        raise RuntimeError("native runtime lacks generation options; rebuild the library")
    errors = []

    def deliver(token, piece, size, _data):
        try:
            on_token(token, ctypes.string_at(piece, size))
            return 0
        except BaseException as exc:  # noqa: BLE001 - never unwind through a ctypes callback
            errors.append(exc)
            return 1

    callback = _TokenCallback(deliver) if on_token is not None else _TokenCallback()
    if setter(runtime.ctx, int(ignore_eog), callback, None) != 0:
        raise RuntimeError("native generation options rejected")
    try:
        yield
    except BaseException:
        if errors:
            raise errors[0]
        raise
    finally:
        setter(runtime.ctx, 0, _TokenCallback(), None)
    if errors:
        raise errors[0]


def _run_generate_native_visual_payload(
    image_path: Path | str,
    prompt: str,
    visual_payload: ImageEmbedsBatch | SurfaceBackedVisualDescriptor,
    max_new_tokens: int,
    config: SplitConfig,
    *,
    on_token=None,
    ignore_eog=False,
):
    options = {"on_token": on_token, "ignore_eog": ignore_eog} if on_token is not None or ignore_eog else {}
    if options and not (isinstance(visual_payload, SurfaceBackedVisualDescriptor)
                        and config.detected_model_family == EDGEFLOW_MODEL_FAMILY_QWEN25VL):
        if isinstance(visual_payload, SurfaceBackedVisualDescriptor):
            visual_payload.release()
        raise ValueError("generation options currently require the Qwen native surface path")
    if isinstance(visual_payload, SurfaceBackedVisualDescriptor):
        # Both non-Qwen families require v2's explicit family and dtype.
        # Missing v2 support must not send their bytes into Qwen M-RoPE ingest.
        if config.detected_model_family in (
            EDGEFLOW_MODEL_FAMILY_FASTVLM, EDGEFLOW_MODEL_FAMILY_LLAVA15,
        ):
            return _invoke_native_generate_from_descriptor_v2(
                image_path,
                prompt,
                visual_payload,
                int(max_new_tokens),
                config,
            )
        try:
            use_surface_path = (
                # Qwen's v1 C ABI has no dtype field: its bytes must be FP32.
                # CPU-staged FP16 still needs the explicit batch conversion.
                visual_payload.dtype == "float32"
                and config.ggml_native_surface_path_enabled
                and _surface_path_capability_available(config)
            )
        except BaseException:
            visual_payload.release()
            raise
        if not use_surface_path:
            if visual_payload.requires_native_surface_consumer or options:
                visual_payload.release()
                raise RuntimeError(
                    "SurfaceBackedVisualDescriptor data_plane='ane_output_surface' "
                    "requires native surface consumer support; refusing CPU readback fallback"
                )
            try:
                batch = visual_payload.materialize_to_batch()
            finally:
                # materialize_to_batch() releases the lease only on SUCCESS;
                # release() is idempotent, so this finally only matters when
                # it raises partway (dtype/size validation, read_into
                # failure). Mirrors the try/finally the descriptor-path
                # siblings use — without it the leaked lease starves a
                # capacity-1 LeasePool permanently.
                visual_payload.release()
            return _invoke_native_generate_from_batch(
                image_path,
                prompt,
                batch,
                int(max_new_tokens),
                config,
            )
        return _invoke_native_generate_from_descriptor(
            image_path,
            prompt,
            visual_payload,
            int(max_new_tokens),
            config,
            **options,
        )

    if isinstance(visual_payload, ImageEmbedsBatch):
        return _invoke_native_generate_from_batch(
            image_path,
            prompt,
            visual_payload,
            int(max_new_tokens),
            config,
        )

    raise TypeError(
        f"unsupported visual payload type: {type(visual_payload).__name__}"
    )


def _run_generate_native_from_batch(
    image_path: Path | str,
    prompt: str,
    batch: ImageEmbedsBatch,
    max_new_tokens: int,
    config: SplitConfig,
) -> GGMLGenerateResult:
    del image_path
    request = _prepare_batch_request_for_native(batch)
    request["model_path"] = config.ggml_model_path or ""
    request["prompt"] = prompt
    request["max_new_tokens"] = int(max_new_tokens)
    return _run_native_request(request, config)


def _run_generate_native_from_descriptor(
    image_path: Path | str,
    prompt: str,
    descriptor: SurfaceBackedVisualDescriptor,
    max_new_tokens: int,
    config: SplitConfig,
    *,
    on_token=None,
    ignore_eog=False,
) -> GGMLGenerateResult:
    try:
        del image_path
        if descriptor.dtype != "float32":
            raise ValueError(
                "Qwen v1 surface path requires dtype='float32', "
                f"got dtype={descriptor.dtype!r}"
            )
        request = _prepare_descriptor_request_for_native(descriptor)
        request["model_path"] = config.ggml_model_path or ""
        request["prompt"] = prompt
        request["max_new_tokens"] = int(max_new_tokens)
        with (_acquire_native_llm_runtime(config) as runtime,
              _generation_options(runtime, on_token, ignore_eog)):
            response = _invoke_native_surface_llm(runtime, request)
        text = response.get("text")
        if not isinstance(text, str):
            raise RuntimeError("native llama.cpp surface response missing text")
        return GGMLGenerateResult(
            text=text,
            first_token_ms=float(response["first_token_ms"]) if response.get("first_token_ms") is not None else None,
            prompt_eval_ms=float(response["prompt_eval_ms"]) if response.get("prompt_eval_ms") is not None else None,
            eval_ms=float(response["eval_ms"]) if response.get("eval_ms") is not None else None,
            eval_count=int(response["eval_count"]) if response.get("eval_count") is not None else None,
        )
    finally:
        descriptor.release()


def _run_generate_native_from_descriptor_v2(
    image_path: Path | str,
    prompt: str,
    descriptor: SurfaceBackedVisualDescriptor,
    max_new_tokens: int,
    config: SplitConfig,
) -> GGMLGenerateResult:
    """FastVLM/LLaVA-1.5 v2 surface-lease descriptor path.

    Mirrors `_run_generate_native_from_descriptor` (release-on-exit semantics)
    but routes through the v2 entry point with explicit model_family +
    visual_dtype. These descriptors must carry fp16 embeddings — anything
    else is rejected up front so we don't silently misinterpret the bytes.
    """
    try:
        del image_path
        family = config.detected_model_family
        names = {
            EDGEFLOW_MODEL_FAMILY_FASTVLM: "FastVLM",
            EDGEFLOW_MODEL_FAMILY_LLAVA15: "LLaVA-1.5",
        }
        if family not in names:
            raise ValueError("v2 descriptor path requires FastVLM or LLaVA-1.5")
        if descriptor.dtype != "float16":
            raise RuntimeError(
                f"{names[family]} v2 surface path requires dtype='float16', got dtype={descriptor.dtype!r}"
            )
        request = _prepare_descriptor_request_for_native(descriptor)
        request["model_path"] = config.ggml_model_path or ""
        request["prompt"] = prompt
        request["max_new_tokens"] = int(max_new_tokens)
        request["model_family"] = family
        request["visual_dtype"] = EDGEFLOW_DTYPE_FP16
        with _acquire_native_llm_runtime(config) as runtime:
            response = _invoke_native_surface_llm_v2(runtime, request)
        text = response.get("text")
        if not isinstance(text, str):
            raise RuntimeError("native llama.cpp surface v2 response missing text")
        return GGMLGenerateResult(
            text=text,
            first_token_ms=float(response["first_token_ms"]) if response.get("first_token_ms") is not None else None,
            prompt_eval_ms=float(response["prompt_eval_ms"]) if response.get("prompt_eval_ms") is not None else None,
            eval_ms=float(response["eval_ms"]) if response.get("eval_ms") is not None else None,
            eval_count=int(response["eval_count"]) if response.get("eval_count") is not None else None,
        )
    finally:
        descriptor.release()


# Preserve test and call-site patch points while routing implementation through
# the new batch/descriptor helpers.
_invoke_native_generate_from_batch = _run_generate_native_from_batch
_invoke_native_generate_from_descriptor = _run_generate_native_from_descriptor
_invoke_native_generate_from_descriptor_v2 = _run_generate_native_from_descriptor_v2


def _run_native_request(request: dict[str, object], config: SplitConfig) -> GGMLGenerateResult:
    with _acquire_native_llm_runtime(config) as runtime:
        if config.detected_model_family == EDGEFLOW_MODEL_FAMILY_QWEN25VL:
            response = _invoke_native_llm(runtime, request)
        else:
            request["model_family"] = config.detected_model_family
            response = _invoke_native_llm_v2(runtime, request)
    text = response.get("text")
    if not isinstance(text, str):
        raise RuntimeError("native llama.cpp response missing text")
    return GGMLGenerateResult(
        text=text,
        first_token_ms=float(response["first_token_ms"]) if response.get("first_token_ms") is not None else None,
        prompt_eval_ms=float(response["prompt_eval_ms"]) if response.get("prompt_eval_ms") is not None else None,
        eval_ms=float(response["eval_ms"]) if response.get("eval_ms") is not None else None,
        eval_count=int(response["eval_count"]) if response.get("eval_count") is not None else None,
    )


def _run_generate_native(
    image_path: Path | str,
    prompt: str,
    batch: ImageEmbedsBatch,
    max_new_tokens: int,
    config: SplitConfig,
) -> GGMLGenerateResult:
    return _run_generate_native_visual_payload(
        image_path,
        prompt,
        batch,
        int(max_new_tokens),
        config,
    )


def _run_generate_native_from_handle(
    image_path: Path | str,
    prompt: str,
    handle: SharedTensorHandle,
    max_new_tokens: int,
    config: SplitConfig,
) -> GGMLGenerateResult:
    del image_path
    request = _prepare_handle_request_for_native(handle)
    request["model_path"] = config.ggml_model_path or ""
    request["prompt"] = prompt
    request["max_new_tokens"] = int(max_new_tokens)
    return _run_native_request(request, config)


def run_split_model_generate_ggml(
    image_path: Path | str,
    prompt: str,
    batch: ImageEmbedsBatch,
    *,
    max_new_tokens: int = 20,
    config: SplitConfig | None = None,
) -> GGMLGenerateResult:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    cfg = config or SplitConfig()
    if cfg.ggml_transport != "native":
        raise ValueError("run_split_model_generate_ggml requires ggml_transport='native'")

    return _run_generate_native(image_path, prompt, batch, int(max_new_tokens), cfg)


def run_split_model_generate_ggml_from_handle(
    image_path: Path | str,
    prompt: str,
    handle: SharedTensorHandle,
    *,
    max_new_tokens: int = 20,
    config: SplitConfig | None = None,
) -> GGMLGenerateResult:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    cfg = config or SplitConfig()
    if cfg.ggml_transport != "native":
        raise ValueError("run_split_model_generate_ggml_from_handle requires ggml_transport='native'")
    return _run_generate_native_from_handle(image_path, prompt, handle, int(max_new_tokens), cfg)


def _resolve_native_library_path(config: SplitConfig) -> str:
    if config.ggml_library_path is not None:
        return config.ggml_library_path
    return _default_native_library_path()


def _default_native_library_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "tools" / "edgeflow_llm_runtime" / "build" / "libedgeflow_llm.dylib")

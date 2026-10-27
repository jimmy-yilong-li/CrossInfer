from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

from edgeflow_split.config import SplitConfig
from edgeflow_split.contracts import ImageEmbedsBatch
from edgeflow_split.ggml_transport_contracts import GGMLGenerateResult
from edgeflow_split.ggml_transport_contracts import prepare_batch_request_for_ggml
from edgeflow_split.ggml_transport_contracts import prepare_handle_request_for_ggml
from edgeflow_split.shared_tensor import SharedTensorHandle

# Frozen FFI transport path.
# Keep this implementation stable for rollback/compatibility only.
# Do not expand this module with new transport behavior.
_GGML_FFI_LIB_CACHE: dict[str, ctypes.CDLL] = {}


def _load_ggml_ffi_library(config: SplitConfig) -> ctypes.CDLL:
    path = _resolve_ggml_library_path(config)
    cached = _GGML_FFI_LIB_CACHE.get(path)
    if cached is not None:
        return cached
    lib = ctypes.CDLL(path)
    lib.edgeflow_ggml_generate_json.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_int,
        ctypes.c_double,
    ]
    lib.edgeflow_ggml_generate_json.restype = ctypes.c_void_p
    lib.edgeflow_ggml_free_string.argtypes = [ctypes.c_void_p]
    lib.edgeflow_ggml_free_string.restype = None
    _GGML_FFI_LIB_CACHE[path] = lib
    return lib


load_ggml_ffi_library = _load_ggml_ffi_library


def _invoke_ggml_ffi(lib, request: dict[str, object]) -> dict[str, object]:
    # The caller must pass the original request dict through unchanged. The
    # "_payload_array" field is an intentional GC pin: keeping the full dict on
    # the stack for the duration of this call keeps the numpy backing store
    # alive while Go reads request["payload_ptr"] synchronously.
    response_ptr = lib.edgeflow_ggml_generate_json(
        os.fsencode(str(request["model_path"])),
        os.fsencode(str(request["prompt"])),
        ctypes.c_void_p(int(request["payload_ptr"])),
        ctypes.c_int(int(request["payload_nbytes"])),
        ctypes.c_int(int(request["image_grid_thw"][0])),
        ctypes.c_int(int(request["image_grid_thw"][1])),
        ctypes.c_int(int(request["image_grid_thw"][2])),
        ctypes.c_int(int(request["num_visual_tokens"])),
        ctypes.c_int(int(request["hidden_size"])),
        os.fsencode(str(request["dtype"])),
        os.fsencode(str(request["payload_dtype"])),
        ctypes.c_int(int(request["max_new_tokens"])),
        ctypes.c_longlong(int(request["handle_id"])) if "handle_id" in request else ctypes.c_longlong(0),
        ctypes.c_longlong(int(request["buffer_id"])) if "buffer_id" in request else ctypes.c_longlong(0),
        ctypes.c_int(int(request["frame_id"])) if "frame_id" in request else ctypes.c_int(0),
        ctypes.c_double(float(request["timestamp_ms"])) if "timestamp_ms" in request else ctypes.c_double(0.0),
    )
    if not response_ptr:
        raise RuntimeError("ggml ffi returned a null response pointer")
    try:
        raw = ctypes.string_at(response_ptr)
    finally:
        lib.edgeflow_ggml_free_string(response_ptr)
    try:
        response = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("ggml ffi returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise RuntimeError("ggml ffi returned unexpected response shape")
    if response.get("error"):
        raise RuntimeError(str(response["error"]))
    return response


def _run_ffi_request(request: dict[str, object], config: SplitConfig) -> GGMLGenerateResult:
    response = _invoke_ggml_ffi(_load_ggml_ffi_library(config), request)
    if not isinstance(response, dict):
        raise RuntimeError("ggml ffi returned unexpected response shape")
    text = response.get("text")
    if not isinstance(text, str):
        raise RuntimeError("ggml ffi response missing text")
    return GGMLGenerateResult(
        text=text,
        first_token_ms=float(response["first_token_ms"]) if response.get("first_token_ms") is not None else None,
        prompt_eval_ms=float(response["prompt_eval_ms"]) if response.get("prompt_eval_ms") is not None else None,
        eval_ms=float(response["eval_ms"]) if response.get("eval_ms") is not None else None,
        eval_count=int(response["eval_count"]) if response.get("eval_count") is not None else None,
        ingress_copy_ms=float(response["ingress_copy_ms"]) if response.get("ingress_copy_ms") is not None else None,
        ingress_payload_bytes=int(response["ingress_payload_bytes"]) if response.get("ingress_payload_bytes") is not None else None,
    )


def _run_generate_ffi(
    image_path: Path | str,
    prompt: str,
    batch: ImageEmbedsBatch,
    max_new_tokens: int,
    config: SplitConfig,
) -> GGMLGenerateResult:
    del image_path
    batch_metadata, contiguous = prepare_batch_request_for_ggml(batch)
    request = {
        "model_path": config.ggml_model_path or "",
        "prompt": prompt,
        "max_new_tokens": int(max_new_tokens),
        "image_grid_thw": tuple(int(v) for v in batch.image_grid_thw),
        "num_visual_tokens": int(batch.num_visual_tokens),
        "hidden_size": int(batch.hidden_size),
        "dtype": str(batch.dtype),
        "payload_dtype": str(batch_metadata["payload_dtype"]),
        "payload_nbytes": int(contiguous.nbytes),
        "payload_ptr": int(contiguous.ctypes.data),
        "source_image_size": tuple(batch.source_image_size) if batch.source_image_size is not None else None,
        "model_revision": batch.model_revision,
        # Keep a strong Python reference for the duration of the ctypes call so
        # the underlying numpy buffer cannot be garbage-collected while Go reads it.
        "_payload_array": contiguous,
        "_batch_metadata": batch_metadata,
    }
    return _run_ffi_request(request, config)


def _run_generate_ffi_from_handle(
    image_path: Path | str,
    prompt: str,
    handle: SharedTensorHandle,
    max_new_tokens: int,
    config: SplitConfig,
) -> GGMLGenerateResult:
    del image_path
    metadata, contiguous = prepare_handle_request_for_ggml(handle)
    request = {
        "model_path": config.ggml_model_path or "",
        "prompt": prompt,
        "max_new_tokens": int(max_new_tokens),
        "image_grid_thw": tuple(int(v) for v in handle.image_grid_thw),
        "num_visual_tokens": int(handle.token_count),
        "hidden_size": int(handle.hidden_size),
        "shape": tuple(int(v) for v in metadata["shape"]),
        "dtype": str(handle.dtype),
        "payload_dtype": str(metadata["payload_dtype"]),
        "payload_nbytes": int(contiguous.nbytes),
        "payload_ptr": int(contiguous.ctypes.data),
        "source_image_size": tuple(handle.source_image_size) if handle.source_image_size is not None else None,
        "model_revision": handle.model_revision,
        "handle_id": int(handle.handle_id),
        "buffer_id": int(handle.buffer_id),
        "frame_id": int(handle.frame_id),
        "timestamp_ms": float(handle.timestamp_ms),
        # Keep a strong Python reference for the duration of the ctypes call so
        # the underlying numpy buffer cannot be garbage-collected before Go
        # consumes the payload pointer.
        "_payload_array": contiguous,
        "_handle": handle,
        "_descriptor_metadata": metadata,
    }
    return _run_ffi_request(request, config)


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
    if cfg.ggml_transport != "ffi":
        raise ValueError("run_split_model_generate_ggml_from_handle requires ggml_transport='ffi'")
    return _run_generate_ffi_from_handle(image_path, prompt, handle, int(max_new_tokens), cfg)


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
    if cfg.ggml_transport != "ffi":
        raise ValueError("run_split_model_generate_ggml requires ggml_transport='ffi'")
    return _run_generate_ffi(image_path, prompt, batch, int(max_new_tokens), cfg)


def _resolve_ggml_library_path(config: SplitConfig) -> str:
    if config.ggml_library_path is not None:
        return config.ggml_library_path
    return _default_ggml_library_path()


def _default_ggml_library_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "tools" / "ggml_text_backend" / "libedgeflow_ggml.dylib")

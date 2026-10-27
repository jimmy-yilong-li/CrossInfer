from __future__ import annotations

from edgeflow_split.config import SplitConfig
from edgeflow_split.contracts import ImageEmbedsBatch
from edgeflow_split.ggml_transport_contracts import as_numpy_and_bytes_dtype
from edgeflow_split.ggml_transport_contracts import GGMLGenerateResult
from edgeflow_split.ggml_transport_contracts import prepare_batch_request_for_ggml
from edgeflow_split.ggml_transport_contracts import prepare_handle_request_for_ggml
from edgeflow_split.ggml_transport_contracts import serialize_batch_request_for_ggml
from edgeflow_split.shared_tensor import SharedTensorHandle


def _as_numpy_and_bytes_dtype(value, dtype: str) -> tuple[np.ndarray, str]:
    return as_numpy_and_bytes_dtype(value, dtype)


def _prepare_batch_request_for_ggml(batch: ImageEmbedsBatch) -> tuple[dict[str, object], np.ndarray]:
    return prepare_batch_request_for_ggml(batch)


def _serialize_batch_request_for_ggml(batch: ImageEmbedsBatch) -> tuple[dict[str, object], bytes]:
    return serialize_batch_request_for_ggml(batch)


def _prepare_handle_request_for_ggml(handle: SharedTensorHandle) -> tuple[dict[str, object], np.ndarray]:
    return prepare_handle_request_for_ggml(handle)


def _ggml_http_module():
    from edgeflow_split import _ggml_transport_http as module

    return module


def _ggml_ffi_module():
    from edgeflow_split import _ggml_transport_ffi as module

    return module


def _ggml_native_module():
    from edgeflow_split import _ggml_transport_native as module

    return module


def check_ggml_sidecar_health(config: SplitConfig | None = None) -> dict[str, object]:
    return _ggml_http_module().check_ggml_sidecar_health(config)


def serialize_batch_for_ggml(batch: ImageEmbedsBatch) -> dict[str, object]:
    return _ggml_http_module().serialize_batch_for_ggml(batch)


def serialize_batch_payload_for_ggml(batch: ImageEmbedsBatch) -> bytes:
    return _ggml_http_module().serialize_batch_payload_for_ggml(batch)


def run_split_model_generate_ggml(
    image_path,
    prompt: str,
    batch: ImageEmbedsBatch,
    *,
    max_new_tokens: int = 20,
    config: SplitConfig | None = None,
) -> GGMLGenerateResult:
    cfg = config or SplitConfig()
    if cfg.ggml_transport == "ffi":
        return _ggml_ffi_module().run_split_model_generate_ggml(
            image_path,
            prompt,
            batch,
            max_new_tokens=max_new_tokens,
            config=cfg,
        )
    if cfg.ggml_transport == "native":
        return _ggml_native_module().run_split_model_generate_ggml(
            image_path,
            prompt,
            batch,
            max_new_tokens=max_new_tokens,
            config=cfg,
        )
    return _ggml_http_module().run_split_model_generate_ggml(
        image_path,
        prompt,
        batch,
        max_new_tokens=max_new_tokens,
        config=cfg,
    )


def run_split_model_generate_ggml_from_handle(
    image_path,
    prompt: str,
    handle: SharedTensorHandle,
    *,
    max_new_tokens: int = 20,
    config: SplitConfig | None = None,
) -> GGMLGenerateResult:
    cfg = config or SplitConfig()
    if cfg.ggml_transport == "native":
        return _ggml_native_module().run_split_model_generate_ggml_from_handle(
            image_path,
            prompt,
            handle,
            max_new_tokens=max_new_tokens,
            config=cfg,
        )
    if cfg.ggml_transport == "ffi":
        return _ggml_ffi_module().run_split_model_generate_ggml_from_handle(
            image_path,
            prompt,
            handle,
            max_new_tokens=max_new_tokens,
            config=cfg,
        )
    raise ValueError("run_split_model_generate_ggml_from_handle requires ggml_transport='ffi' or 'native'")

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from edgeflow_split.config import SplitConfig
from edgeflow_split.contracts import ImageEmbedsBatch
from edgeflow_split.ggml_transport_contracts import GGMLGenerateResult
from edgeflow_split.ggml_transport_contracts import serialize_batch_request_for_ggml

# Frozen baseline HTTP rollback paths.
# Keep binary and shared-memory support only for baseline/rollback compatibility.
# Do not add new transport investment here.
_EDGEFLOW_META_HEADER = "X-Edgeflow-Meta"


def _sidecar_url(config: SplitConfig, path: str) -> str:
    base = config.ggml_sidecar_url.rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


def serialize_batch_for_ggml(batch: ImageEmbedsBatch) -> dict[str, object]:
    metadata, _ = serialize_batch_request_for_ggml(batch)
    return metadata


def serialize_batch_payload_for_ggml(batch: ImageEmbedsBatch) -> bytes:
    _, payload = serialize_batch_request_for_ggml(batch)
    return payload


def check_ggml_sidecar_health(config: SplitConfig | None = None) -> dict[str, object]:
    cfg = config or SplitConfig()
    return _post_json(_sidecar_url(cfg, "/v1/health"), {}, cfg.ggml_timeout_s)


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
    if cfg.ggml_transport not in {"binary", "shm"}:
        raise ValueError("run_split_model_generate_ggml requires ggml_transport='binary' or 'shm'")

    batch_metadata, payload_bytes = serialize_batch_request_for_ggml(batch)
    payload = {
        "model_id": cfg.model_id,
        "ggml_model_path": cfg.ggml_model_path,
        "image_path": str(image_path),
        "prompt": prompt,
        "max_new_tokens": int(max_new_tokens),
        "batch": batch_metadata,
    }
    if cfg.ggml_transport == "binary":
        response = _post_binary(
            _sidecar_url(cfg, "/v1/generate"),
            payload,
            payload_bytes,
            cfg.ggml_timeout_s,
        )
    else:
        # Frozen baseline shared-memory transport kept for rollback compatibility only.
        shm_path = _write_shared_memory_payload(payload_bytes, cfg)
        payload["batch"]["transport"] = "shm"
        payload["batch"]["shm_path"] = shm_path
        try:
            response = _post_binary(
                _sidecar_url(cfg, "/v1/generate"),
                payload,
                b"",
                cfg.ggml_timeout_s,
            )
        finally:
            try:
                os.unlink(shm_path)
            except FileNotFoundError:
                pass

    text = response.get("text")
    if not isinstance(text, str):
        raise RuntimeError("ggml sidecar response missing text")
    return GGMLGenerateResult(
        text=text,
        first_token_ms=float(response["first_token_ms"]) if response.get("first_token_ms") is not None else None,
        prompt_eval_ms=float(response["prompt_eval_ms"]) if response.get("prompt_eval_ms") is not None else None,
        eval_ms=float(response["eval_ms"]) if response.get("eval_ms") is not None else None,
        eval_count=int(response["eval_count"]) if response.get("eval_count") is not None else None,
        ingress_copy_ms=float(response["ingress_copy_ms"]) if response.get("ingress_copy_ms") is not None else None,
        ingress_payload_bytes=int(response["ingress_payload_bytes"]) if response.get("ingress_payload_bytes") is not None else None,
    )


def _post_json(url: str, payload: dict[str, object], timeout_s: float) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ggml sidecar request failed ({exc.code}): {body}") from exc
    except urllib_error.URLError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise RuntimeError(f"ggml sidecar unavailable at {url}: {exc.reason}") from exc

    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ggml sidecar returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("ggml sidecar returned unexpected response shape")
    return result


def _post_binary(url: str, metadata: dict[str, object], payload: bytes, timeout_s: float) -> dict[str, object]:
    req = urllib_request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/octet-stream",
            _EDGEFLOW_META_HEADER: json.dumps(metadata),
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:  # pragma: no cover
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ggml sidecar request failed ({exc.code}): {body}") from exc
    except urllib_error.URLError as exc:  # pragma: no cover
        raise RuntimeError(f"ggml sidecar unavailable at {url}: {exc.reason}") from exc

    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ggml sidecar returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("ggml sidecar returned unexpected response shape")
    return result


def _write_shared_memory_payload(payload: bytes, config: SplitConfig) -> str:
    os.makedirs(config.ggml_shared_memory_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(
        prefix="edgeflow_ggml_",
        suffix=".bin",
        dir=config.ggml_shared_memory_dir,
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
    except Exception:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise
    return path

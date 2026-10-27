from __future__ import annotations

import codecs
import ctypes
import hashlib
import os
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

from edgeflow_split.model_spec import ModelFamily

LLM_ABI_VERSION = 1
SESSION_ABI_VERSION = 3
_SHA256_HEX_SIZE = 65
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


class PositionContract(IntEnum):
    TEXT_1D = 0
    QWEN25VL_MROPE = 1
    QWEN35_IMROPE = 2


class SessionModelFamily(IntEnum):
    """Additive session families; existing shared family values stay unchanged."""

    QWEN25VL = int(ModelFamily.QWEN25VL)
    FASTVLM = int(ModelFamily.FASTVLM)
    LLAVA15 = int(ModelFamily.LLAVA15)
    QWEN35 = 3


class KVDataType(IntEnum):
    FP16 = 0


class BackendPlacement(IntEnum):
    CPU_ONLY = 0
    GPU_FULL = 1
    MIXED = 2


class _SessionParams(ctypes.Structure):
    _fields_ = [
        ("n_gpu_layers", ctypes.c_int),
        ("n_ctx", ctypes.c_int),
        ("n_batch", ctypes.c_int),
        ("n_ubatch", ctypes.c_int),
        ("n_seq_max", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("kv_unified", ctypes.c_int),
        ("model_family", ctypes.c_int),
        ("kv_type_k", ctypes.c_int),
        ("kv_type_v", ctypes.c_int),
        ("position_contract", ctypes.c_int),
        ("prompt_contract_version", ctypes.c_uint32),
    ]


class _SessionSignature(ctypes.Structure):
    _fields_ = [
        ("session_abi_version", ctypes.c_uint32),
        ("requested_gpu_layers", ctypes.c_int),
        ("observed_gpu_layers", ctypes.c_int),
        ("backend_placement", ctypes.c_int),
        ("n_ctx", ctypes.c_int),
        ("n_batch", ctypes.c_int),
        ("n_ubatch", ctypes.c_int),
        ("n_seq_max", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("kv_unified", ctypes.c_int),
        ("model_family", ctypes.c_int),
        ("kv_type_k", ctypes.c_int),
        ("kv_type_v", ctypes.c_int),
        ("position_contract", ctypes.c_int),
        ("prompt_contract_version", ctypes.c_uint32),
        ("model_sha256", ctypes.c_char * _SHA256_HEX_SIZE),
        ("runtime_sha256", ctypes.c_char * _SHA256_HEX_SIZE),
        (
            "runtime_source_fingerprint_sha256",
            ctypes.c_char * _SHA256_HEX_SIZE,
        ),
    ]


class _SequenceInfo(ctypes.Structure):
    _fields_ = [
        ("active", ctypes.c_int),
        ("next_position", ctypes.c_int),
        ("has_logits", ctypes.c_int),
        ("state_bytes", ctypes.c_uint64),
    ]


class _DecodeStepResult(ctypes.Structure):
    _fields_ = [
        ("token_id", ctypes.c_int),
        ("token_position", ctypes.c_int),
        ("is_eog", ctypes.c_int),
        ("next_token_id", ctypes.c_int),
        ("n_vocab", ctypes.c_int),
    ]


class _QwenRequest(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("visual_embd", ctypes.POINTER(ctypes.c_float)),
        ("num_visual_tokens", ctypes.c_int),
        ("hidden_size", ctypes.c_int),
        ("image_grid_thw", ctypes.c_int * 3),
        ("max_new_tokens", ctypes.c_int),
    ]


class _SurfaceAppendReceipt(ctypes.Structure):
    _fields_ = [
        ("surface_pointer", ctypes.c_uint64),
        ("decoder_pointer", ctypes.c_uint64),
        ("payload_bytes", ctypes.c_uint64),
        ("next_position", ctypes.c_int),
    ]


@dataclass(frozen=True)
class SurfaceAppendReceipt:
    surface_pointer: int
    decoder_pointer: int
    payload_bytes: int
    next_position: int


@dataclass(frozen=True)
class SessionConfig:
    model_path: Path
    library_path: Path
    n_gpu_layers: int = 999
    n_ctx: int = 4096
    n_batch: int = 512
    n_ubatch: int = 512
    n_seq_max: int = 4
    n_threads: int = 4
    model_family: ModelFamily | SessionModelFamily = ModelFamily.QWEN25VL
    position_contract: PositionContract = PositionContract.TEXT_1D
    prompt_contract_version: int = 1
    kv_type_k: KVDataType = KVDataType.FP16
    kv_type_v: KVDataType = KVDataType.FP16


@dataclass(frozen=True)
class StateCompatibilitySignature:
    session_abi_version: int
    model_sha256: str
    runtime_sha256: str
    runtime_source_fingerprint_sha256: str
    requested_gpu_layers: int
    observed_gpu_layers: int
    backend_placement: BackendPlacement
    n_ctx: int
    n_batch: int
    n_ubatch: int
    n_seq_max: int
    n_threads: int
    kv_unified: bool
    model_family: ModelFamily | SessionModelFamily
    kv_type_k: KVDataType
    kv_type_v: KVDataType
    position_contract: PositionContract
    prompt_contract_version: int


@dataclass(frozen=True, eq=False)
class RootStateSnapshot:
    """Owned native root state with immutable source metadata."""

    _lib: ctypes.CDLL = field(repr=False)
    _handle: ctypes.c_void_p | None = field(repr=False)
    source_signature: StateCompatibilitySignature
    source_session_incarnation_id: str
    bound_stream_id: str | None
    next_position: int | None

    def _require_open(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise RuntimeError("root state snapshot is closed")
        return self._handle

    @property
    def size_bytes(self) -> int:
        return int(self._lib.edgeflow_llm_root_snapshot_size(self._require_open()))

    def same_state(self, other: RootStateSnapshot) -> bool:
        """Compare native KV bytes and position metadata outside timed execution."""
        left, right = self._require_open(), other._require_open()
        compare = getattr(self._lib, "edgeflow_llm_root_snapshot_equal", None)
        if compare is None:
            raise NotImplementedError("native runtime does not support snapshot comparison")
        if self.source_signature.runtime_sha256 != other.source_signature.runtime_sha256:
            return False
        result = int(compare(left, right))
        if result < 0:
            raise RuntimeError(f"native snapshot comparison failed with status {result}")
        return bool(result)

    def close(self) -> None:
        if self._handle is None:
            return
        handle = self._handle
        object.__setattr__(self, "_handle", None)
        self._lib.edgeflow_llm_root_snapshot_free(handle)

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self.close()


@dataclass(frozen=True)
class SequenceInfo:
    sequence_id: int
    active: bool
    next_position: int
    has_logits: bool
    state_bytes: int


@dataclass(frozen=True)
class DecodeStep:
    sequence_id: int
    token_id: int
    token_position: int
    is_eog: bool
    next_token_id: int
    logits: NDArray[np.float32] | None


class TokenPieceStreamDecoder:
    """Decode native token bytes without corrupting split UTF-8 sequences."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._finished = False

    def feed(self, piece: bytes) -> str:
        if self._finished:
            raise RuntimeError("token-piece decoder is already finished")
        if not isinstance(piece, bytes):
            raise TypeError("token piece must be bytes")
        return self._decoder.decode(piece, final=False)

    def finish(self) -> str:
        if self._finished:
            return ""
        self._finished = True
        return self._decoder.decode(b"", final=True)


def _encode_native_text(value: str, *, label: str) -> bytes:
    if "\0" in value:
        raise ValueError(f"{label} must not contain NUL characters")
    return value.encode("utf-8")


def _checked_integer(
    value: object, *, label: str, minimum: int = 0, maximum: int = _INT32_MAX
) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _prepare_session_params(config: SessionConfig) -> _SessionParams:
    fields = {
        "n_gpu_layers": (_INT32_MIN, _INT32_MAX),
        "n_ctx": (1, _INT32_MAX),
        "n_batch": (1, _INT32_MAX),
        "n_ubatch": (1, _INT32_MAX),
        "n_seq_max": (2, _INT32_MAX),
        "n_threads": (1, _INT32_MAX),
        "model_family": (0, max(SessionModelFamily)),
        "kv_type_k": (0, 0),
        "kv_type_v": (0, 0),
        "position_contract": (0, max(PositionContract)),
        "prompt_contract_version": (1, _UINT32_MAX),
    }
    values = {
        name: _checked_integer(
            getattr(config, name), label=name, minimum=minimum, maximum=maximum
        )
        for name, (minimum, maximum) in fields.items()
    }
    if values["n_ubatch"] > values["n_batch"]:
        raise ValueError("n_ubatch must not exceed n_batch")
    qwen35 = values["model_family"] == SessionModelFamily.QWEN35
    if qwen35 != (values["position_contract"] == PositionContract.QWEN35_IMROPE) or (
        qwen35 and values["prompt_contract_version"] != 1
    ):
        raise ValueError("Qwen3.5 requires its IMRoPE and prompt-v1 contract")
    return _SessionParams(kv_unified=1, **values)


def _validated_session_inputs(
    config: SessionConfig,
) -> tuple[_SessionParams, Path, Path]:
    params = _prepare_session_params(config)
    model_path = config.model_path.expanduser().resolve()
    library_path = config.library_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"stateful session model does not exist: {model_path}")
    if not library_path.is_file():
        raise FileNotFoundError(f"stateful session library does not exist: {library_path}")
    return params, model_path, library_path


def validate_session_config(config: SessionConfig) -> None:
    """Check parameters and artifact paths without hashing, loading or allocating."""
    _validated_session_inputs(config)


def _prepare_qwen_request(
    *,
    prompt: str | None,
    visual_embeddings: object,
    image_grid_thw: tuple[int, int, int],
) -> tuple[object, _QwenRequest]:
    if len(image_grid_thw) != 3:
        raise ValueError("image_grid_thw must contain exactly three integers")
    grid = tuple(
        _checked_integer(value, label="image_grid_thw", minimum=1)
        for value in image_grid_thw
    )
    import numpy as np

    visual = np.ascontiguousarray(visual_embeddings, dtype=np.float32)
    if visual.ndim != 2 or visual.shape[0] <= 0 or visual.shape[1] <= 0:
        raise ValueError("visual_embeddings must be a non-empty rank-2 array")
    request = _QwenRequest(
        prompt=(
            _encode_native_text(prompt, label="prompt") if prompt is not None else None
        ),
        visual_embd=visual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        num_visual_tokens=_checked_integer(
            visual.shape[0], label="num_visual_tokens", minimum=1
        ),
        hidden_size=_checked_integer(visual.shape[1], label="hidden_size", minimum=1),
        image_grid_thw=(ctypes.c_int * 3)(*grid),
        max_new_tokens=0,
    )
    return visual, request


@lru_cache(maxsize=16)
def _sha256_file_content(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_file_content(os.fspath(path), stat.st_size, stat.st_mtime_ns)


def _uint32_function(lib: ctypes.CDLL, name: str):
    fn = getattr(lib, name, None)
    if fn is None:
        raise RuntimeError(f"stateful session ABI is missing symbol: {name}")
    fn.argtypes = []
    fn.restype = ctypes.c_uint32
    return fn


def _bind_session_library(lib: ctypes.CDLL) -> None:
    llm_version = int(_uint32_function(lib, "edgeflow_llm_abi_version")())
    if llm_version != LLM_ABI_VERSION:
        raise RuntimeError(
            f"native llama.cpp ABI mismatch: expected {LLM_ABI_VERSION}, "
            f"got {llm_version}"
        )
    version = int(_uint32_function(lib, "edgeflow_llm_session_abi_version")())
    if version != SESSION_ABI_VERSION:
        raise RuntimeError(
            f"stateful session ABI mismatch: expected {SESSION_ABI_VERSION}, got {version}"
        )
    expected_sizes = {
        "edgeflow_llm_request_sizeof": ctypes.sizeof(_QwenRequest),
        "edgeflow_llm_session_params_sizeof": ctypes.sizeof(_SessionParams),
        "edgeflow_llm_session_signature_sizeof": ctypes.sizeof(_SessionSignature),
        "edgeflow_llm_sequence_info_sizeof": ctypes.sizeof(_SequenceInfo),
        "edgeflow_llm_decode_step_result_sizeof": ctypes.sizeof(_DecodeStepResult),
    }
    for symbol, expected in expected_sizes.items():
        actual = int(_uint32_function(lib, symbol)())
        if actual != expected:
            raise RuntimeError(
                f"stateful session ABI mismatch: {symbol} expected {expected}, got {actual}"
            )

    lib.edgeflow_llm_session_init.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(_SessionParams),
    ]
    lib.edgeflow_llm_session_init.restype = ctypes.c_void_p
    lib.edgeflow_llm_session_free.argtypes = [ctypes.c_void_p]
    lib.edgeflow_llm_session_free.restype = None
    lib.edgeflow_llm_session_last_init_error.argtypes = []
    lib.edgeflow_llm_session_last_init_error.restype = ctypes.c_char_p
    lib.edgeflow_llm_session_last_error.argtypes = [ctypes.c_void_p]
    lib.edgeflow_llm_session_last_error.restype = ctypes.c_char_p
    lib.edgeflow_llm_session_get_signature.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_SessionSignature),
    ]
    lib.edgeflow_llm_session_get_signature.restype = ctypes.c_int
    lib.edgeflow_llm_session_vocab_size.argtypes = [ctypes.c_void_p]
    lib.edgeflow_llm_session_vocab_size.restype = ctypes.c_int
    lib.edgeflow_llm_session_get_sequence_info.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(_SequenceInfo),
    ]
    lib.edgeflow_llm_session_get_sequence_info.restype = ctypes.c_int
    lib.edgeflow_llm_session_ingest_text.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.edgeflow_llm_session_ingest_text.restype = ctypes.c_int
    lib.edgeflow_llm_session_ingest_user_prompt.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.edgeflow_llm_session_ingest_user_prompt.restype = ctypes.c_int
    lib.edgeflow_llm_session_ingest_tokens.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
    ]
    lib.edgeflow_llm_session_ingest_tokens.restype = ctypes.c_int
    lib.edgeflow_llm_session_ingest_qwen25vl.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(_QwenRequest),
    ]
    lib.edgeflow_llm_session_ingest_qwen25vl.restype = ctypes.c_int
    lib.edgeflow_llm_session_ingest_qwen25vl_visual_prefix.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(_QwenRequest),
    ]
    lib.edgeflow_llm_session_ingest_qwen25vl_visual_prefix.restype = ctypes.c_int
    append = getattr(lib, "edgeflow_llm_session_append_qwen25vl_visual", None)
    if append is not None:
        append.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_QwenRequest)]
        append.restype = ctypes.c_int
    lib.edgeflow_llm_session_ingest_qwen25vl_query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.edgeflow_llm_session_ingest_qwen25vl_query.restype = ctypes.c_int
    lib.edgeflow_llm_session_fork_prefix.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.edgeflow_llm_session_fork_prefix.restype = ctypes.c_int
    peek = getattr(lib, "edgeflow_llm_session_peek_token", None)
    if peek is not None:
        peek.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        peek.restype = ctypes.c_int
    lib.edgeflow_llm_session_decode_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(_DecodeStepResult),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.edgeflow_llm_session_decode_step.restype = ctypes.c_int
    lib.edgeflow_llm_session_token_piece.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.edgeflow_llm_session_token_piece.restype = ctypes.c_int
    lib.edgeflow_llm_session_remove_sequence.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.edgeflow_llm_session_remove_sequence.restype = ctypes.c_int

    # Root-state transfer is an optional extension of session ABI 3.
    root_functions = {
        "edgeflow_llm_session_export_root": (
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int
        ),
        "edgeflow_llm_session_import_root": (
            [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int
        ),
        "edgeflow_llm_root_snapshot_free": ([ctypes.c_void_p], None),
        "edgeflow_llm_root_snapshot_size": ([ctypes.c_void_p], ctypes.c_uint64),
        "edgeflow_llm_root_snapshot_equal": (
            [ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int
        ),
        "edgeflow_llm_session_reset_root": ([ctypes.c_void_p], ctypes.c_int),
        "edgeflow_llm_session_set_ignore_eog": (
            [ctypes.c_void_p, ctypes.c_int], ctypes.c_int
        ),
    }
    for name, (argtypes, restype) in root_functions.items():
        fn = getattr(lib, name, None)
        if fn is not None:
            fn.argtypes = argtypes
            fn.restype = restype

    for family in ("qwen25vl", "qwen35"):
        append = getattr(lib, f"edgeflow_llm_session_append_{family}_visual_from_surface_lease", None)
        if append is None:
            continue
        size = lib.edgeflow_llm_surface_append_receipt_sizeof
        size.argtypes, size.restype = [], ctypes.c_uint32
        if int(size()) != ctypes.sizeof(_SurfaceAppendReceipt):
            raise RuntimeError("native surface append receipt ABI mismatch")
        append.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_QwenRequest),
                           ctypes.c_void_p, ctypes.POINTER(_SurfaceAppendReceipt)]
        append.restype = ctypes.c_int
    query = getattr(lib, "edgeflow_llm_session_ingest_qwen35_query", None)
    if query is not None:
        query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p,
                          ctypes.POINTER(ctypes.c_int)]
        query.restype = ctypes.c_int


class StatefulLLMSession:
    """Single-thread owner for local unified-KV sequence forks.

    Operations remain time-sliced, not continuously batched. Optional native
    root snapshots support cross-placement handoff with native format validation.
    """

    def __init__(self, config: SessionConfig) -> None:
        params, model_path, library_path = _validated_session_inputs(config)

        self._owner_thread = threading.get_ident()
        self._model_path = model_path
        self._lib = ctypes.CDLL(os.fspath(library_path))
        _bind_session_library(self._lib)
        runtime_sha256 = _sha256_file(library_path)
        raw_ctx = self._lib.edgeflow_llm_session_init(
            os.fsencode(model_path), ctypes.byref(params)
        )
        if not raw_ctx:
            error = self._lib.edgeflow_llm_session_last_init_error()
            reason = (
                error.decode("utf-8", errors="replace") if error else "unknown error"
            )
            raise RuntimeError(f"stateful session initialization failed: {reason}")
        self._ctx = ctypes.c_void_p(raw_ctx)
        self._runtime_sha256 = runtime_sha256
        try:
            self._signature = self._read_signature()
            self._vocab_size = self._read_vocab_size()
        except BaseException:
            self._lib.edgeflow_llm_session_free(self._ctx)
            self._ctx = None
            raise
        self._session_incarnation_id = uuid.uuid4().hex
        self._bound_stream_id: str | None = None

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("stateful session must be used by its creating thread")
        if self._ctx is None:
            raise RuntimeError("stateful session is closed")

    def _raise_for_status(self, status: int, operation: str) -> None:
        if status == 0:
            return
        raw = self._lib.edgeflow_llm_session_last_error(self._ctx)
        reason = raw.decode("utf-8", errors="replace") if raw else "unknown error"
        raise RuntimeError(f"{operation} failed with status {status}: {reason}")

    def _read_signature(self) -> StateCompatibilitySignature:
        raw = _SessionSignature()
        status = int(
            self._lib.edgeflow_llm_session_get_signature(self._ctx, ctypes.byref(raw))
        )
        self._raise_for_status(status, "get session signature")
        model_sha256 = bytes(raw.model_sha256).split(b"\0", 1)[0].decode("ascii")
        runtime_sha256 = bytes(raw.runtime_sha256).split(b"\0", 1)[0].decode("ascii")
        runtime_source_fingerprint_sha256 = (
            bytes(raw.runtime_source_fingerprint_sha256)
            .split(b"\0", 1)[0]
            .decode("ascii")
        )
        for label, digest in (
            ("model", model_sha256),
            ("runtime binary", runtime_sha256),
            ("runtime source fingerprint", runtime_source_fingerprint_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest.lower()
            ):
                raise RuntimeError(
                    f"native session returned an invalid {label} identity"
                )
        actual_model_sha256 = _sha256_file(self._model_path)
        if model_sha256 != actual_model_sha256:
            raise RuntimeError(
                "native session model identity does not match the loaded GGUF"
            )
        if runtime_sha256 != self._runtime_sha256:
            raise RuntimeError(
                "native session runtime identity does not match the loaded dylib"
            )
        return StateCompatibilitySignature(
            session_abi_version=int(raw.session_abi_version),
            model_sha256=model_sha256,
            runtime_sha256=runtime_sha256,
            runtime_source_fingerprint_sha256=runtime_source_fingerprint_sha256,
            requested_gpu_layers=int(raw.requested_gpu_layers),
            observed_gpu_layers=int(raw.observed_gpu_layers),
            backend_placement=BackendPlacement(raw.backend_placement),
            n_ctx=int(raw.n_ctx),
            n_batch=int(raw.n_batch),
            n_ubatch=int(raw.n_ubatch),
            n_seq_max=int(raw.n_seq_max),
            n_threads=int(raw.n_threads),
            kv_unified=bool(raw.kv_unified),
            model_family=(SessionModelFamily.QWEN35 if raw.model_family == 3
                          else ModelFamily(raw.model_family)),
            kv_type_k=KVDataType(raw.kv_type_k),
            kv_type_v=KVDataType(raw.kv_type_v),
            position_contract=PositionContract(raw.position_contract),
            prompt_contract_version=int(raw.prompt_contract_version),
        )

    def _read_vocab_size(self) -> int:
        value = int(self._lib.edgeflow_llm_session_vocab_size(self._ctx))
        if value < 0:
            self._raise_for_status(-value, "get session vocabulary size")
        if value == 0:
            raise RuntimeError("native session returned an invalid vocabulary size: 0")
        return value

    @property
    def signature(self) -> StateCompatibilitySignature:
        return self._signature

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def session_incarnation_id(self) -> str:
        return self._session_incarnation_id

    def bind_stream(self, stream_id: str) -> None:
        """Permanently bind this native context's root KV to one logical stream."""
        self._require_owner()
        if not stream_id:
            raise ValueError("stream_id must not be empty")
        if self._bound_stream_id is None:
            self._bound_stream_id = stream_id
            return
        if self._bound_stream_id != stream_id:
            raise RuntimeError(
                f"session is already bound to stream {self._bound_stream_id}; "
                "destroy and recreate it before changing streams"
            )

    def _optional_function(self, name: str):
        fn = getattr(self._lib, name, None)
        if fn is None:
            raise NotImplementedError(f"native library does not support {name}")
        return fn

    def export_root_state(self) -> RootStateSnapshot:
        self._require_owner()
        export = self._optional_function("edgeflow_llm_session_export_root")
        self._optional_function("edgeflow_llm_root_snapshot_free")
        self._optional_function("edgeflow_llm_root_snapshot_size")
        try:
            next_position = self.sequence_info(0).next_position
        except (RuntimeError, NotImplementedError):
            next_position = None
        handle = ctypes.c_void_p()
        status = int(export(self._ctx, ctypes.byref(handle)))
        self._raise_for_status(status, "export root state")
        if not handle.value:
            raise RuntimeError("native root export returned a null snapshot")
        return RootStateSnapshot(
            _lib=self._lib,
            _handle=handle,
            source_signature=self.signature,
            source_session_incarnation_id=self.session_incarnation_id,
            bound_stream_id=self._bound_stream_id,
            next_position=next_position,
        )

    def import_root_state(self, snapshot: RootStateSnapshot) -> None:
        self._require_owner()
        import_root = self._optional_function("edgeflow_llm_session_import_root")
        if not isinstance(snapshot, RootStateSnapshot):
            raise TypeError("snapshot must be a RootStateSnapshot")
        handle = snapshot._require_open()
        if (
            self._bound_stream_id is not None
            and self._bound_stream_id != snapshot.bound_stream_id
        ):
            raise RuntimeError(
                f"session is already bound to stream {self._bound_stream_id}; "
                "reset root before importing a different or unbound stream"
            )
        status = int(import_root(self._ctx, handle))
        self._raise_for_status(status, "import root state")
        self._session_incarnation_id = uuid.uuid4().hex
        self._bound_stream_id = snapshot.bound_stream_id

    def reset_root(self) -> None:
        self._require_owner()
        reset = self._optional_function("edgeflow_llm_session_reset_root")
        status = int(reset(self._ctx))
        self._raise_for_status(status, "reset root")
        self._session_incarnation_id = uuid.uuid4().hex
        self._bound_stream_id = None

    def sequence_info(self, sequence_id: int) -> SequenceInfo:
        self._require_owner()
        sequence_id = self._checked_sequence_id(sequence_id)
        raw = _SequenceInfo()
        status = int(
            self._lib.edgeflow_llm_session_get_sequence_info(
                self._ctx, sequence_id, ctypes.byref(raw)
            )
        )
        self._raise_for_status(status, "get sequence info")
        return SequenceInfo(
            sequence_id=sequence_id,
            active=bool(raw.active),
            next_position=int(raw.next_position),
            has_logits=bool(raw.has_logits),
            state_bytes=int(raw.state_bytes),
        )

    def ingest_text(
        self,
        sequence_id: int,
        text: str,
        *,
        add_special: bool = False,
        parse_special: bool = True,
    ) -> int:
        self._require_owner()
        sequence_id = self._checked_sequence_id(sequence_id)
        if not text:
            raise ValueError("text must not be empty")
        token_count = ctypes.c_int()
        status = int(
            self._lib.edgeflow_llm_session_ingest_text(
                self._ctx,
                sequence_id,
                _encode_native_text(text, label="text"),
                int(add_special),
                int(parse_special),
                ctypes.byref(token_count),
            )
        )
        self._raise_for_status(status, "ingest text")
        return int(token_count.value)

    def ingest_user_prompt(self, sequence_id: int, prompt: str) -> int:
        """Render and ingest a user prompt exactly as the legacy generate ABI does."""
        self._require_owner()
        sequence_id = self._checked_sequence_id(sequence_id)
        if not prompt:
            raise ValueError("prompt must not be empty")
        token_count = ctypes.c_int()
        status = int(
            self._lib.edgeflow_llm_session_ingest_user_prompt(
                self._ctx,
                sequence_id,
                _encode_native_text(prompt, label="prompt"),
                ctypes.byref(token_count),
            )
        )
        self._raise_for_status(status, "ingest user prompt")
        return int(token_count.value)

    def ingest_tokens(self, sequence_id: int, tokens: Sequence[int]) -> None:
        self._require_owner()
        sequence_id = self._checked_sequence_id(sequence_id)
        values = tuple(self._checked_token_id(token) for token in tokens)
        if not values:
            raise ValueError("tokens must not be empty")
        _checked_integer(
            len(values), label="token_count", minimum=1,
            maximum=min(_INT32_MAX, self._signature.n_batch),
        )
        array = (ctypes.c_int32 * len(values))(*values)
        status = int(
            self._lib.edgeflow_llm_session_ingest_tokens(
                self._ctx, sequence_id, array, len(values)
            )
        )
        self._raise_for_status(status, "ingest tokens")

    def append_qwen35_visual(self, sequence_id: int, *, descriptor: object) -> SurfaceAppendReceipt:
        """Consume an output-backed FP32 descriptor, releasing it on return/error.

        First append opens a user turn; subsequent appends add ordered images.
        Native completion precedes release. Receipt proves only the borrowed
        shared-output boundary, not absence of internal llama backend copies.
        """
        return self._append_qwen_visual_surface(sequence_id, descriptor, "qwen35",
                                                SessionModelFamily.QWEN35)

    def append_qwen25vl_visual_surface(self, sequence_id: int, *, descriptor: object) -> SurfaceAppendReceipt:
        """Borrow a leased visual prefix/append until native completion; no host readback."""
        return self._append_qwen_visual_surface(sequence_id, descriptor, "qwen25vl",
                                                SessionModelFamily.QWEN25VL)

    def _append_qwen_visual_surface(self, sequence_id, descriptor, family, model_family):
        try:
            self._require_owner()
            sequence_id = self._checked_sequence_id(sequence_id)
            if sequence_id != 0 or self.signature.model_family != model_family:
                raise ValueError(f"{family} visual append requires root zero and matching family")
            if getattr(descriptor, "text_prefix_tokens", 0) or getattr(descriptor, "text_suffix_tokens", 0):
                raise ValueError("packed prompt surfaces cannot be appended as plain images")
            append = self._optional_function(f"edgeflow_llm_session_append_{family}_visual_from_surface_lease")
            if descriptor.released or descriptor.surface_lease is None:
                raise ValueError("descriptor has no active surface lease")
            if descriptor.dtype != "float32" or descriptor.data_plane != "ane_output_surface":
                raise ValueError("Qwen requires an output-backed contiguous FP32 descriptor")
            grid = tuple(_checked_integer(x, label="image_grid_thw", minimum=1)
                         for x in descriptor.image_grid_thw)
            count = _checked_integer(descriptor.token_count, label="token_count", minimum=1)
            width = _checked_integer(descriptor.hidden_size, label="hidden_size", minimum=1)
            if len(grid) != 3 or grid[0] != 1 or grid[1] % 2 or grid[2] % 2 or count != (grid[1] // 2) * (grid[2] // 2):
                raise ValueError("image grid must match the single-image merged token count")
            lease = descriptor.surface_lease
            if count * width * 4 > lease.surface_bytes:
                raise ValueError("descriptor payload exceeds leased surface size")
            request = _QwenRequest(num_visual_tokens=count, hidden_size=width,
                                   image_grid_thw=(ctypes.c_int * 3)(*grid))
            receipt = _SurfaceAppendReceipt()
            status = int(append(self._ctx, sequence_id, ctypes.byref(request),
                                lease._handle_ptr(), ctypes.byref(receipt)))
            self._raise_for_status(status, f"append {family} visual surface")
            if (not receipt.surface_pointer or receipt.surface_pointer != receipt.decoder_pointer
                    or receipt.payload_bytes != count * width * 4):
                raise RuntimeError("native visual append returned an invalid pointer receipt")
            return SurfaceAppendReceipt(receipt.surface_pointer, receipt.decoder_pointer,
                                        receipt.payload_bytes, receipt.next_position)
        finally:
            descriptor.release()

    def ingest_qwen35_query(self, sequence_id: int, prompt: str) -> int:
        """Close the image turn on a fork and append a non-thinking assistant header."""
        self._require_owner()
        sequence_id = self._checked_sequence_id(sequence_id)
        if sequence_id == 0 or not prompt:
            raise ValueError("Qwen3.5 query requires a branch and nonempty prompt")
        query = self._optional_function("edgeflow_llm_session_ingest_qwen35_query")
        count = ctypes.c_int()
        status = int(query(self._ctx, sequence_id, _encode_native_text(prompt, label="prompt"),
                           ctypes.byref(count)))
        self._raise_for_status(status, "ingest Qwen3.5 query")
        return int(count.value)

    def ingest_qwen25vl(
        self,
        sequence_id: int,
        *,
        prompt: str,
        visual_embeddings: object,
        image_grid_thw: tuple[int, int, int],
    ) -> None:
        """Ingest one completed-image Qwen message into an empty sequence."""
        self._require_owner()
        sequence_id = self._checked_sequence_id(sequence_id)
        if not prompt:
            raise ValueError("prompt must not be empty")
        _visual, request = _prepare_qwen_request(
            prompt=prompt,
            visual_embeddings=visual_embeddings,
            image_grid_thw=image_grid_thw,
        )
        status = int(
            self._lib.edgeflow_llm_session_ingest_qwen25vl(
                self._ctx, sequence_id, ctypes.byref(request)
            )
        )
        self._raise_for_status(status, "ingest completed Qwen image")

    def append_qwen25vl_visual(
        self,
        sequence_id: int,
        *,
        visual_embeddings: object,
        image_grid_thw: tuple[int, int, int],
    ) -> None:
        """Append one ordered image; this is not native video temporal M-RoPE."""
        import numpy as np

        self._require_owner()
        sequence_id = self._checked_sequence_id(sequence_id)
        append = getattr(self._lib, "edgeflow_llm_session_append_qwen25vl_visual", None)
        if append is None:
            raise NotImplementedError("native runtime has no visual append capability")
        if len(image_grid_thw) != 3 or any(
            not isinstance(v, (int, np.integer))
            or isinstance(v, (bool, np.bool_))
            or not 0 < v <= 2**31 - 1
            for v in image_grid_thw
        ):
            raise ValueError("image grid must contain three positive int32 values")
        t, h, w = image_grid_thw
        if t != 1 or h % 2 or w % 2:
            raise ValueError("visual append requires a single-image even spatial grid")
        visual, request = _prepare_qwen_request(
            prompt=None,
            visual_embeddings=visual_embeddings,
            image_grid_thw=image_grid_thw,
        )
        if visual.shape[0] != (h // 2) * (w // 2) or not np.isfinite(visual).all():
            raise ValueError(
                "visual append requires finite embeddings matching the grid"
            )
        status = int(append(self._ctx, sequence_id, ctypes.byref(request)))
        self._raise_for_status(status, "append Qwen visual image")

    def ingest_qwen25vl_visual_prefix(
        self,
        sequence_id: int,
        *,
        visual_embeddings: object,
        image_grid_thw: tuple[int, int, int],
    ) -> None:
        """Ingest a reusable Qwen image prefix before any branch-local query."""
        self._require_owner()
        sequence_id = self._checked_sequence_id(sequence_id)
        _visual, request = _prepare_qwen_request(
            prompt=None,
            visual_embeddings=visual_embeddings,
            image_grid_thw=image_grid_thw,
        )
        status = int(
            self._lib.edgeflow_llm_session_ingest_qwen25vl_visual_prefix(
                self._ctx, sequence_id, ctypes.byref(request)
            )
        )
        self._raise_for_status(status, "ingest Qwen visual prefix")

    def ingest_qwen25vl_query(self, sequence_id: int, prompt: str) -> int:
        """Append one branch-local query to an exact Qwen visual boundary."""
        self._require_owner()
        sequence_id = self._checked_sequence_id(sequence_id)
        if not prompt:
            raise ValueError("prompt must not be empty")
        token_count = ctypes.c_int()
        status = int(
            self._lib.edgeflow_llm_session_ingest_qwen25vl_query(
                self._ctx,
                sequence_id,
                _encode_native_text(prompt, label="prompt"),
                ctypes.byref(token_count),
            )
        )
        self._raise_for_status(status, "ingest Qwen branch query")
        return int(token_count.value)

    def fork_prefix(
        self, source_sequence_id: int, destination_sequence_id: int, end_position: int
    ) -> None:
        self._require_owner()
        source_sequence_id = self._checked_sequence_id(source_sequence_id)
        destination_sequence_id = self._checked_sequence_id(destination_sequence_id)
        end_position = _checked_integer(
            end_position, label="end_position", minimum=1,
            maximum=min(_INT32_MAX, self._signature.n_ctx),
        )
        status = int(
            self._lib.edgeflow_llm_session_fork_prefix(
                self._ctx, source_sequence_id, destination_sequence_id, end_position
            )
        )
        self._raise_for_status(status, "fork sequence prefix")

    @contextmanager
    def forked_sequence(
        self, source_sequence_id: int, destination_sequence_id: int, end_position: int
    ) -> Iterator[int]:
        self.fork_prefix(source_sequence_id, destination_sequence_id, end_position)
        body_error: BaseException | None = None
        try:
            yield destination_sequence_id
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            try:
                if (
                    self._ctx is not None
                    and self.sequence_info(destination_sequence_id).active
                ):
                    self.remove_sequence(destination_sequence_id)
            except BaseException as cleanup_error:
                self.close()
                if body_error is None:
                    raise
                body_error.add_note(
                    f"stateful session cleanup also failed: {cleanup_error}"
                )

    def set_ignore_eog(self, enabled: bool) -> None:
        """Mask end-of-generation tokens for fixed-workload measurements."""
        self._require_owner()
        if type(enabled) is not bool:
            raise TypeError("enabled must be a bool")
        setter = self._optional_function("edgeflow_llm_session_set_ignore_eog")
        self._raise_for_status(int(setter(self._ctx, int(enabled))), "set ignore EOG")

    def peek_token(self, sequence_id: int) -> tuple[int, bool]:
        """Read the cached token decision without advancing the sequence."""
        self._require_owner()
        sequence_id = self._checked_sequence_id(sequence_id)
        peek = self._optional_function("edgeflow_llm_session_peek_token")
        token_id = ctypes.c_int()
        is_eog = ctypes.c_int()
        status = int(peek(
            self._ctx, sequence_id, ctypes.byref(token_id), ctypes.byref(is_eog)
        ))
        self._raise_for_status(status, "peek token")
        return int(token_id.value), bool(is_eog.value)

    def decode_step(
        self, sequence_id: int, *, capture_logits: bool = False
    ) -> DecodeStep:
        self._require_owner()
        sequence_id = self._checked_sequence_id(sequence_id)
        raw = _DecodeStepResult()
        logits_array = (ctypes.c_float * self._vocab_size)() if capture_logits else None
        status = int(
            self._lib.edgeflow_llm_session_decode_step(
                self._ctx,
                sequence_id,
                ctypes.byref(raw),
                logits_array,
                self._vocab_size if capture_logits else 0,
            )
        )
        self._raise_for_status(status, "decode step")
        logits = (
            np.frombuffer(logits_array, dtype=np.float32).copy()
            if logits_array is not None
            else None
        )
        if logits is not None and not np.isfinite(logits).all():
            self.close()
            raise RuntimeError(
                "native decode returned non-finite logits; session was closed"
            )
        if logits is not None:
            logits.flags.writeable = False
        return DecodeStep(
            sequence_id=sequence_id,
            token_id=int(raw.token_id),
            token_position=int(raw.token_position),
            is_eog=bool(raw.is_eog),
            next_token_id=int(raw.next_token_id),
            logits=logits,
        )

    def token_piece_bytes(self, token_id: int) -> bytes:
        """Return one native vocabulary piece without assuming UTF-8 boundaries."""
        self._require_owner()
        token_id = self._checked_token_id(token_id)
        size = ctypes.c_int()
        status = int(
            self._lib.edgeflow_llm_session_token_piece(
                self._ctx, token_id, None, 0, ctypes.byref(size)
            )
        )
        self._raise_for_status(status, "size token piece")
        if size.value == 0:
            return b""
        buffer = ctypes.create_string_buffer(size.value)
        status = int(
            self._lib.edgeflow_llm_session_token_piece(
                self._ctx,
                token_id,
                buffer,
                len(buffer),
                ctypes.byref(size),
            )
        )
        self._raise_for_status(status, "decode token piece")
        return bytes(buffer.raw[: size.value])

    def remove_sequence(self, sequence_id: int) -> None:
        self._require_owner()
        sequence_id = self._checked_sequence_id(sequence_id)
        status = int(
            self._lib.edgeflow_llm_session_remove_sequence(self._ctx, sequence_id)
        )
        self._raise_for_status(status, "remove sequence")

    def _checked_sequence_id(self, sequence_id: object) -> int:
        return _checked_integer(
            sequence_id, label="sequence_id",
            maximum=min(_INT32_MAX, self._signature.n_seq_max - 1),
        )

    def _checked_token_id(self, token_id: object) -> int:
        return _checked_integer(
            token_id, label="token_id", maximum=min(_INT32_MAX, self._vocab_size - 1)
        )

    def close(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("stateful session must be closed by its creating thread")
        if self._ctx is None:
            return
        ctx = self._ctx
        self._ctx = None
        self._lib.edgeflow_llm_session_free(ctx)

    def __enter__(self) -> Self:
        self._require_owner()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()

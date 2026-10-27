"""ctypes adapter for the retained FastVLM language path using llama.cpp."""

import ctypes
import operator
from pathlib import Path

import numpy as np


class _QueryInfo(ctypes.Structure):
    _fields_ = [("prefill_finished_s", ctypes.c_double),
                ("actual_count", ctypes.c_int32), ("root", ctypes.c_int32),
                ("cleanup_ok", ctypes.c_int32), ("logits_captured", ctypes.c_int32)]


class _HandoffInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in (
        "text_ingest_started_s", "text_ingest_finished_s",
        "visual_ingest_started_s", "visual_ingest_finished_s",
        "surface_setup_s", "surface_teardown_s")]


def _positive_int(value, name, maximum):
    value = operator.index(value)
    if not 0 < value <= maximum:
        raise ValueError(f"{name} must be in [1, {maximum}]")
    return value


class GGMLLanguage:
    """Single-owner retained context with optional additive native bridge exports.

    last_query records prefill_finished_s, actual_count, root and cleanup_ok,
    plus first_logits only when requested and captured. Failure metadata remains
    available after a native error. A failed cleanup requires reset().
    last_handoff separates completed text/visual ingest from surface setup and
    teardown; handoff_s excludes both ingest stages and Python queueing.
    Archived runtimes retain default init/query behavior; unavailable metadata
    is None. Larger contexts and new query options require the new exports.
    """

    def __init__(self, model, library, n_ctx=4096):
        self.ctx = None
        self.n_ctx = _positive_int(n_ctx, "n_ctx", 32768)
        self.lib = ctypes.CDLL(str(library))
        integer = ctypes.POINTER(ctypes.c_int32)
        floating = ctypes.POINTER(ctypes.c_float)
        self.lib.fv_error.restype = ctypes.c_char_p
        self.lib.fv_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.fv_init.restype = ctypes.c_void_p
        init_with_context = getattr(self.lib, "fv_init_with_context", None)
        if init_with_context is not None:
            init_with_context.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            init_with_context.restype = ctypes.c_void_p
        elif self.n_ctx != 4096:
            raise RuntimeError("nondefault n_ctx requires a native bridge rebuild")
        self.lib.fv_free.argtypes = [ctypes.c_void_p]
        self.lib.fv_free.restype = None
        self.lib.fv_reset.argtypes = [ctypes.c_void_p]
        self.lib.fv_reset.restype = ctypes.c_int
        self.lib.fv_append.argtypes = [ctypes.c_void_p, integer, ctypes.c_int,
                                       floating, ctypes.c_int, ctypes.c_int]
        self.lib.fv_append_surface.argtypes = [ctypes.c_void_p, integer, ctypes.c_int,
                                               ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.fv_append.restype = self.lib.fv_append_surface.restype = ctypes.c_int
        self._begin_append = getattr(self.lib, "fv_begin_append", None)
        self._cancel_append = getattr(self.lib, "fv_cancel_append", None)
        if self._begin_append is not None and self._cancel_append is not None:
            self._begin_append.argtypes = [ctypes.c_void_p, integer, ctypes.c_int, ctypes.c_int]
            self._begin_append.restype = ctypes.c_int
            self._cancel_append.argtypes = [ctypes.c_void_p]
            self._cancel_append.restype = ctypes.c_int
        self.lib.fv_last_embedding_pointer.argtypes = [ctypes.c_void_p]
        self.lib.fv_last_embedding_pointer.restype = ctypes.c_uint64
        self._last_handoff = getattr(self.lib, "fv_last_handoff", None)
        if self._last_handoff is not None:
            self._last_handoff.argtypes = [ctypes.c_void_p, ctypes.POINTER(_HandoffInfo)]
            self._last_handoff.restype = ctypes.c_int
        vocab_size = getattr(self.lib, "fv_vocab_size", None)
        if vocab_size is not None:
            vocab_size.argtypes = [ctypes.c_void_p]
            vocab_size.restype = ctypes.c_int
        self.lib.fv_query.argtypes = [ctypes.c_void_p, integer, ctypes.c_int,
                                      ctypes.c_int, integer, ctypes.POINTER(ctypes.c_double)]
        self.lib.fv_query.restype = ctypes.c_int
        self._query_ex = getattr(self.lib, "fv_query_ex", None)
        if self._query_ex is not None:
            self._query_ex.argtypes = [*self.lib.fv_query.argtypes, floating, ctypes.c_int,
                                      ctypes.c_int, ctypes.POINTER(_QueryInfo)]
            self._query_ex.restype = ctypes.c_int
        paths = (str(model).encode(), str(Path(library).parent).encode())
        self.ctx = (init_with_context(*paths, self.n_ctx) if init_with_context is not None
                    else self.lib.fv_init(*paths))
        if not self.ctx:
            raise RuntimeError(self.lib.fv_error().decode())
        try:
            self.vocab_size = self.checked(vocab_size(self.ctx)) if vocab_size is not None else None
            if self.vocab_size is not None and self.vocab_size <= 0:
                raise RuntimeError("native vocabulary is empty")
        except BaseException:
            self.close()
            raise
        self.position = 0
        self.last_handoff = None
        self.last_query = None

    def _require_open(self):
        if not self.ctx:
            raise RuntimeError("ggml language is closed")

    def checked(self, status):
        if status < 0:
            raise RuntimeError(self.lib.fv_error().decode())
        return status

    def reset(self):
        self._require_open()
        self.checked(self.lib.fv_reset(self.ctx))
        self.position = 0
        self.last_handoff = self.last_query = None

    def _handoff_timing(self):
        if self._last_handoff is None:
            return dict.fromkeys([name for name, _ in _HandoffInfo._fields_]
                                 + ["text_ingest_s", "visual_ingest_s", "handoff_s"])
        info = _HandoffInfo()
        self.checked(self._last_handoff(self.ctx, ctypes.byref(info)))
        timing = {name: getattr(info, name) for name, _ in info._fields_}
        timing["text_ingest_s"] = info.text_ingest_finished_s - info.text_ingest_started_s
        timing["visual_ingest_s"] = info.visual_ingest_finished_s - info.visual_ingest_started_s
        timing["handoff_s"] = info.surface_setup_s + info.surface_teardown_s
        return timing

    def begin_append(self, tokens, visual_count):
        """Stage a frame label on the owner; position stays at the published root."""
        self._require_open()
        if self._begin_append is None or self._cancel_append is None:
            raise RuntimeError("early frame labels require a native bridge rebuild")
        tokens = np.ascontiguousarray(tokens, dtype=np.int32).reshape(-1)
        count = _positive_int(tokens.size, "token_count", 2**31 - 1)
        visual_count = _positive_int(visual_count, "visual_count", 2**31 - 1)
        self.last_handoff = None
        return self.checked(self._begin_append(
            self.ctx, tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), count, visual_count))

    def cancel_append(self):
        """Discard any staged label if the visual producer or consumer fails."""
        self._require_open()
        if self._cancel_append is not None:
            self.position = self.checked(self._cancel_append(self.ctx))

    def append(self, tokens, features):
        """Borrow a descriptor's lease until ingest completes; caller still owns release()."""
        self._require_open()
        self.last_handoff = None
        if tokens is None:
            token_pointer, token_count = None, 0
        else:
            tokens = np.ascontiguousarray(tokens, dtype=np.int32).reshape(-1)
            token_count = _positive_int(tokens.size, "token_count", 2**31 - 1)
            token_pointer = tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        lease = None
        if hasattr(features, "surface_lease"):
            lease = features.surface_lease
            if getattr(features, "released", False) or lease is None or not lease.lease_handle:
                raise RuntimeError("descriptor has no active surface lease")
            if features.dtype != "float32":
                raise ValueError("surface embeddings must be float32")
            if getattr(features, "text_prefix_tokens", 0) or getattr(features, "text_suffix_tokens", 0):
                raise ValueError("packed prompt surfaces are not supported by this visual append")
            handle, count, width = lease.lease_handle, features.token_count, features.hidden_size
            receipt = dict(features.handoff_receipt)
            producer_pointer = receipt.get("producer_pointer", 0)
        elif hasattr(features, "lease_handle"):
            handle, count, width = features.lease_handle, features.shape[-2], features.shape[-1]
            receipt, producer_pointer = dict(features.receipt), features.pointer
        else:
            features = np.ascontiguousarray(features, dtype=np.float32)
            if features.ndim not in (2, 3) or (features.ndim == 3 and features.shape[0] != 1):
                raise ValueError("visual shape must be [tokens, hidden] or [1, tokens, hidden]")
            self.position = self.checked(self.lib.fv_append(
                self.ctx, token_pointer, token_count,
                features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                _positive_int(features.shape[-2], "visual_count", 2**31 - 1),
                _positive_int(features.shape[-1], "hidden_size", 2**31 - 1)))
            producer_pointer = int(features.ctypes.data)
            receipt = {"scope": "GPU frontend FP32 view to native batch; no CoreML surface"}
        if lease is not None or hasattr(features, "lease_handle"):
            if not handle:
                raise ValueError("surface lease handle is null")
            producer_pointer = _positive_int(producer_pointer, "producer_pointer", 2**64 - 1)
            count = _positive_int(count, "visual_count", 2**31 - 1)
            width = _positive_int(width, "hidden_size", 2**31 - 1)
            self.position = self.checked(self.lib.fv_append_surface(
                self.ctx, token_pointer, token_count,
                handle, count, width))
            receipt["application_embedding_copy_bytes"] = 0
        pointer = int(self.lib.fv_last_embedding_pointer(self.ctx))
        self.last_handoff = {**receipt, **self._handoff_timing(),
                             "producer_pointer": producer_pointer, "consumer_pointer": pointer,
                             "producer_consumer_alias": pointer == producer_pointer}
        if pointer != producer_pointer:
            raise RuntimeError("producer output and native embedding input do not alias")
        return self.position

    def query(self, tokens, budget, *, capture_first_logits=False, stop_at_eog=False):
        """Return IDs and the unchanged three timestamps; EOG counts as an output ID."""
        self._require_open()
        self.last_query = None
        if self._query_ex is None and (capture_first_logits or stop_at_eog):
            raise RuntimeError("query options require a native bridge rebuild")
        if capture_first_logits and self.vocab_size is None:
            raise RuntimeError("capture_first_logits requires a native bridge rebuild with vocab_size")
        budget = _positive_int(budget, "budget", self.n_ctx)
        tokens = np.ascontiguousarray(tokens, dtype=np.int32).reshape(-1)
        _positive_int(tokens.size, "token_count", 2**31 - 1)
        output = np.empty(budget, dtype=np.int32)
        times = np.empty(3, dtype=np.float64)
        arguments = (
            self.ctx, tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), tokens.size,
            budget, output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if self._query_ex is None:
            status = self.lib.fv_query(*arguments)
            self.last_query = {"prefill_finished_s": None,
                               "actual_count": budget if status >= 0 else None,
                               "root": status if status >= 0 else self.position,
                               "cleanup_ok": True if status >= 0 else None}
        else:
            logits = np.empty(self.vocab_size, dtype=np.float32) if capture_first_logits else None
            info = _QueryInfo()
            status = self._query_ex(
                *arguments,
                logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if logits is not None else None,
                self.vocab_size if logits is not None else 0, int(bool(stop_at_eog)), ctypes.byref(info))
            prefill_finished = info.prefill_finished_s if np.isfinite(info.prefill_finished_s) else None
            self.last_query = {"prefill_finished_s": prefill_finished,
                               "actual_count": info.actual_count, "root": info.root,
                               "cleanup_ok": bool(info.cleanup_ok)}
            if logits is not None and info.logits_captured:
                self.last_query["first_logits"] = logits
        root = self.checked(status)
        if root != self.position or self.last_query["root"] != root or not self.last_query["cleanup_ok"]:
            raise RuntimeError("ggml private suffix changed the retained root")
        count = self.last_query["actual_count"]
        if not 0 < count <= budget:
            raise RuntimeError("native query returned an invalid output count")
        return output[:count].tolist(), times.tolist()

    def close(self):
        if self.ctx:
            self.lib.fv_free(self.ctx)
            self.ctx = None

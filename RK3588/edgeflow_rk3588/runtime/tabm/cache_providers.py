"""
TABM Core: lane-local CacheProvider interface and implementations.

NullCacheProvider represents unsupported lanes; MNNCacheFileProvider models
CPU-MNN cache handles and delegates execution to an injected backend.
Public imports are available through `...tabm.prefix_registry`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from RK3588.edgeflow_rk3588.runtime.tabm.prefix_store import (  # noqa: F401
    PrefixEntry,
    PrefixValidity,
)

# ---------------------------------------------------------------------------
# CacheProvider interface + implementations
# ---------------------------------------------------------------------------

class CacheProvider(ABC):
    """
    Lane-local view of prefix-cache capability.
    One CacheProvider per lane. NullCacheProvider is the always-unsupported
    baseline; MNNCacheFileProvider delegates materialization to an injected
    worker backend.
    """

    @property
    @abstractmethod
    def lane_id(self) -> str: ...

    @property
    @abstractmethod
    def supports_prefix_cache(self) -> bool: ...

    @abstractmethod
    def can_materialize(self, entry: PrefixEntry) -> bool: ...

    @abstractmethod
    def is_warm(self, entry: PrefixEntry) -> bool: ...

    @abstractmethod
    def run_with_prefix(self, entry: PrefixEntry, suffix: str):
        ...


def _default_handle_exists(_handle: str) -> bool:
    """Default handle_exists() for MNNCacheFileProvider: ALWAYS False —
    conservative-by-default; NO real file probing is hardcoded here."""
    return False


class NullCacheProvider(CacheProvider):
    """
    Always-unsupported CacheProvider — e.g. the GPU/OpenCL lane.

    The MNN OpenCL backend has no KVMeta (llm.cpp:462). This provider never
    materializes or warms a prefix and rejects prefix-cache execution.
    """

    def __init__(self, lane_id: str) -> None:
        self._lane_id = lane_id

    @property
    def lane_id(self) -> str:
        return self._lane_id

    @property
    def supports_prefix_cache(self) -> bool:
        return False

    def can_materialize(self, entry: PrefixEntry) -> bool:
        return False

    def is_warm(self, entry: PrefixEntry) -> bool:
        return False

    def run_with_prefix(self, entry: PrefixEntry, suffix: str):
        raise NotImplementedError(
            "NullCacheProvider does not support prefix caching at all (no "
            "KVMeta on this lane's backend; llm.cpp:462)"
        )


class MNNCacheFileProvider(CacheProvider):
    """
    CPU MNN-lane CacheProvider for ``Llm::setPrefixCacheFile`` handles.
    Derives a deterministic filename and models warmth from registry
    validity plus an injected existence check. Execution requires an
    injected backend and prefix-text resolver.

    handle_exists : injected callable, handle -> bool. Models "does the
    engine's *_sync.k/.v file for this handle exist on disk" WITHOUT
    hardcoding file probing. Defaults to _default_handle_exists (always
    False, conservative by default).
    """

    def __init__(
        self,
        lane_id: str,
        backend_id: str,
        handle_exists: Callable[[str], bool] = _default_handle_exists,
        backend=None,
        prefix_text_resolver: "Callable | None" = None,
    ) -> None:
        """
        backend / prefix_text_resolver (both optional):
          backend : an MNNAdapterWorkerBackend-compatible object exposing
              run_prefix_request(name, prefix_text, suffix_text, *,
              max_tokens, ...). When None (the default), run_with_prefix
              raises NotImplementedError.
          prefix_text_resolver : callable (PrefixEntry) -> Optional[str]
              returning the prefix's RAW UN-TEMPLATED text (the same
              byte-string whose tokenization the entry's identity hashes).
              PrefixEntry deliberately does not carry
              raw text; whoever owns the registry aliases injects this.
        """
        self._lane_id = lane_id
        self._backend_id = backend_id
        self._handle_exists = handle_exists
        self._backend = backend
        self._prefix_text_resolver = prefix_text_resolver

    @property
    def lane_id(self) -> str:
        return self._lane_id

    @property
    def supports_prefix_cache(self) -> bool:
        return True

    @staticmethod
    def handle_for(prefix_id: str) -> str:
        """Deterministic setPrefixCacheFile stem using the full prefix_id.

        Truncation can create collisions and silently reuse the wrong KV.
        """
        return f"tabm_prefix_{prefix_id}"

    def can_materialize(self, entry: PrefixEntry) -> bool:
        return self.supports_prefix_cache and entry.compat.backend_id == self._backend_id

    def is_warm(self, entry: PrefixEntry) -> bool:
        """Warmth = backend-compat AND validity (MATERIALIZED/REUSABLE) AND
        handle_exists(attached-or-derived handle).

        Backend compatibility is required even for attached handles because
        attach_handle() does not consult providers.

        When no handle is attached for this lane (e.g. an
        entry restored via PrefixRegistry.load() -- physical_handles is
        runtime-only and always empty right after load), fall back to the
        DERIVED handle_for(entry.prefix_id) instead of reporting cold by
        construction. An explicitly attached handle (a custom name) still
        wins when present; the injected handle_exists() stays the ground
        truth either way, so a derived name with no real files on disk is
        still correctly cold.
        """
        if not self.can_materialize(entry):
            return False
        if entry.validity not in (PrefixValidity.MATERIALIZED, PrefixValidity.REUSABLE):
            return False
        handle = entry.physical_handles.get(self._lane_id) or self.handle_for(entry.prefix_id)
        return self._handle_exists(handle)

    def run_with_prefix(self, entry: PrefixEntry, suffix: str, *, max_tokens: int = 16):
        """
        Execute a prefix-cache request on this lane's worker. Requires a
        wired ``backend`` and ``prefix_text_resolver`` (see __init__). An
        unwired provider raises NotImplementedError.

        Returns the backend's result dict (ok/output_ids/prefill_us/
        decode_us/pfx{hit,c1,c2,...}). Registry bookkeeping (validity
        transition on first materialization, attach_handle, record_measured)
        is deliberately the CALLER's job — the provider does not mutate the
        registry.
        """
        if self._backend is None:
            raise NotImplementedError(
                "MNNCacheFileProvider.run_with_prefix: no backend wired -- "
                "construct with "
                "backend=<MNNAdapterWorkerBackend> and prefix_text_resolver= "
                "to execute"
            )
        if self._prefix_text_resolver is None:
            raise ValueError(
                "MNNCacheFileProvider.run_with_prefix: backend wired but no "
                "prefix_text_resolver -- cannot recover the prefix's raw text"
            )
        prefix_text = self._prefix_text_resolver(entry)
        if not prefix_text:
            raise ValueError(
                "MNNCacheFileProvider.run_with_prefix: prefix_text_resolver "
                f"returned nothing for prefix_id={entry.prefix_id[:16]}..."
            )
        handle = entry.physical_handles.get(self._lane_id) or self.handle_for(entry.prefix_id)
        return self._backend.run_prefix_request(
            handle, prefix_text, suffix, max_tokens=max_tokens
        )

"""Single-coordinator LRU of exported visual roots, not a native KV codec."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np

from RK3588.edgeflow_rk3588.runtime.tabm.visual_context import (
    ReplaySnapshot,
    validate_replay_snapshot,
)


@dataclass(frozen=True)
class _Entry:
    source_snapshot: ReplaySnapshot
    path: Path
    nbytes: int


class VisualKVCache:
    """Reuse complete roots with identical numerical inputs across requests.

    Call serially with donors/receivers matching the constructor's contract.
    ``model_key`` must identify both weights and tokenizer; ``representation``
    must identify numerical and position semantics, including the supported
    onevision-1d-contiguous-per-frame contract. Exact raw prefix text is a
    conservative token-ID identity under that same tokenizer. The receiver's
    ``import_cached_root`` validates source/target payload and partition before
    the native visual_prefix_v1 cache_import validates and aliases the bundle.
    No logical identity is rewritten by this cache.
    """

    def __init__(
        self,
        directory,
        model_key: str,
        prefix_text: str,
        representation: dict | str,
        capacity: int = 4,
        max_bytes: int = 64 << 20,
    ) -> None:
        if not isinstance(model_key, str) or not model_key.strip():
            raise ValueError("model_key must be a non-empty string")
        if not isinstance(prefix_text, str):
            raise ValueError("prefix_text must be a string")  # noqa: TRY004 - match runtime config validation.
        if not isinstance(representation, (dict, str)):
            raise ValueError("representation must be a JSON dictionary or string")  # noqa: TRY004
        for name, value in (("capacity", capacity), ("max_bytes", max_bytes)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        # Freeze the caller's representation, including nested contract fields.
        self._contract = json.dumps(
            ["visual-kv-cache-v1", model_key, prefix_text, representation],
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        self._capacity = capacity
        self._max_bytes = max_bytes
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._hits = self._misses = self._evictions = self._resident_bytes = 0
        self._closed = False
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self._directory = TemporaryDirectory(prefix="visual-kv-cache-", dir=directory)

    def content_key(self, snapshot: ReplaySnapshot) -> str:
        """Hash immutable inputs, excluding stream/session/event labels and time."""
        validate_replay_snapshot(snapshot)
        metadata = json.dumps(
            [np.dtype(np.float32).str, snapshot.hidden, snapshot.event_token_counts],
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(self._contract)
        digest.update(b"\0")
        digest.update(metadata)
        digest.update(b"\0")
        for chunk in snapshot.chunks:
            digest.update(chunk)
        return digest.hexdigest()

    def stats(self) -> dict:
        """Hits are successful imports; misses count attempted source exports."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "entries": len(self._entries),
            "resident_bytes": self._resident_bytes,
        }

    def materialize(self, snapshot: ReplaySnapshot, donor, receiver) -> dict:
        """Export on a miss, then always import with the original source identity."""
        if self._closed:
            raise RuntimeError("visual KV cache is closed")
        key = self.content_key(snapshot)
        if not snapshot.version:
            raise ValueError("visual KV cache requires at least one visual event")
        entry = self._entries.get(key)
        hit = entry is not None
        if hit:
            receipt = receiver.import_cached_root(
                snapshot, entry.path, entry.source_snapshot,
            ).result()
            if not receipt.get("ok", True):
                raise RuntimeError("cached root import failed")
            self._entries.move_to_end(key)
            self._hits += 1
        else:
            self._misses += 1
            path = Path(self._directory.name) / f"{uuid4().hex}.bundle"
            try:
                source_receipt = donor.export_root(snapshot, path).result()
                if not source_receipt.get("ok", True):
                    raise RuntimeError("root export failed")
                nbytes = path.stat().st_size
                receipt = receiver.import_cached_root(snapshot, path, snapshot).result()
                if not receipt.get("ok", True):
                    raise RuntimeError("cached root import failed")
                if nbytes > self._max_bytes:
                    path.unlink()
                else:
                    while (len(self._entries) >= self._capacity
                           or self._resident_bytes + nbytes > self._max_bytes):
                        _, oldest = self._entries.popitem(last=False)
                        self._resident_bytes -= oldest.nbytes
                        self._evictions += 1
                        oldest.path.unlink(missing_ok=True)
                    self._entries[key] = _Entry(snapshot, path, nbytes)
                    self._resident_bytes += nbytes
            except BaseException:
                path.unlink(missing_ok=True)
                raise
        result = {
            "cache_hit": hit,
            "content_key": key,
            "skipped_source_frames": snapshot.version if hit else 0,
            "import_receipt": receipt,
            "counters": self.stats(),
        }
        if not hit:
            result["source_receipt"] = source_receipt
        return result

    def close(self) -> None:
        """Delete only this instance's temporary directory and owned bundles."""
        self._directory.cleanup()
        self._entries.clear()
        self._resident_bytes = 0
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("visual KV cache is closed")
        return self

    def __exit__(self, *exc) -> None:
        self.close()

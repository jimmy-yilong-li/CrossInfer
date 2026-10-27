"""Bounded, single-owner projected-embedding replay. No KV state or promises."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class ReplaySnapshot:
    """Published visual rows only; replay rebuilds KV on the destination lane."""

    stream_id: str
    session_id: str
    version: int
    event_ids: tuple[str, ...]
    event_times_ns: tuple[int, ...]
    event_token_counts: tuple[int, ...]
    hidden: int
    content_digest: str
    _payload: bytes | tuple[bytes, ...] = field(repr=False)
    _owner: object = field(repr=False)
    shared_events: tuple = field(default=(), repr=False, compare=False)
    mode: str = field(default="REPLAY", init=False)

    @property
    def embeddings(self) -> np.ndarray:
        # Only legacy contiguous consumers materialize the entire prefix.
        payload = self._payload if isinstance(self._payload, bytes) else b"".join(self._payload)
        return np.frombuffer(payload, dtype=np.float32).reshape(-1, self.hidden)

    @property
    def chunks(self) -> tuple:
        if isinstance(self._payload, tuple):
            return self._payload
        view = memoryview(self._payload)
        offset, chunks = 0, []
        for count in self.event_token_counts:
            end = offset + count * self.hidden * 4
            chunks.append(view[offset:end])
            offset = end
        return tuple(chunks)

    def frame(self, index: int) -> np.ndarray:
        return np.frombuffer(self.chunks[index], dtype=np.float32).reshape(-1, self.hidden)

    def has_prefix(self, prefix: ReplaySnapshot) -> bool:
        return (prefix._owner is self._owner and prefix.stream_id == self.stream_id
                and prefix.session_id == self.session_id and prefix.hidden == self.hidden
                and prefix.version <= self.version
                and prefix.event_ids == self.event_ids[:prefix.version]
                and prefix.event_times_ns == self.event_times_ns[:prefix.version]
                and prefix.event_token_counts == self.event_token_counts[:prefix.version]
                and prefix.chunks == self.chunks[:prefix.version])

    @property
    def token_count(self) -> int:
        return self.nbytes // (self.hidden * 4)

    @property
    def nbytes(self) -> int:
        return len(self._payload) if isinstance(self._payload, bytes) else sum(map(len, self._payload))


def validate_replay_snapshot(snapshot: ReplaySnapshot) -> None:
    """Check published metadata and the in-memory identity used by KV handoff."""
    if not isinstance(snapshot, ReplaySnapshot):
        raise ValueError("replay snapshot must be a ReplaySnapshot")
    if (any(not isinstance(value, str) or not value.strip()
            for value in (snapshot.stream_id, snapshot.session_id))
            or type(snapshot.version) is not int or snapshot.version < 0
            or type(snapshot.hidden) is not int or snapshot.hidden <= 0
            or not (isinstance(snapshot._payload, bytes) or
                    (isinstance(snapshot._payload, tuple)
                     and len(snapshot._payload) == snapshot.version
                     and all(isinstance(chunk, bytes) for chunk in snapshot._payload)))):
        raise ValueError("invalid replay snapshot identity or immutable payload")
    events = (snapshot.event_ids, snapshot.event_times_ns, snapshot.event_token_counts)
    if any(not isinstance(values, tuple) or len(values) != snapshot.version for values in events):
        raise ValueError("replay snapshot event count does not match version")
    if (any(not isinstance(event, str) or not event.strip() for event in snapshot.event_ids)
            or len(set(snapshot.event_ids)) != snapshot.version
            or any(type(t) is not int or t < 0 for t in snapshot.event_times_ns)
            or any(a > b for a, b in zip(snapshot.event_times_ns, snapshot.event_times_ns[1:]))
            or any(type(n) is not int or n <= 0 for n in snapshot.event_token_counts)
            or sum(snapshot.event_token_counts) * snapshot.hidden * 4 != snapshot.nbytes):
        raise ValueError("invalid replay snapshot event metadata or byte size")
    metadata = [
        "visual-replay-v1", snapshot.stream_id, snapshot.session_id,
        np.dtype(np.float32).str, snapshot.hidden,
        list(zip(snapshot.event_ids, snapshot.event_times_ns, snapshot.event_token_counts)),
    ]
    digest = hashlib.blake2b(
        json.dumps(metadata, separators=(",", ":")).encode("utf-8"), digest_size=32
    )
    digest.update(b"\0")
    for chunk, count in zip(snapshot.chunks, snapshot.event_token_counts):
        if len(chunk) != count * snapshot.hidden * 4:
            raise ValueError("replay frame byte size does not match its partition")
        digest.update(chunk)
    if digest.hexdigest() != snapshot.content_digest:
        raise ValueError("replay snapshot identity does not match its metadata and bytes")


@dataclass(frozen=True)
class BoundVisualQuery:
    """Private query text bound to a published prefix, never appended to it."""

    query_id: str
    query_text: str
    snapshot: ReplaySnapshot


@dataclass(frozen=True)
class _Event:
    event_id: str
    event_time_ns: int
    token_count: int
    payload: bytes
    shared: object | None = field(default=None, repr=False, compare=False)


class ProjectedEmbeddingLog:
    """Append-only visual events with immutable, incarnation-local snapshots.

    Times are nondecreasing; append order breaks ties. Version 0 is empty.
    Limits bound stored visual payload, not caller-held snapshots or text/KV.
    Call serially; this is not a concurrent ingestion or pending-event manager.
    """

    def __init__(
        self,
        stream_id: str,
        session_id: str,
        hidden: int,
        max_tokens: int,
        max_bytes: int,
        *,
        shared_inputs: bool = False,
        max_event_tokens: int | None = None,
    ) -> None:
        self._stream_id = _text(stream_id, "stream_id")
        self._session_id = _text(session_id, "session_id")
        self._hidden = _integer(hidden, "hidden", 1)
        self._max_tokens = _integer(max_tokens, "max_tokens", 1)
        self._max_bytes = _integer(max_bytes, "max_bytes", 1)
        self._owner = object()
        self._events: tuple[_Event, ...] = ()
        self._shared_inputs = shared_inputs
        self._max_event_tokens = max_event_tokens
        self._closed = False

    def append(
        self, event_id: str, event_time_ns: int, embeddings: np.ndarray
    ) -> ReplaySnapshot:
        """Copy and publish one finite fp32 [tokens, hidden] image event."""
        if self._closed:
            raise RuntimeError("visual log is closed")
        event_id = _text(event_id, "event_id")
        event_time_ns = _integer(event_time_ns, "event_time_ns", 0)
        if any(event.event_id == event_id for event in self._events):
            raise ValueError("duplicate event_id")
        if self._events and event_time_ns < self._events[-1].event_time_ns:
            raise ValueError("event_time_ns must be nondecreasing")
        if (
            not isinstance(embeddings, np.ndarray)
            or embeddings.dtype != np.float32
            or embeddings.ndim != 2
            or embeddings.shape[0] == 0
            or embeddings.shape[1] != self._hidden
        ):
            raise ValueError("embeddings must be fp32 [tokens > 0, hidden]")
        tokens = sum(event.token_count for event in self._events) + embeddings.shape[0]
        if tokens > self._max_tokens or tokens * self._hidden * 4 > self._max_bytes:
            raise ValueError("visual replay capacity exceeded")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must be finite")
        if self._max_event_tokens is not None and embeddings.shape[0] > self._max_event_tokens:
            raise ValueError("visual event exceeds token bound")
        shared = None
        if self._shared_inputs:
            from RK3588.edgeflow_rk3588.runtime.tabm.shm_slot import SharedFrameLease
            shared = SharedFrameLease(embeddings)
        event = _Event(
            event_id, event_time_ns, embeddings.shape[0], embeddings.tobytes(order="C"), shared
        )
        events = (*self._events, event)
        published = self._snapshot(events)
        self._events = events
        return published

    def close(self) -> None:
        """Drop log ownership; outstanding snapshots retain their frame leases."""
        self._closed = True
        self._events = ()

    def snapshot(self, version: int | None = None) -> ReplaySnapshot:
        """Return complete ordered visual rows through an already-published version."""
        version = (
            len(self._events) if version is None else _integer(version, "version", 0)
        )
        if version > len(self._events):
            raise ValueError("version has not been published")
        return self._snapshot(self._events[:version])

    def bind_query(
        self, query_id: str, query_text: str, *, snapshot: ReplaySnapshot | None = None
    ) -> BoundVisualQuery:
        """Bind now, or use a captured older snapshot; never wait for future events."""
        query_id = _text(query_id, "query_id")
        query_text = _text(query_text, "query_text")
        if snapshot is None:
            snapshot = self.snapshot()
        elif (
            not isinstance(snapshot, ReplaySnapshot)
            or snapshot._owner is not self._owner
            or snapshot != self.snapshot(snapshot.version)
        ):
            raise ValueError(
                "snapshot is foreign or does not match its published version"
            )
        return BoundVisualQuery(query_id, query_text, snapshot)

    def _snapshot(self, events: tuple[_Event, ...]) -> ReplaySnapshot:
        payload = tuple(event.payload for event in events)
        metadata = [
            "visual-replay-v1",
            self._stream_id,
            self._session_id,
            np.dtype(np.float32).str,
            self._hidden,
            [(e.event_id, e.event_time_ns, e.token_count) for e in events],
        ]
        # Native handoff needs a content identity, separate from file checksums.
        digest = hashlib.blake2b(
            json.dumps(metadata, separators=(",", ":")).encode("utf-8"),
            digest_size=32,
        )
        digest.update(b"\0")
        for chunk in payload:
            digest.update(chunk)
        return ReplaySnapshot(
            stream_id=self._stream_id,
            session_id=self._session_id,
            version=len(events),
            event_ids=tuple(e.event_id for e in events),
            event_times_ns=tuple(e.event_time_ns for e in events),
            event_token_counts=tuple(e.token_count for e in events),
            hidden=self._hidden,
            content_digest=digest.hexdigest(),
            _payload=payload,
            _owner=self._owner,
            shared_events=tuple(e.shared for e in events) if self._shared_inputs else (),
        )

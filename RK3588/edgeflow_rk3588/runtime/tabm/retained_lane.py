"""Owned retained context, consuming shared frame leases or a private staging slot."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from RK3588.edgeflow_rk3588.runtime.tabm.adapter_protocol import build_retained_req
from RK3588.edgeflow_rk3588.runtime.tabm.shm_slot import ShmSlotPool
from RK3588.edgeflow_rk3588.runtime.tabm.visual_context import (
    ReplaySnapshot,
    validate_replay_snapshot,
)


@dataclass(frozen=True, slots=True)
class LaneResidency:
    """A completed root, or an empty record while rebuilding/after failure.

    Incarnation is the snapshot's opaque owner marker, compared by identity.
    It is deliberately not inferred from a digest or a recyclable object id.
    """

    lane_id: str
    stream_id: str | None
    session_id: str | None
    version: int | None
    content_digest: str | None
    prefix_digest: str
    incarnation: object | None


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _native_stream_id(snapshot: ReplaySnapshot) -> str:
    identity = json.dumps(
        [snapshot.stream_id, snapshot.session_id], separators=(",", ":")
    ).encode("utf-8")
    return "stream_" + hashlib.blake2b(identity, digest_size=32).hexdigest()


def _add_note(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)
    else:
        # Python 3.10 has no add_note; keep the same inspectable evidence field.
        notes = list(getattr(error, "__notes__", ()))
        notes.append(note)
        error.__notes__ = notes


class RetainedLane:
    """Serial actor for an injected synchronous retained backend, not a router.

    Construction starts non-cancellable bootstrap without waiting for the
    factory. ``ready().result()`` observes its outcome. ``max_pending`` bounds
    running plus queued prepare/query work; closing and saturation raise at
    submission. Queued work is cancellable; running native calls are not.

    Results retain final native receipt fields at top level, plus logical
    identity, requested_version, replay_count, residency and host intervals.
    ``operations`` contains every native receipt unchanged, including optional
    native timing fields. Host monotonic timestamps are not native timestamps.

    A native exception or bad receipt permanently fails the lane. Termination
    and SHM cleanup stay on the owner. If death is unconfirmed, SHM stays leased
    and ``close()`` may be retried; it never frees a live worker's payload.
    """

    def __init__(
        self,
        lane_id: str,
        backend_factory: Callable[[], Any],
        prefix_text: str,
        hidden: int = 896,
        max_event_tokens: int = 197,
        max_pending: int = 8,
    ) -> None:
        if not isinstance(lane_id, str) or not lane_id.strip():
            raise ValueError("lane_id must be a non-empty string")
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        build_retained_req(0, "reset", "lane", 0, prefix_text=prefix_text)
        self._lane_id = lane_id
        self._prefix_text = prefix_text
        self._prefix_digest = hashlib.blake2b(prefix_text.encode("utf-8"), digest_size=32).hexdigest()
        self._hidden = _positive_integer(hidden, "hidden")
        self._max_event_tokens = _positive_integer(max_event_tokens, "max_event_tokens")
        self._max_pending = _positive_integer(max_pending, "max_pending")
        self._lock = threading.RLock()
        self._pending = 0
        self._queue = deque()
        self._drain_scheduled = False
        self._closing = False
        self._closed = False
        self._failure: BaseException | None = None
        self._owner_thread: int | None = None
        self._backend: Any = None
        self._pool: ShmSlotPool | None = None
        self._worker_dead = False
        self._root: ReplaySnapshot | None = None
        self._root_receipt: dict | None = None
        self._residency = self._empty_residency()
        self._close_future: Future | None = None
        self._ready: Future = Future()
        self._ready.set_running_or_notify_cancel()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"retained-{lane_id}"
        )
        self._executor.submit(self._bootstrap, backend_factory)

    @property
    def lane_id(self) -> str:
        return self._lane_id

    @property
    def pending_count(self) -> int:
        with self._lock:
            return self._pending

    def ready(self) -> Future:
        return self._ready

    def residency_snapshot(self) -> LaneResidency:
        with self._lock:
            return self._residency

    def prepare(self, snapshot: ReplaySnapshot) -> Future:
        return self._submit("prepare", snapshot)

    def export_root(self, snapshot: ReplaySnapshot, path) -> Future:
        """Prepare the donor snapshot if needed, then export on the owner queue."""
        return self._submit("export", snapshot, path)

    def import_root(self, snapshot: ReplaySnapshot, path) -> Future:
        """Replace root metadata without replay; layer packing remains native."""
        return self._submit("import", snapshot, path)

    def import_cached_root(self, snapshot: ReplaySnapshot, path, source: ReplaySnapshot) -> Future:
        """Reuse identical physical input while binding a new logical request."""
        return self._submit("cache_import", snapshot, path, source)

    def query(
        self,
        snapshot: ReplaySnapshot,
        suffix_text: str,
        max_tokens: int = 64,
        logits_name: str | None = None,
    ) -> Future:
        return self._submit("query", snapshot, suffix_text, max_tokens, logits_name)

    def close(self) -> None:
        """Drain accepted work, release/terminate on-owner, then stop the executor."""
        if threading.get_ident() == self._owner_thread:
            raise RuntimeError("close cannot block the retained lane owner thread")
        with self._lock:
            if self._closed:
                future = None
            else:
                self._closing = True
                future = self._close_future
                if future is None or future.done():
                    future = self._executor.submit(self._close_owner)
                    self._close_future = future
        error = None
        try:
            if future is not None:
                future.result()
        except BaseException as exc:  # noqa: BLE001 - preserve the first interrupt through shutdown.
            error = exc
        with self._lock:
            closed = self._closed
        if closed:
            try:
                self._executor.shutdown(wait=True)
            except BaseException as exc:  # noqa: BLE001 - never mask an earlier KeyboardInterrupt.
                if error is None:
                    error = exc
                elif isinstance(exc, KeyboardInterrupt) and not isinstance(error, KeyboardInterrupt):
                    _add_note(exc, f"retained lane close also failed: {error!r}")
                    error = exc
                else:
                    _add_note(error, f"retained executor shutdown also failed: {exc!r}")
        if error is not None:
            raise error

    def _empty_residency(self) -> LaneResidency:
        return LaneResidency(self._lane_id, None, None, None, None, self._prefix_digest, None)

    def _publish(self, snapshot: ReplaySnapshot | None) -> None:
        with self._lock:
            self._residency = (
                self._empty_residency()
                if snapshot is None
                else LaneResidency(
                    self._lane_id,
                    snapshot.stream_id,
                    snapshot.session_id,
                    snapshot.version,
                    snapshot.content_digest,
                    self._prefix_digest,
                    snapshot._owner,
                )
            )

    def _bootstrap(self, backend_factory: Callable[[], Any]) -> None:
        self._owner_thread = threading.get_ident()
        try:
            self._backend = backend_factory()
            if self._backend.is_alive() is not True:
                raise RuntimeError("retained backend did not start alive")
            self._pool = ShmSlotPool(1, self._max_event_tokens, self._hidden)
        except BaseException as exc:  # noqa: BLE001 - deliver bootstrap failure via ready.
            self._fail_owner(exc)
            self._ready.set_exception(exc)
        else:
            self._ready.set_result(None)

    def _submit(self, op: str, snapshot: ReplaySnapshot, *args: Any) -> Future:
        with self._lock:
            if self._closing:
                raise RuntimeError("retained lane is closing")
            if self._failure is not None:
                raise RuntimeError("retained lane has failed") from self._failure
            if self._pending >= self._max_pending:
                raise RuntimeError("retained lane max_pending exceeded")
            future: Future = Future()
            self._pending += 1
            future.add_done_callback(self._cancelled)
            self._queue.append((future, op, snapshot, args, time.monotonic_ns()))
            if not self._drain_scheduled:
                self._drain_scheduled = True
                try:
                    self._executor.submit(self._drain)
                except BaseException:
                    self._queue.pop()
                    self._drain_scheduled = False
                    self._pending -= 1
                    raise
            return future

    def _cancelled(self, future: Future) -> None:
        if future.cancelled():
            with self._lock:
                self._pending -= 1
                for index, job in enumerate(self._queue):
                    if job[0] is future:
                        del self._queue[index]
                        break

    def _drain(self) -> None:
        # Only this drain is queued on the executor; cancelled payloads are removable.
        while True:
            with self._lock:
                if not self._queue:
                    self._drain_scheduled = False
                    return
                job = self._queue.popleft()
            try:
                self._run(*job)
            except BaseException:
                logging.getLogger(__name__).exception("Retained lane completion callback failed")

    def _run(
        self, future: Future, op: str, snapshot: ReplaySnapshot, args: tuple, submitted: int
    ) -> None:
        if not future.set_running_or_notify_cancel():
            return
        result = None
        error = None
        try:
            self._ready.result()
            with self._lock:
                if self._failure is not None:
                    raise RuntimeError("retained lane has failed") from self._failure
            result = self._execute(op, snapshot, args, submitted)
        except BaseException as exc:  # noqa: BLE001 - deliver interruptions via the Future.
            error = exc
        finally:
            # Completion is visible only after its admission credit is returned.
            with self._lock:
                self._pending -= 1
        if error is None:
            future.set_result(result)
        else:
            future.set_exception(error)

    def _execute(self, op: str, snapshot: ReplaySnapshot, args: tuple, submitted: int) -> dict:
        started = time.monotonic_ns()
        validate_replay_snapshot(snapshot)
        if snapshot.hidden != self._hidden:
            raise ValueError("snapshot hidden does not match lane geometry")
        if any(count > self._max_event_tokens for count in snapshot.event_token_counts):
            raise ValueError("snapshot event exceeds max_event_tokens")
        if any(not np.isfinite(snapshot.frame(i)).all() for i in range(snapshot.version)):
            raise ValueError("snapshot embeddings must be finite")
        stream = _native_stream_id(snapshot)
        if op == "query":
            suffix, max_tokens, logits_name = args
            build_retained_req(
                0, "query", stream, snapshot.version,
                suffix_text=suffix, max_tokens=max_tokens, logits_name=logits_name,
            )
            if not snapshot.version:
                raise ValueError("query requires at least one visual event")
        elif op in ("export", "import", "cache_import"):
            cached_source = None
            if op == "cache_import":
                source = args[1]
                validate_replay_snapshot(source)
                if (not isinstance(source, ReplaySnapshot) or source.hidden != snapshot.hidden
                        or source.event_token_counts != snapshot.event_token_counts
                        or source.chunks != snapshot.chunks):
                    raise ValueError("cached physical prefix differs from requested input/partition")
                cached_source = (_native_stream_id(source), source.version, source.content_digest)
            build_retained_req(
                0, op, stream, snapshot.version, snapshot_path=args[0],
                content_digest=snapshot.content_digest,
                cached_source=cached_source,
            )
        operations: list[dict] = []
        try:
            if op in ("import", "cache_import"):
                self._publish(None)
                reset = self._root_receipt is None
                if reset:
                    self._call("reset", stream, 0, operations, self._prefix_text)
                keywords = {"content_digest": snapshot.content_digest}
                if op == "cache_import":
                    keywords["cached_source"] = cached_source
                receipt = self._call(op, stream, snapshot.version, operations, args[0], **keywords)
                if receipt["root_end"] != receipt["prefix_tokens"] + sum(snapshot.event_token_counts):
                    raise RuntimeError("invalid retained import receipt: snapshot root length")
                self._root = snapshot
                self._publish(snapshot)
                replay_count = 0
            else:
                historical_export = (op == "export" and self._root is not None
                                     and self._root.has_prefix(snapshot))
                if historical_export:
                    replay_count, reset = 0, False
                else:
                    replay_count, reset = self._prepare_owner(snapshot, stream, operations)
                if op == "query":
                    receipt = self._call(
                        "query", stream, snapshot.version, operations, suffix,
                        max_tokens=max_tokens, logits_name=logits_name,
                    )
                elif op == "export":
                    extra = {}
                    if self._root.version != snapshot.version:
                        extra["export_prefix"] = (
                            snapshot.version, self._root_receipt["prefix_tokens"] + snapshot.token_count)
                    receipt = self._call(
                        "export", stream, self._root.version, operations, args[0],
                        content_digest=snapshot.content_digest, **extra,
                    )
                else:
                    receipt = operations[-1]["receipt"] if operations else {}
        except BaseException as exc:
            self._fail_owner(exc)
            exc.retained_lane_context = {
                "lane_id": self._lane_id,
                "requested_version": snapshot.version,
                "replay_count": sum(
                    operation["op"] == "append" and operation.get("verified", False)
                    for operation in operations
                ),
                "host_submitted_ns": submitted,
                "host_start_ns": started,
                "host_end_ns": time.monotonic_ns(),
                "operations": tuple(operations),
            }
            raise
        return {
            **receipt,
            "ok": True,
            "lane_id": self._lane_id,
            "stream_id": stream,
            "logical_stream_id": snapshot.stream_id,
            "session_id": snapshot.session_id,
            "version": snapshot.version,
            "requested_version": snapshot.version,
            "content_digest": snapshot.content_digest,
            "prefix_digest": self._prefix_digest,
            "replay_count": replay_count,
            "host_staging_bytes": sum(row.get("host_staging_bytes", 0) for row in operations),
            "reset": reset,
            "residency": self.residency_snapshot(),
            "host_submitted_ns": submitted,
            "host_start_ns": started,
            "host_end_ns": time.monotonic_ns(),
            "operations": tuple(operations),
        }

    def _prepare_owner(
        self, snapshot: ReplaySnapshot, stream: str, operations: list[dict]
    ) -> tuple[int, bool]:
        root = self._root
        extend = (root is not None and snapshot.has_prefix(root)
                  and self.residency_snapshot().prefix_digest == self._prefix_digest)
        first = root.version if extend else 0
        if not extend or first < snapshot.version:
            self._publish(None)
        if not extend:
            self._call("reset", stream, 0, operations, self._prefix_text)
        for index in range(first, snapshot.version):
            shared = snapshot.shared_events[index] if snapshot.shared_events else None
            if shared is not None:
                self._call("append", stream, index, operations, shared.descriptor,
                           expected_crc=shared.crc32)
                operations[-1]["host_staging_bytes"] = 0
                operations[-1]["shared_frame"] = shared.descriptor.shm_name
                continue
            frame = snapshot.frame(index)
            slot = self._pool.acquire(index)
            if slot is None:
                raise RuntimeError("retained lane SHM lease is still in use")
            slot.write_embedding(frame)
            self._pool.mark_ready(slot)
            desc = self._pool.buffer_descriptor(slot)
            crc = f"{zlib.crc32(memoryview(frame).cast('B')) & 0xFFFFFFFF:08x}"
            self._call("append", stream, index, operations, desc, expected_crc=crc)
            operations[-1]["host_staging_bytes"] = frame.nbytes
            self._pool.release(slot)
        self._root = snapshot
        self._publish(snapshot)
        return snapshot.version - first, not extend

    def _call(
        self, op: str, stream: str, version: int, operations: list[dict],
        *args: Any, expected_crc: str | None = None, **kwargs: Any,
    ) -> dict:
        interval = {"op": op, "version": version, "host_start_ns": time.monotonic_ns()}
        try:
            receipt = getattr(self._backend, "retained_" + op)(stream, version, *args, **kwargs)
            interval["receipt"] = dict(receipt)
        except BaseException as exc:
            interval["error"] = repr(exc)
            raise
        finally:
            interval["host_end_ns"] = time.monotonic_ns()
            operations.append(interval)
        self._verify_receipt(op, stream, version, receipt, args, kwargs, expected_crc)
        interval["verified"] = True
        self._root_receipt = dict(receipt)
        return receipt

    def _verify_receipt(
        self, op: str, stream: str, version: int, receipt: dict,
        args: tuple, kwargs: dict, expected_crc: str | None,
    ) -> None:
        def require(condition: bool, message: str) -> None:
            if not condition:
                raise RuntimeError(f"invalid retained {op} receipt: {message}")

        previous = self._root_receipt
        before = previous["root_end"] if previous is not None else 0
        require(receipt.get("ok") is True, str(receipt.get("error", "operation failed")))
        require(receipt.get("op") == op and receipt.get("valid") is True, "operation/valid")
        for key in ("version", "root_before", "root_end", "logical_position",
                    "physical_history", "pending_remove", "prefix_tokens"):
            require(type(receipt.get(key)) is int and receipt[key] >= 0, key)
        require(receipt["root_before"] == before, "root_before")
        require(
            receipt["logical_position"] == receipt["root_end"]
            and receipt["physical_history"] - receipt["pending_remove"] == receipt["root_end"],
            "rollback positions",
        )
        if op == "release":
            require(
                receipt.get("active") is False and receipt.get("stream_id") is None
                and receipt["version"] == receipt["root_end"] == receipt["prefix_tokens"] == 0,
                "release ownership",
            )
            return
        require(
            receipt.get("active") is True and receipt.get("stream_id") == stream
            and receipt["version"] == version + (op == "append"),
            "stream/version ownership",
        )
        if op == "reset":
            require(receipt["root_end"] == 0 and receipt["prefix_tokens"] > 0, "reset root")
            return
        if op in ("import", "cache_import"):
            require(
                type(receipt.get("n_input")) is int and receipt["n_input"] == 0
                and receipt["root_end"] > 0 and receipt["prefix_tokens"] > 0
                and receipt["physical_history"] == receipt["root_end"],
                "import root metadata",
            )
            return
        require(receipt["prefix_tokens"] == previous["prefix_tokens"], "fixed prefix")
        if op == "export":
            require(
                receipt["root_end"] == before
                and type(receipt.get("n_input")) is int and receipt["n_input"] == 0,
                "export changed root or consumed input",
            )
        elif op == "append":
            desc = args[0]
            n_input = desc.shape[0] + (previous["prefix_tokens"] if before == 0 else 0)
            borrowed = getattr(self._backend, "execution_environment", {}).get(
                "EDGEFLOW_TABM_RETAINED_BORROW_INPUT"
            ) == "1"
            copied_bytes = 0 if borrowed and before else n_input * self._hidden * 4
            require(
                receipt.get("visual_nbytes") == desc.nbytes
                and receipt.get("visual_crc32") == expected_crc,
                "visual CRC/nbytes",
            )
            require(
                receipt.get("n_input") == n_input and receipt["root_end"] == before + n_input
                and receipt.get("copied_bytes") == copied_bytes,
                "append payload/root",
            )
        else:
            require(receipt["root_end"] == before and receipt.get("rollback_verified") is True,
                    "query rollback")
            ids = receipt.get("output_ids")
            require(isinstance(ids, list) and 0 < len(ids) <= kwargs["max_tokens"], "query output")

    def _fail_owner(self, error: BaseException) -> None:
        with self._lock:
            self._failure = error
        self._root = self._root_receipt = None
        self._publish(None)
        try:
            self._dispose_owner()
        except BaseException as cleanup_error:  # noqa: BLE001 - preserve the primary failure.
            _add_note(error, f"retained lane cleanup failed; resources retained: {cleanup_error!r}")

    def _dispose_owner(self) -> None:
        if not self._worker_dead:
            if self._backend is not None:
                self._backend.terminate()
                if self._backend.is_alive() is not False:
                    raise RuntimeError("worker death is unconfirmed; retaining SHM")
            self._worker_dead = True
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def _close_owner(self) -> None:
        release_error = None
        try:
            if self._failure is None and self._root_receipt is not None:
                receipt = self._root_receipt
                self._call("release", receipt["stream_id"], receipt["version"], [])
        except BaseException as exc:  # noqa: BLE001 - terminate even if release is interrupted.
            release_error = exc
        self._root = self._root_receipt = None
        self._publish(None)
        try:
            self._dispose_owner()
        except BaseException as exc:
            if release_error is not None:
                if isinstance(release_error, KeyboardInterrupt):
                    _add_note(release_error, f"retained lane cleanup failed; resources retained: {exc!r}")
                    raise release_error
                _add_note(exc, f"retained root release also failed: {release_error!r}")
            raise
        with self._lock:
            self._closed = True
        if release_error is not None:
            raise release_error

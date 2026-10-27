"""One transport owner with private retained-stream views for lane executors."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

from RK3588.edgeflow_rk3588.runtime.tabm.adapter_protocol import build_retained_req
from RK3588.edgeflow_rk3588.runtime.tabm.lane import (
    RequestValidationError,
    WorkerTerminationError,
)
from RK3588.edgeflow_rk3588.runtime.tabm.retained_lane import _add_note


class SharedRetainedBackend:
    """Construct/call/terminate one adapter on a dedicated owner thread.

    Pass ``lambda: group.view(stable_stream_key)`` to each RetainedLane.
    Close every lane before closing the group. A released view can detach
    without killing its sibling; abnormal termination fences the whole group.
    The factory must explicitly enable the matching native stream capability.
    """

    def __init__(self, factory, *, streams: int = 2):
        if not callable(factory) or type(streams) is not int or streams not in (1, 2):
            raise ValueError("shared backend requires a factory and streams=1 or 2")
        self._streams = streams
        self._lock = threading.Lock()
        self._backend = None
        self._owner_thread = None
        self._views = {}
        self._aliases = {}
        self._info = {}
        self._failure = None
        self._dead = self._closing = self._closed = False
        self._close_future = None
        self._ready = Future()
        self._ready.set_running_or_notify_cancel()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shared-retained")
        self._executor.submit(self._bootstrap, factory)

    def _bootstrap(self, factory):
        self._owner_thread = threading.get_ident()
        try:
            self._backend = factory()
            if self._backend.is_alive() is not True:
                raise RuntimeError("shared retained backend did not start alive")
            metadata = dict(self._backend.metadata)
            if metadata.get("retained_streams", 1) != self._streams:
                raise RuntimeError("shared backend native stream capability differs from requested streams")
            self._info = {"metadata": metadata,
                "execution_environment": dict(self._backend.execution_environment),
                "reference_contract": self._backend.reference_contract,
                "load_s": self._backend.load_s}
        except BaseException as error:  # noqa: BLE001 - Publish bootstrap failures through ready after owner cleanup.
            self._failure = error
            try:
                self._terminate_owner()
            except BaseException as cleanup:  # noqa: BLE001 - Preserve the bootstrap failure and attach cleanup details.
                _add_note(error, f"Shared backend cleanup failed: {cleanup!r}")
            self._ready.set_exception(error)
        else:
            self._ready.set_result(None)

    def ready(self) -> Future:
        return self._ready

    def _invoke(self, call, *args, cleanup=False):
        if threading.get_ident() == self._owner_thread:
            raise RuntimeError("shared backend cannot block its owner thread")
        with self._lock:
            if self._closed and cleanup:
                return None
            if self._closed or (self._closing and not cleanup):
                raise RuntimeError("shared backend is closing")
            future = self._executor.submit(call, *args)
        return future.result()

    def view(self, stream_key: str):
        build_retained_req(0, "release", stream_key, 0)
        return self._invoke(self._view_owner, stream_key)

    def _view_owner(self, stream_key):
        self._ready.result()
        if self._failure is not None:
            raise RuntimeError("shared backend has failed") from self._failure
        previous = self._views.get(stream_key)
        if previous is not None and not previous._detached:
            raise ValueError("stream view already has an owner")
        if previous is None and len(self._views) >= self._streams:
            raise ValueError("shared native stream capacity exhausted")
        view = _RetainedStreamView(self, stream_key)
        self._views[stream_key] = view
        return view

    def _request(self, view, op, stream_id, version, args, kwargs):
        if self._failure is not None or self._dead:
            raise RuntimeError("shared backend has failed or terminated") from self._failure
        if view._detached:
            raise RuntimeError("retained stream view is detached")
        if op not in ("reset", "import", "cache_import") and stream_id != view._logical_id:
            raise RequestValidationError("wrong retained stream for this view")
        if self._aliases.get(stream_id, view._key) != view._key:
            raise RequestValidationError("retained stream belongs to another view")
        if op == "cache_import":
            source, source_version, digest = kwargs["cached_source"]
            kwargs = {**kwargs, "cached_source":
                      (self._aliases.get(source, source), source_version, digest)}
        try:
            result = getattr(self._backend, "retained_" + op)(view._key, version, *args, **kwargs)
        except RequestValidationError:
            raise
        except BaseException as error:
            self._failure = error
            try:
                self._terminate_owner()
            except BaseException as cleanup:  # noqa: BLE001 - Preserve the request failure without masking unconfirmed death.
                _add_note(error, f"Shared backend cleanup failed: {cleanup!r}")
            raise
        view._receipt = dict(result)
        if result["ok"] and op in ("reset", "import", "cache_import"):
            view._logical_id = stream_id
            self._aliases[stream_id] = view._key
        if not result["active"]:
            view._logical_id = None
        return {**result, "native_stream_id": result["stream_id"],
                "stream_id": view._logical_id if result["stream_id"] is not None else None}

    def _terminate_owner(self):
        if self._dead:
            return
        if self._backend is not None:
            self._backend.terminate()
            if self._backend.is_alive() is not False:
                raise WorkerTerminationError("shared worker death is unconfirmed; retaining SHM")
        self._dead = True

    def _detach_owner(self, view):
        if view._detached:
            return
        if self._failure is not None or (view._receipt is not None and view._receipt["active"]):
            self._failure = self._failure or RuntimeError("active shared stream terminated")
            self._terminate_owner()
        view._detached = True

    def close(self):
        if threading.get_ident() == self._owner_thread:
            raise RuntimeError("shared backend close cannot block its owner thread")
        with self._lock:
            if self._closed:
                return
            self._closing = True
            if self._close_future is None or self._close_future.done():
                self._close_future = self._executor.submit(self._terminate_owner)
            future = self._close_future
        future.result()
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True)


class _RetainedStreamView:
    def __init__(self, owner, key):
        self._owner, self._key = owner, key
        self._logical_id = self._receipt = None
        self._detached = False

    @property
    def metadata(self):
        return dict(self._owner._info["metadata"])

    @property
    def execution_environment(self):
        return dict(self._owner._info["execution_environment"])

    @property
    def reference_contract(self):
        return self._owner._info["reference_contract"]

    @property
    def load_s(self):
        return self._owner._info["load_s"]

    def is_alive(self):
        return bool(self._owner._invoke(lambda: not self._detached and not self._owner._dead
                    and self._owner._backend.is_alive(), cleanup=True))

    def terminate(self):
        self._owner._invoke(self._owner._detach_owner, self, cleanup=True)

    def _call(self, op, stream_id, version, *args, **kwargs):
        return self._owner._invoke(self._owner._request, self, op, stream_id, version, args, kwargs)

    def retained_reset(self, stream_id, version, prefix_text, *, timeout=None):
        return self._call("reset", stream_id, version, prefix_text, timeout=timeout)

    def retained_append(self, stream_id, version, embeds_desc, *, timeout=None):
        return self._call("append", stream_id, version, embeds_desc, timeout=timeout)

    def retained_query(self, stream_id, version, suffix_text, *, max_tokens=8, logits_name=None, timeout=None):
        return self._call("query", stream_id, version, suffix_text,
                          max_tokens=max_tokens, logits_name=logits_name, timeout=timeout)

    def retained_release(self, stream_id, version, *, timeout=None):
        return self._call("release", stream_id, version, timeout=timeout)

    def retained_export(self, stream_id, version, snapshot_path, *, content_digest,
                        export_prefix=None, timeout=None):
        extra = {} if export_prefix is None else {"export_prefix": export_prefix}
        return self._call("export", stream_id, version, snapshot_path,
                          content_digest=content_digest, timeout=timeout, **extra)

    def retained_import(self, stream_id, version, snapshot_path, *, content_digest, timeout=None):
        return self._call("import", stream_id, version, snapshot_path,
                          content_digest=content_digest, timeout=timeout)

    def retained_cache_import(self, stream_id, version, snapshot_path, *, content_digest, cached_source, timeout=None):
        return self._call("cache_import", stream_id, version, snapshot_path,
                          content_digest=content_digest, cached_source=cached_source, timeout=timeout)

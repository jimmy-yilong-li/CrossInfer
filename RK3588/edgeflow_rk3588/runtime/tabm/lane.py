"""
TABM Core: WorkerBackend interface + Lane state.

WorkerBackend  : abstract interface for a decode-worker transport.
FakeWorkerBackend : deterministic in-process backend (no real processes).
ProcessWorkerBackend : multiprocessing.Process transport.
Lane           : per-lane state (owner, backend, health, EWMA).

Design note
-----------
ProcessWorkerBackend owns a subprocess
and two queues and delegates all other state management to LaneDispatcher.
"""
from __future__ import annotations

import multiprocessing
import queue
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from RK3588.edgeflow_rk3588.runtime.tabm.slot import LaneOwner


# ---------------------------------------------------------------------------
# Work item and result
# ---------------------------------------------------------------------------

class RequestValidationError(ValueError):
    """Request rejected before submission, with no in-flight worker state."""


class WorkerTerminationError(RuntimeError):
    """Worker exit is unconfirmed; any shared-memory lease must stay owned."""


@dataclass
class WorkItem:
    """Payload sent to a decode worker."""
    request_id: int
    prompt_ids: list[int]
    max_decode_tokens: int
    prompt_text: str = ""   # Text prompt for decoder backends.
    # A SlotBufferDescriptor (as dict) when the caller wants embeds mode
    # (buffer_kind=="shm"); None => text mode. See MNNAdapterWorkerBackend.
    embeds_desc: Optional[dict] = None
    # When ALL THREE are set, the adapter submits the
    # worker's `prefix` REQ mode instead of text (cross-request prefix-cache
    # execution on the PRODUCTION submit/poll path). prefix_name is the
    # setPrefixCacheFile filename stem (MNNCacheFileProvider.handle_for /
    # attached handle); texts are RAW UN-TEMPLATED segments (TOK identity
    # discipline). Text-only for now: combining with embeds_desc is refused
    # by the adapter (VLM+prefix is future work).
    prefix_name: Optional[str] = None
    prefix_text: Optional[str] = None
    suffix_text: Optional[str] = None
    # Host-local absolute time.monotonic() deadline, not an epoch timestamp.
    # Direct adapter callers may set it without admission; None keeps the
    # adapter operation timeout. Never serialized into the native REQ line.
    deadline_monotonic: Optional[float] = None


@dataclass
class WorkResult:
    """Reply from a decode worker."""
    request_id: int
    output_ids: list[int]
    ok: bool
    error: Optional[str]
    service_time: float
    # Exact engine metrics from MNNAdapterWorkerBackend: prefill_us,
    # decode_us, copied_bytes, n_prompt, load_s. None for backends that don't
    # report them (e.g. FakeWorkerBackend, MNNDemoWorkerBackend).
    extra: Optional[dict] = None
    # Failure classification consumed by LaneDispatcher.reap() when ok=False
    # (ignored when ok=True):
    #   True  (default) — the LANE is suspect (worker died, protocol
    #           violation, unclassified failure): reap() fences the lane
    #           (failure_count += 1, healthy = False).  The conservative
    #           default preserves pre-field behaviour for every backend that
    #           does not classify its failures.
    #   False — REQUEST-level failure from a live, protocol-correct worker
    #           (tabm_decoder_worker replies RES ok=0 and keeps serving:
    #           "vlm prompt lacks <image> placeholder", "shm_open failed",
    #           "empty prompt", ...): reap() reclaims the slot but leaves
    #           the lane healthy and selectable.
    lane_fault: bool = True


# ---------------------------------------------------------------------------
# Abstract WorkerBackend
# ---------------------------------------------------------------------------

class WorkerBackend(ABC):
    """
    Transport interface for a single decode-worker.

    Implementations must be non-blocking: submit() enqueues work, poll()
    returns None when the worker hasn't finished yet.
    """

    @abstractmethod
    def submit(self, work: WorkItem) -> None:
        """Begin one decode; RequestValidationError must precede transport I/O."""

    @abstractmethod
    def poll(self) -> Optional[WorkResult]:
        """Return None if not ready; a WorkResult when done or failed."""

    @abstractmethod
    def is_alive(self) -> bool:
        """False if the worker process/backend has died."""

    @abstractmethod
    def terminate(self) -> None:
        """Boundedly confirm worker exit, or raise WorkerTerminationError.

        Idempotent. A caller must not release a failed request's shared slot
        until this returns and is_alive() is False.
        """


# ---------------------------------------------------------------------------
# FakeWorkerBackend
# ---------------------------------------------------------------------------

class FakeWorkerBackend(WorkerBackend):
    """
    Deterministic, in-process backend for testing.

    Crash behaviour (documented contract)
    --------------------------------------
    When a crash is triggered:
      - poll() returns WorkResult(ok=False, error="simulated crash").
      - is_alive() immediately returns False.
    This mirrors the combined is_alive()==False + ok==False signal that the
    LaneDispatcher reap() observes from a real crashed subprocess.

    Parameters
    ----------
    service_time : float
        Reported service_time in successful WorkResults.
    crash_on_request_id : Optional[int]
        If set, poll() for this exact request_id triggers a crash.
    crash_after_n_submits : Optional[int]
        If set, any poll() after more than N total submit() calls crashes.
    """

    def __init__(
        self,
        service_time: float = 0.01,
        crash_on_request_id: Optional[int] = None,
        crash_after_n_submits: Optional[int] = None,
    ) -> None:
        self._service_time = service_time
        self._crash_on_request_id = crash_on_request_id
        self._crash_after_n_submits = crash_after_n_submits
        self._submit_count: int = 0
        self._pending: Optional[WorkItem] = None
        self._alive: bool = True

    def submit(self, work: WorkItem) -> None:
        self._submit_count += 1
        self._pending = work

    def poll(self) -> Optional[WorkResult]:
        if self._pending is None:
            return None
        work = self._pending
        self._pending = None

        should_crash = False
        if (
            self._crash_on_request_id is not None
            and work.request_id == self._crash_on_request_id
        ):
            should_crash = True
        if (
            self._crash_after_n_submits is not None
            and self._submit_count > self._crash_after_n_submits
        ):
            should_crash = True

        if should_crash:
            self._alive = False
            return WorkResult(
                request_id=work.request_id,
                output_ids=[],
                ok=False,
                error="simulated crash",
                service_time=0.0,
            )

        return WorkResult(
            request_id=work.request_id,
            output_ids=list(work.prompt_ids),
            ok=True,
            error=None,
            service_time=self._service_time,
        )

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self._alive = False


# ---------------------------------------------------------------------------
# Worker functions for ProcessWorkerBackend
# ---------------------------------------------------------------------------

def _default_worker_fn(
    in_q: "multiprocessing.Queue[Optional[WorkItem]]",
    out_q: "multiprocessing.Queue[WorkResult]",
) -> None:
    """
    Stub decode worker.  Reads WorkItems, echoes prompt_ids as output_ids.

    Blocks indefinitely on in_q.get() — exits ONLY on the None shutdown
    sentinel or if the queue raises (broken pipe / process killed).  No idle
    timeout: a timeout expiry would look like a crash to the dispatcher.
    """
    while True:
        try:
            item = in_q.get()  # blocking — exits only on sentinel or exception
        except Exception:
            # queue broken or process killed — exit cleanly
            break
        if item is None:  # shutdown sentinel
            break
        t_start = time.monotonic()
        result = WorkResult(
            request_id=item.request_id,
            output_ids=list(item.prompt_ids),
            ok=True,
            error=None,
            service_time=time.monotonic() - t_start,
        )
        out_q.put(result)


def _stub_crashing_worker_fn(
    in_q: "multiprocessing.Queue[Optional[WorkItem]]",
    out_q: "multiprocessing.Queue[WorkResult]",
) -> None:
    """
    Stub worker that raises immediately.  Used only by the crash-detection
    smoke test.  Defined at module level so it is picklable with spawn context.
    """
    raise RuntimeError("simulated worker crash")


# ---------------------------------------------------------------------------
# ProcessWorkerBackend
# ---------------------------------------------------------------------------

class ProcessWorkerBackend(WorkerBackend):
    """
    Production transport: a subprocess running worker_fn, communicating via
    multiprocessing queues.

    worker_fn(in_q, out_q) should loop reading WorkItems and writing WorkResults
    until it receives None (sentinel) or raises.

    poll() contract
    ---------------
    - Returns a WorkResult if one is available in out_q.
    - If out_q is empty AND the process is dead, synthesises
      WorkResult(ok=False) carrying the IN-FLIGHT request_id (the id of the last
      submit() that has not yet produced a result), so a crash can be attributed
      to the right request.
    - Returns None if out_q is empty and the process is still running.
    """

    def __init__(
        self,
        worker_fn: Callable = _default_worker_fn,
    ) -> None:
        ctx = multiprocessing.get_context("spawn")
        self._in_q: Any = ctx.Queue()
        self._out_q: Any = ctx.Queue()
        self._proc: Any = ctx.Process(
            target=worker_fn,
            args=(self._in_q, self._out_q),
            daemon=True,
        )
        # request_id of the work currently in flight; -1 when idle.  Used to
        # attribute a crash (death without a result) to the right request so the
        # synthesised failure WorkResult carries the real id, not a placeholder.
        self._inflight_request_id: int = -1
        self._proc.start()

    def submit(self, work: WorkItem) -> None:
        self._inflight_request_id = work.request_id
        self._in_q.put(work)

    def poll(self) -> Optional[WorkResult]:
        try:
            result = self._out_q.get_nowait()
            # Normal result delivered: this request is no longer in flight.
            self._inflight_request_id = -1
            return result
        except queue.Empty:
            pass
        except Exception:
            pass  # queue might be in a broken state if process died abruptly

        if not self._proc.is_alive():
            return WorkResult(
                request_id=self._inflight_request_id,
                output_ids=[],
                ok=False,
                error=f"worker process died (exit code {self._proc.exitcode})",
                service_time=0.0,
            )
        return None

    def is_alive(self) -> bool:
        return self._proc.is_alive()

    def terminate(self) -> None:
        # Graceful shutdown: send the sentinel and give the worker a chance to
        # exit cleanly (e.g. a real MNN worker releasing device handles) BEFORE
        # resorting to SIGTERM.
        try:
            self._in_q.put(None, timeout=1.0)  # shutdown sentinel
        except Exception:
            pass
        try:
            self._proc.join(timeout=3.0)
        except Exception:
            pass
        # Force-kill only if the worker did not exit on the sentinel.
        try:
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=3.0)
        except Exception:
            pass
        # Close queues to avoid feeder-thread hang at exit.
        try:
            self._in_q.close()
            self._in_q.cancel_join_thread()
        except Exception:
            pass
        try:
            self._out_q.close()
            self._out_q.cancel_join_thread()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Lane
# ---------------------------------------------------------------------------

@dataclass
class Lane:
    """
    Per-lane state for one decode lane (CPU or GPU).

    _current_slot: the Slot currently OWNED by this lane (set by dispatch,
    cleared by reap).  Not included in __init__ signature; initialised to None
    via __post_init__.

    Prefix-cache affinity
    -----------------------------------------
    supports_prefix_cache : a capability of this lane's underlying
        decode engine, not a per-request toggle.  E.g. the MNN CPU lane
        supports reuse_kv/prompt_cache + KVMeta; the OpenCL/GPU lane's
        llm.cpp path has none (llm.cpp:462) and must be constructed with
        supports_prefix_cache=False.  Default False (conservative: a lane
        must OPT IN to being considered for affinity routing).
    prefix_lru_capacity / _prefix_lru : a small per-lane LRU of recently
        served prefix_ids.  This is a HEURISTIC MODEL of the decode
        engine's cache state, NOT a query into the real engine — the
        engine's actual cache contents are not observable from TABM (the
        engine is a black box; see slot.py's "NO KV region is stored in
        the slot" note).  "Recently dispatched here" is used as a proxy
        for "likely still warm in the engine's real cache"; it can be
        wrong in either direction (the engine may have evicted it sooner,
        or kept it longer).  See note_prefix() / has_prefix().
    """

    owner: LaneOwner
    backend: WorkerBackend = field(compare=False, repr=False)
    busy: bool = False
    failure_count: int = 0
    healthy: bool = True
    service_time_ewma: float = 0.0
    supports_prefix_cache: bool = False
    prefix_lru_capacity: int = 4
    _prefix_lru: "OrderedDict[str, None]" = field(
        default_factory=OrderedDict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        self._current_slot: Any = None
        self._pending_failure: Optional[WorkResult] = None
        self._pending_exception: Optional[BaseException] = None

    # ------------------------------------------------------------------
    # Prefix-cache affinity: heuristic warmth model
    # ------------------------------------------------------------------

    def note_prefix(self, prefix_id: Optional[str]) -> None:
        """
        Record that this lane just served ``prefix_id`` — mark it as the
        most-recently-used entry in the heuristic warmth LRU (see class
        docstring).  Evicts the least-recently-noted prefix once the LRU
        exceeds ``prefix_lru_capacity``.

        No-op when:
          - ``prefix_id`` is None (nothing to record), or
          - ``supports_prefix_cache`` is False (this lane's engine has no
            prefix cache at all — there is nothing meaningful to track).
        """
        if prefix_id is None or not self.supports_prefix_cache:
            return
        if prefix_id in self._prefix_lru:
            self._prefix_lru.move_to_end(prefix_id)
        else:
            self._prefix_lru[prefix_id] = None
            if len(self._prefix_lru) > self.prefix_lru_capacity:
                self._prefix_lru.popitem(last=False)  # evict least-recently-noted

    def has_prefix(self, prefix_id: Optional[str]) -> bool:
        """
        True if ``prefix_id`` is in this lane's heuristic warm set.

        Always False when ``prefix_id`` is None or ``supports_prefix_cache``
        is False — a lane without the underlying capability can never be
        "warm" for any prefix, by construction.  This is what makes
        prefix-cache affinity CAPABILITY-aware rather than just
        recency-aware: a GPU/OpenCL lane with no KVMeta support never wins
        an affinity comparison no matter how recently it ran a request.
        """
        if prefix_id is None or not self.supports_prefix_cache:
            return False
        return prefix_id in self._prefix_lru

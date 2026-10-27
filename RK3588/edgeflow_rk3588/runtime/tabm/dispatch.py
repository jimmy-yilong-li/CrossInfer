"""
TABM Core: LaneDispatcher.

Routes READY slots to healthy, idle lanes; reaps results; enforces the slot
pool invariant (in_use_count() + free_count() == capacity) at all times,
including after a worker crash on any lane.

Failure-isolation guarantee
---------------------------
If lane A's worker crashes during reap():
  - lane A.failure_count is incremented, lane A.healthy is set to False.
  - The in-flight slot is reclaimed via pool.release() → FREE (no leak).
  - All other lanes and their in-flight slots are UNAFFECTED.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

from RK3588.edgeflow_rk3588.runtime.tabm.lane import (
    Lane,
    RequestValidationError,
    WorkerTerminationError,
    WorkItem,
    WorkResult,
)
from RK3588.edgeflow_rk3588.runtime.tabm.slot import (
    LaneOwner,
    Slot,
    SlotPool,
    SlotState,
    SlotTransitionError,
)


@dataclass
class PrefixAffinityConfig:
    """
    Tunables for prefix-cache-affinity scoring in
    ``LaneDispatcher.select_lane``.

    All values are ESTIMATES the dispatcher uses to trade off base load
    balancing against a lane's likely-warm prefix cache; see
    ``affinity_saving_s``.

    alpha : float
        Linear scale on the estimated prefill-time saving.  1.0 = take the
        per-token estimate at face value.  A single knob for later
        calibration against real measurements without touching the scoring
        formula itself.
    per_token_prefill_s : float
        Estimated prefill seconds PER TOKEN, used to convert a warm
        prefix's token count into an estimated time saving.  In SECONDS,
        to stay unit-consistent with ``Lane.service_time_ewma``.
        Supply a rate calibrated for the target workload and co-running
        engines. Zero leaves the uncalibrated affinity bonus disabled;
        this reference distribution does not bundle device measurements.
    """
    alpha: float = 1.0
    per_token_prefill_s: float = 0.0


def affinity_saving_s(
    lane: Lane,
    prefix_id: Optional[str],
    prefix_len: int,
    cfg: PrefixAffinityConfig,
) -> float:
    """
    PURE helper.  Estimated prefill time (SECONDS) saved by dispatching a
    request carrying ``prefix_id`` / ``prefix_len`` onto ``lane``, versus a
    lane with no warm prefix cache.

    Returns 0.0 unless ``lane.supports_prefix_cache`` AND
    ``lane.has_prefix(prefix_id)`` both hold (``has_prefix`` already implies
    ``supports_prefix_cache``; the compound check makes this requirement
    explicit without consulting lane.py).
    Otherwise returns ``cfg.alpha * prefix_len * cfg.per_token_prefill_s`` —
    a linear ESTIMATE of prefill time avoided by skipping ``prefix_len``
    tokens of prefill.

    No mutation, no I/O: same inputs -> same output every call.  Does not
    call ``lane.note_prefix`` — callers decide when bookkeeping happens
    (``LaneDispatcher.dispatch`` only notes on a successful submit).
    """
    if not lane.supports_prefix_cache or not lane.has_prefix(prefix_id):
        return 0.0
    return cfg.alpha * prefix_len * cfg.per_token_prefill_s


# Values dispatch() records into LaneDispatcher.last_dispatch_info["decision"].
DISPATCH_DECISION_AFFINITY = "affinity"     # est_saved_s > 0.0 influenced the pick
DISPATCH_DECISION_BASELINE = "baseline"     # no prefix / no saving; load-balance only
DISPATCH_DECISION_QUEUE_WAIT = "queue_wait" # Deliberately NOT dispatched; waiting
                                            # for a warm-but-busy lane (saving > wait)
DISPATCH_DECISION_COMPLETION_WAIT = "completion_wait"


@dataclass
class QueueWaitConfig:
    """
    Queue-wait routing: when the request's prefix is warm ONLY on a BUSY
    lane and a cold idle lane exists, WAIT for the warm lane iff the
    saved-prefill estimate strictly exceeds the expected wait plus
    ``margin_s``. Disabled by default: every existing caller sees
    byte-identical dispatch behavior unless it opts in AND injects the
    warmth/saving callables (see LaneDispatcher.__init__).

    margin_s : guard band (seconds) added to the expected wait before the
        comparison; raises the bar for waiting. 0.0 means
        "wait iff saving > wait".
    """
    enabled: bool = False
    margin_s: float = 0.0


@dataclass(frozen=True)
class CompletionWaitConfig:
    """Opt-in W+S routing for requests without retained language context.

    Priors must come from measured submit-to-completion wall times on each
    lane. max_wait_s bounds deliberate waiting even if an estimate is wrong.
    A busy-lane prediction expires after its submit-time prior plus max_wait_s
    and is excluded until that work is reaped;
    it does not declare a live worker dead or relinquish its slot.
    Neither prefix-transfer cost nor DMC pressure is modeled here.
    """

    enabled: bool = False
    margin_s: float = 0.0
    max_wait_s: float = 5.0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        for name in ("margin_s", "max_wait_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")


def expected_remaining_s(lane: Lane, now: Optional[float] = None) -> float:
    """
    PURE-ish helper (reads the clock when ``now`` is None): estimated time
    (seconds) until ``lane`` frees, = service_time_ewma minus time already
    spent busy, floored at 0. If the lane has no ``t_busy_since`` stamp
    (stamped by dispatch() on successful submit; absent e.g. for a lane
    made busy by external harness code), falls back to the FULL ewma —
    conservative in the queue-wait direction (over-estimating the wait
    biases AGAINST waiting, never toward a bogus wait).
    """
    t0 = getattr(lane, "t_busy_since", None)
    if t0 is None:
        return max(0.0, lane.service_time_ewma)
    if now is None:
        now = time.time()
    return max(0.0, lane.service_time_ewma - (now - t0))


@dataclass
class ReapRecord:
    """
    Result of reaping one lane, with the slot's accounting SNAPSHOTTED before the
    pool transition.

    Why a snapshot: on the failure path ``reap()`` calls ``pool.release(slot)``,
    which resets the slot header (request_id → -1, copy_count/copy_bytes → 0,
    timestamps → 0). If a caller read these off the slot AFTER
    reap, the failure record would be all zeros and the request_id would be lost.
    ReapRecord captures them BEFORE the transition so failure records keep the
    real request_id and copy accounting.

    Fields
    ------
    slot         : the Slot object (post-transition: DONE on success, FREE on
                   failure).  Its header is already reset on the failure path —
                   read accounting from the snapshot fields below, not the slot.
    result       : the WorkResult returned by the lane's backend.
    request_id   : slot.request_id captured BEFORE the pool transition.
    lane         : slot.lane_owner (CPU/GPU) captured before release.
    copy_count   : slot.copy_count captured before release.
    copy_bytes   : slot.copy_bytes captured before release.
    t_admit      : slot.t_admit captured before release.
    t_ready      : slot.t_ready captured before release.
    t_done       : completion time — slot.t_done after complete() on success;
                   the failure-observation time on the failure path.
    reuse_count  : slot.reuse_count captured before release (kept across release,
                   but snapshotted here for a consistent record).
    """

    slot:        Slot
    result:      WorkResult
    request_id:  int
    lane:        LaneOwner
    copy_count:  int
    copy_bytes:  int
    t_admit:     float
    t_ready:     float
    t_done:      float
    reuse_count: int


class LaneDispatcher:
    """
    Dispatches READY slots to decode lanes and reaps results.

    Parameters
    ----------
    pool : SlotPool
        The shared slot pool.  Dispatcher calls assign/complete/release.
    lanes : list[Lane]
        The available decode lanes (CPU and/or GPU).
    ewma_alpha : float
        Smoothing factor for the per-lane service_time_ewma.
        ewma ← alpha * new_sample + (1 − alpha) * ewma_prev
    affinity_config : PrefixAffinityConfig, optional
        Tunables for prefix-cache-affinity scoring (see module-level
        ``affinity_saving_s``).  Defaults to ``PrefixAffinityConfig()``
        (no calibrated affinity bonus) when not given. This is zero-effect for
        any request that carries no prefix (``prefix_id=None``), so
        existing callers that never set a prefix on their slots see no
        behavior change regardless of this config.
    """

    def __init__(
        self,
        pool: SlotPool,
        lanes: list[Lane],
        ewma_alpha: float = 0.3,
        affinity_config: Optional[PrefixAffinityConfig] = None,
        queue_wait_config: Optional["QueueWaitConfig"] = None,
        warmth: Optional[Callable] = None,
        saving: Optional[Callable] = None,
        completion_wait_config: Optional[CompletionWaitConfig] = None,
        monotonic: Optional[Callable[[], float]] = None,
    ) -> None:
        """
        queue_wait_config / warmth / saving (all optional; omitting ANY
        of them disables queue-wait routing):
          queue_wait_config : QueueWaitConfig with enabled=True to opt in.
          warmth  : callable (lane, prefix_id) -> bool. Registry-backed
                    warmth (prefix_sched.make_registry_warmth) — NOT the
                    lane-LRU heuristic, which keeps serving the
                    idle-lane affinity bonus unchanged.
          saving  : callable (prefix_id, prefix_len) -> (seconds, source_str)
                    with est_/measured_ source tagging
                    (prefix_sched.make_saving_estimator).
        """
        self._pool = pool
        self._lanes = lanes
        self._ewma_alpha = ewma_alpha
        self._affinity_config = (
            affinity_config if affinity_config is not None else PrefixAffinityConfig()
        )
        self._queue_wait = queue_wait_config
        self._warmth = warmth
        self._saving = saving
        self._completion_wait = completion_wait_config or CompletionWaitConfig()
        self._monotonic = monotonic
        self._completion_busy_since: dict[int, float] = {}
        self._completion_prediction_expires: dict[int, float] = {}
        self._completion_stale_lanes: set[int] = set()
        self._completion_wait_started: dict[int, tuple[Slot, int, float]] = {}
        self._completion_reservations: dict[int, Lane] = {}
        if self._completion_wait.enabled:
            if queue_wait_config is not None and queue_wait_config.enabled:
                raise ValueError("completion waiting cannot combine with prefix queue waiting")
            for lane in lanes:
                value = lane.service_time_ewma
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError("completion waiting requires positive finite service priors")
        # Populated only on a SUCCESSFUL dispatch() submit (see dispatch()).
        # Single-scheduler-thread contract (see SlotPool's threading
        # contract): dispatch() calls are never concurrent, so this plain
        # attribute is safe for the harness/metrics layer to read right
        # after calling dispatch().
        self.last_dispatch_info: Optional[dict] = None
        self._undelivered_reaps: list[ReapRecord] = []

    def _now_monotonic(self) -> float:
        return self._monotonic() if self._monotonic is not None else time.monotonic()

    def begin_dispatch_pass(self) -> None:
        """Start an ordered pass over READY slots; earlier waits reserve work.

        Reservations are predictions, not backend submissions. Recompute them
        every pass so completed, expired and cancelled work never lingers.
        """
        self._completion_reservations.clear()
        self._completion_wait_started = {
            request_id: entry
            for request_id, entry in self._completion_wait_started.items()
            if (entry[0].state is SlotState.READY and entry[0].request_id == request_id
                and entry[0].reuse_count == entry[1])
        }

    def _clear_completion_wait(self, request_id: int) -> None:
        self._completion_wait_started.pop(request_id, None)
        self._completion_reservations.pop(request_id, None)

    def _clear_completion_prediction(self, lane: Lane) -> None:
        self._completion_busy_since.pop(id(lane), None)
        self._completion_prediction_expires.pop(id(lane), None)
        self._completion_stale_lanes.discard(id(lane))

    def _should_wait_for_completion(self, slot: Slot, idle: Lane) -> bool:
        config = self._completion_wait
        if not config.enabled:
            return False
        if getattr(slot, "prefix_id", None) is not None:
            raise ValueError("completion waiting currently requires no retained prefix")
        now = self._now_monotonic()
        request_id = slot.request_id
        self._completion_reservations.pop(request_id, None)
        entry = self._completion_wait_started.get(request_id)
        started = (entry[2] if entry is not None and entry[0] is slot
                   and entry[1] == slot.reuse_count else now)
        remaining_budget = config.max_wait_s - (now - started)
        if remaining_budget <= 0:
            return False

        candidates = []
        for target in self._lanes:
            if not target.healthy or not target.busy or not target.backend.is_alive():
                continue
            if id(target) in self._completion_stale_lanes:
                continue
            service = target.service_time_ewma
            if not math.isfinite(service) or service <= 0:
                continue
            since = self._completion_busy_since.get(id(target))
            expires = self._completion_prediction_expires.setdefault(
                id(target), (now if since is None else since) + service + config.max_wait_s,
            )
            if now >= expires:
                # Latch per lane, not per arriving request. Only observed
                # completion restores confidence; revised priors do not.
                self._completion_stale_lanes.add(id(target))
                continue
            remaining = service if since is None else max(0.0, service - (now - since))
            queued = sum(lane is target for lane in self._completion_reservations.values())
            wait = remaining + queued * service
            if wait <= remaining_budget:
                candidates.append((wait + service, wait, target))
        if not candidates:
            return False
        completion, wait, target = min(candidates, key=lambda item: item[0])
        if completion + config.margin_s >= idle.service_time_ewma:
            return False
        self._completion_wait_started[request_id] = (slot, slot.reuse_count, started)
        self._completion_reservations[request_id] = target
        self.last_dispatch_info = {
            "request_id": request_id,
            "decision": DISPATCH_DECISION_COMPLETION_WAIT,
            "wait_for_lane": target.owner,
            "idle_lane": idle.owner,
            "predicted_wait_completion_s": completion,
            "predicted_idle_completion_s": idle.service_time_ewma,
            "predicted_wait_s": wait,
        }
        return True

    # ------------------------------------------------------------------
    # Lane selection
    # ------------------------------------------------------------------

    @staticmethod
    def _mark_failed(lane: Lane) -> None:
        if lane.healthy:
            lane.failure_count += 1
            lane.healthy = False

    def _refresh_idle_health(self, lane: Lane) -> None:
        if lane.healthy and not lane.busy and not lane.backend.is_alive():
            self._mark_failed(lane)

    def _confirm_stop(self, lane: Lane) -> None:
        """No failed lease may be recycled while its worker could still use it."""
        self._mark_failed(lane)
        try:
            lane.backend.terminate()
            if lane.backend.is_alive():
                raise WorkerTerminationError("backend is still alive after terminate()")
        except Exception as exc:
            slot = lane._current_slot
            error = WorkerTerminationError(
                f"lane {lane.owner.value} termination could not be confirmed; "
                f"owned slot for request {slot.request_id} retained"
            )
            primary = lane._pending_exception
            if primary is not None:
                error.__notes__ = [
                    f"worker operation failed: {primary}",
                    *getattr(primary, "__notes__", ()),
                    f"worker shutdown failed: {exc!r}",
                    *getattr(exc, "__notes__", ()),
                ]
                raise error from primary
            raise error from exc

    def select_lane(
        self,
        prefix_id: Optional[str] = None,
        prefix_len: int = 0,
    ) -> Optional[Lane]:
        """
        Return the best HEALTHY, not-busy, LIVE lane.

        A lane is a candidate only if it is healthy, idle, AND its backend is
        still alive.  Liveness matters because a backend can die after a
        successful result (lane left healthy + idle): without the liveness
        filter such a lane could be selected and the next submit would fail.

        Tie-break priority (lowest wins):
          1. failure_count
          2. service_time_ewma, reduced by the estimated prefix-cache-
             affinity saving for this lane (0.0 when no prefix is given, or
             the lane is not warm for it — see ``affinity_saving_s``).

        Prefix-cache affinity
        -----------------------------------------
        A lane that is warm for ``prefix_id`` (cache-capable AND
        recently served it — see lane.py) is preferred in proportion to its
        ESTIMATED prefill-time saving.  This still sits BELOW failure_count
        in priority and is netted against the lane's own EWMA, so a slow or
        flaky "warm" lane does not automatically win: this is base load
        balancing PLUS an affinity bonus, not a replacement for it.

        With the default ``prefix_id=None``, this method is BYTE-IDENTICAL
        to the pre-affinity implementation: ``affinity_saving_s`` always
        returns 0.0 when ``prefix_id`` is None (``Lane.has_prefix(None)`` is
        always False), so the sort key reduces exactly to
        ``(failure_count, service_time_ewma)`` — the original formula,
        regardless of any lane's capability or warm state.

        Returns None if no lane is available.
        """
        for lane in self._lanes:
            self._refresh_idle_health(lane)
        candidates = [
            l for l in self._lanes
            if l.healthy and not l.busy and l.backend.is_alive()
        ]
        if not candidates:
            return None
        cfg = self._affinity_config
        # Third key element = deterministic tie-break pinning the STRICT ">"
        # contract: a warm lane wins only if its saving strictly beats the
        # EWMA gap. At an EXACT tie (adjusted scores equal) the lane WITHOUT
        # affinity is preferred (False < True), instead of falling back to
        # list order (min() stability), which a refactor could silently flip.
        # No-prefix path: saving is 0.0 for every lane, so the third element
        # is False for all — the key reduces exactly to the original
        # (failure_count, service_time_ewma) ordering.
        def _key(l: Lane):
            saving = affinity_saving_s(l, prefix_id, prefix_len, cfg)
            return (l.failure_count, l.service_time_ewma - saving, saving > 0.0)

        return min(candidates, key=_key)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, slot: Slot) -> Optional[Lane]:
        """
        Select a lane and submit work from *slot*.

        The slot must belong to this pool and be READY, including when a
        scheduling policy would otherwise wait instead of calling assign().

        On success:
          - pool.assign(slot, lane.owner)  →  slot state: OWNED
          - lane.busy = True
          - WorkItem built from slot fields; lane.backend.submit() called.
          - lane.note_prefix(prefix_id) — records this lane as warm for the
            request's prefix (no-op if prefix_id is None or the lane does
            not support a prefix cache at all; see lane.py).
          - self.last_dispatch_info is set (see below).
          - Returns the Lane.

        On no lane available:
          - Returns None; slot is UNTOUCHED (still READY, caller re-queues).
          - last_dispatch_info is left UNCHANGED — bookkeeping only happens
            on a successful submit.

        On a READY slot's deadline expiring before lane selection:
          - Releases the slot and returns None without touching any lane.
          - Clears a matching queue-wait decision; other snapshots are kept.

        On QUEUE-WAIT (only when queue_wait_config.enabled AND both
        warmth/saving callables were injected AND the request carries a
        prefix):
          - Returns None; slot is UNTOUCHED (still READY, caller re-queues —
            the wait materializes through the caller's retry loop; when the
            warm lane frees, normal affinity routes the request there).
          - last_dispatch_info IS set (deliberate, documented revision of
            the bookkeeping contract above) to::

                {"request_id", "decision": DISPATCH_DECISION_QUEUE_WAIT,
                 "wait_for_lane": LaneOwner, "est_saved_s": float,
                 "saving_source": "measured_entry"|"measured_table"|"est",
                 "expected_wait_s": float}

            Callers distinguishing wait-None from no-lane-None must check
            ``info["request_id"] == slot.request_id`` and the decision tag.
          - LIVELOCK NOTE: the policy is stateless; if the warm lane never
            frees (hang) the caller's retry loop must bound retries/deadline.
            A lane that turns unhealthy stops
            passing the warmth/candidate filters, ending the wait naturally.

        On submit() raising:
          - RequestValidationError leaves lane health and failure_count unchanged.
          - Other exceptions increment failure_count and set lane.healthy=False.
          - pool.release(slot) → slot state: FREE  (no leak).
          - lane.busy remains False, lane._current_slot remains None.
          - Returns None.
          - NEITHER lane.note_prefix() NOR last_dispatch_info is touched —
            bookkeeping only happens on a successful submit.

        Prefix-cache affinity
        -----------------------------------------
        Reads ``prefix_id = getattr(slot, "prefix_id", None)`` and
        ``prefix_len = getattr(slot, "prefix_len", 0)`` — the same
        ad-hoc-attribute pattern already used below for ``shm_descriptor``;
        NO slot ABI change.  A plain Slot with neither attribute dispatches
        exactly as before (prefix_id=None routes through select_lane's
        no-prefix, byte-identical path).

        On a SUCCESSFUL submit only, ``self.last_dispatch_info`` is set to::

            {
                "request_id": slot.request_id,
                "lane":       lane.owner,   # LaneOwner.CPU / LaneOwner.GPU
                "prefix_hit": bool,         # lane was warm for prefix_id
                                            # BEFORE this call noted it
                "est_saved_s": float,       # affinity_saving_s(...) at
                                            # selection time (seconds)
                "decision":   DISPATCH_DECISION_AFFINITY or
                              DISPATCH_DECISION_BASELINE,
            }

        for the harness/metrics layer to read.  Single-scheduler-thread
        contract (see SlotPool's threading contract): dispatch() calls are
        never concurrent, so this plain attribute is safe to read
        immediately after a dispatch() call.
        """
        self._pool._require_member(slot)
        if slot.state is not SlotState.READY:
            raise SlotTransitionError(
                f"dispatch requires READY; slot is {slot.state.value}"
            )

        # Reject before lane selection, including when every lane is busy or
        # prefix affinity would otherwise keep this request waiting.
        if (
            slot.deadline_monotonic is not None
            and self._now_monotonic() >= slot.deadline_monotonic
        ):
            info = self.last_dispatch_info
            if (
                info is not None and info.get("request_id") == slot.request_id
                and info.get("decision") in {
                    DISPATCH_DECISION_QUEUE_WAIT, DISPATCH_DECISION_COMPLETION_WAIT,
                }
            ):
                self.last_dispatch_info = None
            self._clear_completion_wait(slot.request_id)
            self._pool.release(slot)
            return None

        prefix_id = getattr(slot, "prefix_id", None)
        prefix_len = getattr(slot, "prefix_len", 0)

        lane = self.select_lane(prefix_id=prefix_id, prefix_len=prefix_len)

        # ---- Queue-wait (see docstring contract) -------------------------
        # Fires only when: opted in + callables injected + request has a
        # prefix + an idle lane EXISTS but is NOT warm + some warm lane is
        # busy + saving strictly beats expected wait + margin. When no idle
        # lane exists at all, the no-lane None path below remains unchanged
        # (waiting is not a DECISION when there is no alternative).
        if (
            self._queue_wait is not None and self._queue_wait.enabled
            and self._warmth is not None and self._saving is not None
            and prefix_id is not None and lane is not None
            and not any(
                candidate.healthy and not candidate.busy
                and candidate.backend.is_alive() and self._warmth(candidate, prefix_id)
                for candidate in self._lanes
            )
        ):
            busy_warm = [
                l for l in self._lanes
                if l.healthy and l.busy and l.backend.is_alive()
                and self._warmth(l, prefix_id)
            ]
            if busy_warm:
                now = time.time()
                w = min(busy_warm, key=lambda l: expected_remaining_s(l, now))
                wait_s = expected_remaining_s(w, now)
                saved_s, source = self._saving(prefix_id, prefix_len)
                if saved_s > wait_s + self._queue_wait.margin_s:
                    self.last_dispatch_info = {
                        "request_id": slot.request_id,
                        "decision": DISPATCH_DECISION_QUEUE_WAIT,
                        "wait_for_lane": w.owner,
                        "est_saved_s": saved_s,
                        "saving_source": source,
                        "expected_wait_s": wait_s,
                    }
                    return None

        if lane is None:
            return None
        if self._should_wait_for_completion(slot, lane):
            return None

        self._clear_completion_wait(slot.request_id)
        self._pool.assign(slot, lane.owner)    # READY → OWNED
        work = WorkItem(
            request_id=slot.request_id,
            prompt_ids=list(slot.prompt_ids),
            max_decode_tokens=slot.max_decode_tokens,
            prompt_text=slot.prompt_text,
            # The caller sets slot.shm_descriptor (a SlotBufferDescriptor as
            # dict) before mark_ready when it wants embeds mode; absent (the
            # common case today) => embeds_desc=None => text mode.
            embeds_desc=getattr(slot, "shm_descriptor", None),
            # Same ad-hoc-attribute pattern; all three set => the
            # adapter submits the worker's `prefix` REQ mode (PRODUCTION
            # prefix-cache execution). Cleared by slot._reset_header on
            # release so recycled slots cannot retain stale request fields.
            prefix_name=getattr(slot, "prefix_name", None),
            prefix_text=getattr(slot, "prefix_text", None),
            suffix_text=getattr(slot, "suffix_text", None),
            deadline_monotonic=slot.deadline_monotonic,
        )
        # submit may send a partial request before raising. Record ownership
        # first so an unresolved shutdown never loses the shared-memory lease.
        lane.busy = True
        lane._current_slot = slot
        lane.t_busy_since = time.time()
        if self._completion_wait.enabled:
            self._clear_completion_prediction(lane)
            started = self._now_monotonic()
            self._completion_busy_since[id(lane)] = started
            self._completion_prediction_expires[id(lane)] = (
                started + lane.service_time_ewma + self._completion_wait.max_wait_s
            )
        try:
            lane.backend.submit(work)
        except RequestValidationError:
            self._pool.release(slot)
            lane.busy = False
            lane._current_slot = None
            lane.t_busy_since = None
            self._clear_completion_prediction(lane)
            return None
        except Exception as exc:
            lane._pending_failure = WorkResult(slot.request_id, [], False, str(exc), 0.0)
            lane._pending_exception = exc
            self._confirm_stop(lane)
            self._pool.release(slot)           # OWNED → FREE (no leak)
            lane._pending_failure = None
            lane._pending_exception = None
            lane.busy = False
            lane._current_slot = None
            lane.t_busy_since = None
            self._clear_completion_prediction(lane)
            return None

        # ---- Prefix-cache affinity bookkeeping (SUCCESSFUL submit only) ---
        # Evaluate warmth/saving BEFORE note_prefix() below updates the LRU,
        # so prefix_hit/est_saved_s describe the state this dispatch
        # decision was made under, not the state after it.
        prefix_hit = lane.has_prefix(prefix_id)
        est_saved_s = affinity_saving_s(lane, prefix_id, prefix_len, self._affinity_config)
        lane.note_prefix(prefix_id)
        self.last_dispatch_info = {
            "request_id": slot.request_id,
            "lane": lane.owner,
            "prefix_hit": prefix_hit,
            "est_saved_s": est_saved_s,
            "decision": (
                DISPATCH_DECISION_AFFINITY if est_saved_s > 0.0
                else DISPATCH_DECISION_BASELINE
            ),
        }
        return lane

    # ------------------------------------------------------------------
    # Reap
    # ------------------------------------------------------------------

    def reap(self) -> list[ReapRecord]:
        """
        Poll every busy lane.  For each ready result, build a ReapRecord whose
        accounting is SNAPSHOTTED before the pool transition (see ReapRecord).

        ok=True (SUCCESS):
          slot.output_ids = list(result.output_ids)   # decoded tokens written back
          pool.complete(slot)  →  slot state: DONE
          lane.busy = False; lane.service_time_ewma updated via EWMA.

        ok=False (FAILURE):
          pool.release(slot)  →  slot state: FREE  (no slot leak)
          lane.busy = False; (EWMA not updated on failure)
          Lane health depends on the failure's classification
          (``WorkResult.lane_fault``):
            lane_fault=True (default — worker death, protocol violation,
              unclassified failure): lane.failure_count += 1;
              lane.healthy = False (lane fenced, pre-existing behaviour).
            lane_fault=False (request-level failure from a live,
              protocol-correct worker — well-formed RES ok=0): the lane
              stays healthy and selectable; only the request fails.
          Both success and failure clear ``lane.t_busy_since`` so
          ``expected_remaining_s`` falls back to the full-EWMA conservative
          estimate for any lane busied outside dispatch().

        Success vs failure is decided SOLELY by result.ok.  A delivered result
        with ok=True is a success even if backend.is_alive() later returns
        False — liveness does NOT downgrade a delivered result.  Liveness only
        (a) detects a worker that died WITHOUT producing any result (poll()
        returns None while is_alive() is False → synthesise an ok=False result),
        and (b) gates FUTURE lane selection (see select_lane).

        The OTHER lanes in self._lanes are NEVER touched — failure isolation is
        enforced by processing each lane independently.

        Returns a list of ReapRecord for all lanes that produced a result this
        call.  Because pool.release() zeros the slot header on failure, callers
        MUST read accounting (request_id, copy_bytes, ...) from the ReapRecord,
        not from the (now-reset) slot.
        """
        # A later lane may raise while earlier completions are already DONE.
        # Keep those records until a successful caller handoff, exactly once.
        collected = self._undelivered_reaps
        self._collect_reaps(collected)
        for record in collected:
            if not record.result.ok:
                self._pool.release(record.slot)
        self._undelivered_reaps = []
        return collected

    def _collect_reaps(self, collected: list[ReapRecord]) -> None:
        for lane in self._lanes:
            self._refresh_idle_health(lane)
            if not lane.busy:
                continue

            result = lane._pending_failure
            if result is None:
                try:
                    result = lane.backend.poll()
                except Exception as exc:
                    slot = lane._current_slot
                    result = WorkResult(
                        slot.request_id if slot is not None else -1,
                        [], False, f"worker poll failed: {exc}", 0.0,
                    )
                    lane._pending_failure = result
                    lane._pending_exception = exc
                    self._confirm_stop(lane)
                    self._pool.release(slot)
                    lane._pending_failure = None
                    lane._pending_exception = None
                    lane._current_slot = None
                    lane.busy = False
                    lane.t_busy_since = None
                    self._clear_completion_prediction(lane)
                    # Transport exceptions were historically raised, not
                    # converted to request failures. Preserve identity/notes.
                    raise

            # If poll() returned nothing, check whether the worker died
            # without producing output (e.g. process killed mid-work).
            if result is None and not lane.backend.is_alive():
                slot = lane._current_slot
                result = WorkResult(
                    request_id=slot.request_id if slot is not None else -1,
                    output_ids=[],
                    ok=False,
                    error="worker died without returning a result",
                    service_time=0.0,
                )

            if result is None:
                # Worker still running, nothing to collect yet.
                continue

            slot = lane._current_slot
            if slot is None:
                continue

            # The dispatcher is the final correlation boundary shared by all
            # WorkerBackend implementations.  A stale/crossed result must not
            # complete the slot currently owned by this lane, even when the
            # backend incorrectly labels it ok=True.
            if result.request_id != slot.request_id:
                actual_request_id = result.request_id
                result = WorkResult(
                    request_id=slot.request_id,
                    output_ids=[],
                    ok=False,
                    error=(
                        "protocol error: worker result request_id "
                        f"{actual_request_id} does not match owned slot "
                        f"request_id {slot.request_id}"
                    ),
                    service_time=result.service_time,
                    extra=result.extra,
                )

            # Snapshot the slot's accounting BEFORE any pool transition: the
            # failure path's pool.release() resets request_id/copy_*/timestamps,
            # so this is the only point at which failure-path metrics survive.
            snap_request_id  = slot.request_id
            snap_lane        = slot.lane_owner
            snap_copy_count  = slot.copy_count
            snap_copy_bytes  = slot.copy_bytes
            snap_t_admit     = slot.t_admit
            snap_t_ready     = slot.t_ready
            snap_reuse_count = slot.reuse_count

            # Pool transition happens BEFORE clearing lane state so that a
            # transition exception cannot leave the lane cleared with the slot
            # un-transitioned (safer ordering).
            if result.ok:
                slot.output_ids = list(result.output_ids)   # write decoded tokens back
                self._pool.complete(slot)          # OWNED → DONE (stamps t_done)
                t_done = slot.t_done
                lane._current_slot = None
                lane.busy = False
                lane.t_busy_since = None   # expected_remaining_s must fall
                                           # back to the full ewma, not a stale stamp
                alpha = self._ewma_alpha
                service = result.service_time
                started = self._completion_busy_since.get(id(lane))
                self._clear_completion_prediction(lane)
                if self._completion_wait.enabled:
                    # External submissions have no compatible wall sample.
                    # Keep their prior rather than mixing in engine-only time.
                    service = (None if started is None
                               else self._now_monotonic() - started)
                if service is not None:
                    lane.service_time_ewma = (
                        alpha * service
                        + (1.0 - alpha) * lane.service_time_ewma
                    )
            else:
                # Failure path: stamp a real failure-observation time before
                # release() zeros the slot, then reclaim the slot.
                #
                # lane_fault classification (see WorkResult.lane_fault):
                # only a LANE-level failure (worker death, protocol violation,
                # unclassified ok=False — the conservative default) fences the
                # lane.  A REQUEST-level failure from a live, protocol-correct
                # worker (well-formed RES ok=0: bad prompt, bad descriptor,
                # shm errors) reclaims the slot and leaves the lane healthy —
                # otherwise one malformed request would permanently take a
                # single-lane deployment offline.
                t_done = time.time()
                if getattr(result, "lane_fault", True):
                    lane._pending_failure = result
                    self._confirm_stop(lane)
                # Retain the slot until this completion is delivered. If a
                # later lane raises, recycling it now would let the deferred
                # record refer to a different, newly active request.
                self._pool.complete(slot)          # OWNED → DONE until handoff
                lane._pending_failure = None
                lane._pending_exception = None
                lane._current_slot = None
                lane.busy = False
                lane.t_busy_since = None
                self._clear_completion_prediction(lane)

            self._refresh_idle_health(lane)
            collected.append(ReapRecord(
                slot=slot,
                result=result,
                request_id=snap_request_id,
                lane=snap_lane,
                copy_count=snap_copy_count,
                copy_bytes=snap_copy_bytes,
                t_admit=snap_t_admit,
                t_ready=snap_t_ready,
                t_done=t_done,
                reuse_count=snap_reuse_count,
            ))

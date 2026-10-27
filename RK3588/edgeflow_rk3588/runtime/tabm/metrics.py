"""
TABM Core: metrics + copy-chain + two-traffic accounting.

Records per-request observability data (RequestMetrics) and aggregates it
into a structured summary (MetricsCollector.summary()).

TWO-TRAFFIC SPLIT
-----------------
TABM controls exactly ONE kind of memory traffic and can only ESTIMATE the
other.  The summary keeps them STRICTLY SEPARATE:

  tabm_payload_bytes    — MEASURED.  The sum of copy_bytes from the
                          per-slot copy-chain (e.g. write_embedding calls).
                          TABM actually moved these bytes; this is real data.

  est_weight_stream_bytes — ESTIMATED.  The LPDDR weight-streaming that the
                            decode engine performs while generating tokens.
                            TABM does NOT control this traffic; it bounds it
                            indirectly via the admission concurrency cap.
                            Estimated as model_bytes × total_ok_decode_tokens
                            (batch=1: each decoded token streams the model
                            weights approximately once through LPDDR).

                            This estimate is model-based, not read from
                            hardware bandwidth counters.

  weight_stream_is_estimated: True  — always present on the traffic dict to
                                       signal that the weight-stream figure
                                       is a model-based estimate, NOT a
                                       measurement.

COPY-CHAIN ACCOUNTING (fail-closed)
------------------------------------
A request that performed no accounted copies appears with copy_bytes=0 in
the summary.  Zero is visible; it is NOT silently omitted.  This is
intentional: if a request shows copy_bytes=0 when embeddings should have
been written, that is a signal that copy accounting was skipped, not that
the request was "zero-copy" in a meaningful sense.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Optional

from RK3588.edgeflow_rk3588.runtime.tabm.admission import Decision
from RK3588.edgeflow_rk3588.runtime.tabm.slot import LaneOwner


# ---------------------------------------------------------------------------
# RequestMetrics
# ---------------------------------------------------------------------------

@dataclass
class RequestMetrics:
    """
    Observability record for a single TABM request.

    Populated by the lane / dispatch layer after the request completes (or
    fails).  Passed to MetricsCollector.record().

    Fields
    ------
    request_id        : caller-assigned request identifier.
    decision          : admission outcome (ACCEPT / DELAY / DROP).
    lane              : lane that ran this request; None if not dispatched.
    slot_reuse_count  : slot.reuse_count at time of dispatch.
    copy_count        : number of copy operations (from slot.copy_count).
    copy_bytes        : total bytes copied via the copy-chain (MEASURED).
    t_admit           : epoch seconds at admission.
    t_ready           : epoch seconds when slot was READY.
    t_assign          : epoch seconds when lane was assigned (OWNED).
    t_done            : epoch seconds when decoding completed.
    decode_tokens     : number of tokens the decode engine generated.
    service_time      : seconds the decode engine spent on this request.
    ok                : True if the request completed successfully.
    error             : error description if ok=False; None otherwise.

    Prefix-cache affinity (est_ = ESTIMATE, not measured)
    ------------------------------------------------------------------------
    prefix_id             : the request's prefix identity, if any (mirrors
                             RequestCost.prefix_id); None = no prefix.
    prefix_hit            : True if the lane chosen for this request was
                             already warm for prefix_id (see
                             dispatch.py:LaneDispatcher.dispatch /
                             last_dispatch_info["prefix_hit"]).
    est_prefill_saved_us  : ESTIMATED prefill microseconds avoided by the
                             warm-lane routing decision (see
                             dispatch.affinity_saving_s; SECONDS there,
                             converted to microseconds by the caller for
                             this field).  0 when unused/no saving.
    affinity_decision     : dispatch.DISPATCH_DECISION_AFFINITY /
                             DISPATCH_DECISION_BASELINE, or None if the
                             dispatch layer did not report one.
    """

    request_id:       int
    decision:         Decision
    lane:             Optional[LaneOwner]
    slot_reuse_count: int   = 0
    copy_count:       int   = 0
    copy_bytes:       int   = 0      # MEASURED TABM-controlled payload traffic
    t_admit:          float = 0.0
    t_ready:          float = 0.0
    t_assign:         float = 0.0
    t_done:           float = 0.0
    decode_tokens:    int   = 0
    service_time:     float = 0.0    # seconds spent decoding (from the lane)
    ok:               bool  = False
    error:            Optional[str] = None
    # ---- Prefix-cache affinity (additive; all est_/estimated) ----
    prefix_id:             Optional[str] = None
    prefix_hit:            bool = False
    est_prefill_saved_us:  int  = 0
    affinity_decision:     Optional[str] = None

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def queue_wait(self) -> float:
        """Seconds from admission to lane assignment; clamped to 0."""
        return max(0.0, self.t_assign - self.t_admit)

    @property
    def total_latency(self) -> float:
        """End-to-end seconds from admission to completion; clamped to 0."""
        return max(0.0, self.t_done - self.t_admit)

    @property
    def decode_tps(self) -> float:
        """
        Decode tokens per second for this request.

        Returns 0.0 if service_time <= 0 (avoids ZeroDivisionError).
        """
        if self.service_time <= 0.0:
            return 0.0
        return self.decode_tokens / self.service_time


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

_WEIGHT_STREAM_NOTE = (
    "weight stream is model_bytes*tokens; LPDDR not directly measurable on RK3588."
    " tabm_payload_bytes covers all requests; est_weight_stream_bytes covers ok requests only"
    " (account for this before taking ratios)."
)


class MetricsCollector:
    """
    Aggregates RequestMetrics records and produces a structured summary.

    Parameters
    ----------
    model_bytes : int
        Byte size of the decode model weights.  Used as the basis for the
        estimated weight-stream traffic (est_weight_stream_bytes).
        Example: 265_000_000 for a ~265 MB model.
    """

    def __init__(self, model_bytes: int) -> None:
        self._model_bytes: int = model_bytes
        self._records: list[RequestMetrics] = []

    def record(self, m: RequestMetrics) -> None:
        """Append a RequestMetrics to the collector."""
        self._records.append(m)

    def summary(self) -> dict:
        """
        Return an aggregated summary dict.

        Keys
        ----
        counts : dict
            total, accepted, delayed, dropped, ok, failed
        per_lane : dict
            {CPU: n, GPU: n} — dispatched request counts per lane type.
        throughput : dict
            aggregate_decode_tps: sum(decode_tokens) / sum(service_time)
                over ok requests; 0.0 if no ok requests or total service_time
                is zero.  This is SERVICE-TIME-NORMALIZED — the average per-lane
                decode rate, NOT aggregate serving throughput.  Concurrent lanes
                do NOT add up here (two 1 s / 32-tok lanes report 32, not 64).
            wallclock_aggregate_tps: sum(decode_tokens over ok) / wall-clock span
                (max t_done − min t_assign over ok requests); 0.0 if no ok
                requests or the span is non-positive.  This is the REAL aggregate
                serving throughput — concurrent lanes overlap in the wall-clock
                window, so the two-lane example above reports 64.
        latency : dict
            p50_latency, p95_latency — percentiles over ok requests with
            t_done > t_admit (positive total_latency only).  0.0 if no valid
            latency observations.
        traffic : dict
            tabm_payload_bytes     : sum of copy_bytes over ALL requests
                                     (measured; includes ok and failed).
            est_weight_stream_bytes: model_bytes × sum(decode_tokens) over
                                     ok requests only (estimated).
            weight_stream_is_estimated: True
            note: human-readable caveat about LPDDR measurability.
        affinity : dict (all zeros when no request ever
                   carries a prefix — see RequestMetrics.prefix_id)
            requests_with_prefix      : count of records with prefix_id is
                                         not None (regardless of ok/hit).
            prefix_hits                : count of records with
                                         prefix_hit is True.
            est_prefill_saved_us_total : sum of est_prefill_saved_us over
                                         ALL records (an ESTIMATE; 0 for any
                                         record that did not report one).
        """
        records = self._records

        # ---- Counts --------------------------------------------------------
        total    = len(records)
        accepted = sum(1 for r in records if r.decision is Decision.ACCEPT)
        delayed  = sum(1 for r in records if r.decision is Decision.DELAY)
        dropped  = sum(1 for r in records if r.decision is Decision.DROP)
        ok_count = sum(1 for r in records if r.ok)
        # "failed" means DISPATCHED AND FAILED. DROP/DELAY records also carry
        # ok=False, but an admission rejection never reached a decode engine —
        # counting it here made three too-long requests indistinguishable from
        # three decode crashes in every archived serving summary.
        failed   = sum(
            1 for r in records
            if r.decision is Decision.ACCEPT and r.lane is not None and not r.ok
        )

        # ---- Per-lane dispatched counts ------------------------------------
        cpu_count = sum(1 for r in records if r.lane is LaneOwner.CPU)
        gpu_count = sum(1 for r in records if r.lane is LaneOwner.GPU)

        # ---- Throughput (ok requests only) ---------------------------------
        ok_records = [r for r in records if r.ok]
        total_ok_tokens   = sum(r.decode_tokens for r in ok_records)
        total_ok_svc_time = sum(r.service_time  for r in ok_records)

        if total_ok_svc_time > 0.0:
            aggregate_decode_tps = total_ok_tokens / total_ok_svc_time
        else:
            aggregate_decode_tps = 0.0

        # Wall-clock aggregate serving throughput (the REAL serving number):
        # total ok tokens divided by the wall-clock span from the earliest lane
        # assignment to the latest completion among ok requests.  Concurrent
        # lanes overlap in this window, so two 1 s / 32-tok lanes report 64,
        # not the service-time-normalized 32.  0.0 if no ok requests or the
        # span is non-positive.
        if ok_records:
            wall_start = min(r.t_assign for r in ok_records)
            wall_end   = max(r.t_done   for r in ok_records)
            wall_span  = wall_end - wall_start
            wallclock_aggregate_tps = (
                total_ok_tokens / wall_span if wall_span > 0.0 else 0.0
            )
        else:
            wallclock_aggregate_tps = 0.0

        # ---- Latency percentiles (ok, positive latency only) ---------------
        valid_latencies = [
            r.total_latency
            for r in ok_records
            if r.t_done > r.t_admit
        ]
        if valid_latencies:
            sorted_lats = sorted(valid_latencies)
            n = len(sorted_lats)
            obs_min, obs_max = sorted_lats[0], sorted_lats[-1]
            # p50: median
            p50 = statistics.median(sorted_lats)
            # p95: NEAREST-RANK (never extrapolates beyond the observed data).
            # statistics.quantiles(..., n=100)[94] uses the EXCLUSIVE method,
            # which interpolates and can exceed the observed max for small n
            # (e.g. [1,2,3] -> 3.8 > max 3.0).  Nearest-rank avoids that.
            p95_idx = math.ceil(0.95 * n) - 1
            p95 = sorted_lats[p95_idx]
            # Defensive clamp: percentiles must lie within the observed range.
            p50 = min(max(p50, obs_min), obs_max)
            p95 = min(max(p95, obs_min), obs_max)
        else:
            p50 = 0.0
            p95 = 0.0

        # ---- Two-traffic split (THE KEY SECTION) ---------------------------
        # MEASURED: sum of copy_bytes over ALL requests (not just ok).
        tabm_payload_bytes = sum(r.copy_bytes for r in records)

        # ESTIMATED: model_bytes × decode_tokens over OK requests only.
        # Reuse total_ok_tokens already computed above (same sum, avoids duplication).
        est_weight_stream_bytes = self._model_bytes * total_ok_tokens

        # ---- Prefix-cache affinity (est_; zeros when unused) --------------
        # All three quantities are 0 when no record ever carries a prefix —
        # the fields default to None/False/0, so an unused-affinity summary
        # is all-zero without any special-casing here.
        requests_with_prefix = sum(1 for r in records if r.prefix_id is not None)
        prefix_hits = sum(1 for r in records if r.prefix_hit)
        est_prefill_saved_us_total = sum(r.est_prefill_saved_us for r in records)

        return {
            "counts": {
                "total":    total,
                "accepted": accepted,
                "delayed":  delayed,
                "dropped":  dropped,
                "ok":       ok_count,
                "failed":   failed,
            },
            "per_lane": {
                "CPU": cpu_count,
                "GPU": gpu_count,
            },
            "throughput": {
                "aggregate_decode_tps": aggregate_decode_tps,
                "wallclock_aggregate_tps": wallclock_aggregate_tps,
            },
            "latency": {
                "p50_latency": p50,
                "p95_latency": p95,
            },
            "traffic": {
                # MEASURED — TABM-controlled payload copies
                "tabm_payload_bytes":      tabm_payload_bytes,
                # ESTIMATED — decode-engine LPDDR weight streaming
                "est_weight_stream_bytes": est_weight_stream_bytes,
                # Always True: flag to prevent conflation with measured data
                "weight_stream_is_estimated": True,
                # Explanation of the estimation and LPDDR non-measurability
                "note": _WEIGHT_STREAM_NOTE,
            },
            "affinity": {
                "requests_with_prefix": requests_with_prefix,
                "prefix_hits": prefix_hits,
                "est_prefill_saved_us_total": est_prefill_saved_us_total,
            },
        }

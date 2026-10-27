"""
TABM scheduling glue: turn the Prefix Registry, CacheProviders
and caller-supplied measured savings into the two callables
``LaneDispatcher`` consumes for queue-wait routing.

Scheduling: request carries prefix_id/len -> registry resolves warm
lanes -> warm+idle routes as before -> warm+BUSY: wait iff
saved_prefill_estimate > expected_wait. All numbers keep the est_/measured_
split — every saving returned here carries a SOURCE tag:

  "measured_entry" : PrefixEntry.measured_saving_us (a real cold/warm A/B on
                     that lane fed via record_measured()) — highest precedence.
  "measured_table" : interpolated from a caller-supplied measurement table.
  "est"            : PrefixAffinityConfig fallback (alpha * L * per_token) —
                     same formula as affinity_saving_s.

Layering: dispatch.py stays registry-agnostic (it takes plain callables);
THIS module is the only place registry/providers/table meet the dispatcher.
Pure stdlib; no board access; every function here is deterministic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

from RK3588.edgeflow_rk3588.runtime.tabm.cache_providers import CacheProvider
from RK3588.edgeflow_rk3588.runtime.tabm.prefix_store import PrefixRegistry


@dataclass(frozen=True)
class MeasuredSavingTable:
    """
    Piecewise-linear interpolation over caller-supplied (prefix_len,
    measured_saving_ms) points. Below the smallest measured L, returns 0.0:
    the table has no evidence there, and queue-wait must not invent positive
    savings for tiny prefixes. Above the largest L, returns the largest
    measured saving (conservative: the true curve keeps growing, so clamping
    under-estimates).
    """
    points: tuple  # ((L:int, saving_s:float), ...) sorted by L, len >= 1

    @classmethod
    def from_json(cls, path: str) -> "MeasuredSavingTable":
        """Load a measured sweep JSON file (its `sweep` array)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pts = sorted(
            (int(row["L"]), float(row["measured_saving_ms"]) / 1000.0)
            for row in data["sweep"]
        )
        if not pts:
            raise ValueError(f"MeasuredSavingTable.from_json: empty sweep in {path}")
        ls = [l for l, _ in pts]
        if len(set(ls)) != len(ls):
            # Duplicate L rows would make interpolation divide by zero
            # (l1 - l0 == 0) — fail loud at load, not at lookup time.
            raise ValueError(f"MeasuredSavingTable.from_json: duplicate L in {path}")
        return cls(points=tuple(pts))

    def saving_s(self, prefix_len: int) -> float:
        pts = self.points
        if prefix_len < pts[0][0]:
            return 0.0
        if prefix_len == pts[0][0]:
            return pts[0][1]
        if prefix_len >= pts[-1][0]:
            return pts[-1][1]
        for (l0, s0), (l1, s1) in zip(pts, pts[1:]):
            if l0 <= prefix_len <= l1:
                frac = (prefix_len - l0) / (l1 - l0)
                return s0 + frac * (s1 - s0)
        raise AssertionError("unreachable: pts sorted and bounds checked")


def make_registry_warmth(
    registry: PrefixRegistry,
    providers: dict,
    lane_key: Callable = lambda lane: lane.owner,
) -> Callable:
    """
    Build the ``warmth(lane, prefix_id) -> bool`` callable for
    LaneDispatcher's queue-wait path, backed by REGISTRY VALIDITY + the
    lane's CacheProvider, rather than the lane-LRU heuristic.

    providers maps lane_key(lane) -> CacheProvider. A lane with no provider,
    or an unknown prefix_id, is NEVER warm (conservative).
    """
    def warmth(lane, prefix_id: Optional[str]) -> bool:
        if prefix_id is None:
            return False
        provider: Optional[CacheProvider] = providers.get(lane_key(lane))
        if provider is None:
            return False
        entry = registry.lookup(prefix_id)
        if entry is None:
            return False
        return provider.is_warm(entry)

    return warmth


def make_saving_estimator(
    registry: Optional[PrefixRegistry] = None,
    table: Optional[MeasuredSavingTable] = None,
    est_alpha: float = 1.0,
    est_per_token_s: float = 0.0,
) -> Callable:
    """
    Build ``saving(prefix_id, prefix_len) -> (seconds, source)`` with the
    est_/measured_ precedence contract (module docstring). Supply a calibrated
    ``est_per_token_s`` or measured table for the target workload. Without
    either, the fallback saving is zero; no device calibration is bundled.
    """
    def saving(prefix_id: Optional[str], prefix_len: int):
        if prefix_id is not None and registry is not None:
            entry = registry.lookup(prefix_id)
            if entry is not None and entry.measured_saving_us is not None:
                return entry.measured_saving_us / 1e6, "measured_entry"
        if table is not None and prefix_len > 0:
            return table.saving_s(prefix_len), "measured_table"
        return est_alpha * prefix_len * est_per_token_s, "est"

    return saving

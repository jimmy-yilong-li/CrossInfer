"""
TABM Core: logical prefix registry store
(PrefixValidity state machine, PrefixEntry, JSON persistence, merge, the
PrefixRegistry class itself).

Public imports are available through `...tabm.prefix_registry`.
"""
from __future__ import annotations

import enum
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional

from RK3588.edgeflow_rk3588.runtime.tabm.prefix_identity import (  # noqa: F401
    PrefixCompat,
    PrefixKind,
    _final_prefix_id,
    _provisional_prefix_id,
    _validate_token_identity,
)

# ---------------------------------------------------------------------------
# PrefixValidity + guarded transitions
# ---------------------------------------------------------------------------

class PrefixValidity(enum.Enum):
    UNMATERIALIZED = "unmaterialized"   # Initial state before
                                        # any materialization attempt exists.
    MATERIALIZED   = "materialized"
    REUSABLE       = "reusable"
    EXPIRED        = "expired"


# Legal chain: UNMATERIALIZED -> MATERIALIZED -> REUSABLE
# -> EXPIRED. EXPIRED is terminal (absent from this map => no legal move out).
_LEGAL_VALIDITY_MOVES: dict[PrefixValidity, PrefixValidity] = {
    PrefixValidity.UNMATERIALIZED: PrefixValidity.MATERIALIZED,
    PrefixValidity.MATERIALIZED: PrefixValidity.REUSABLE,
    PrefixValidity.REUSABLE: PrefixValidity.EXPIRED,
}

# Used only by _merge_entries() to pick the "further along" validity when two
# entries collide onto the same final identity (see that function).
_VALIDITY_RANK: dict[PrefixValidity, int] = {
    PrefixValidity.UNMATERIALIZED: 0,
    PrefixValidity.MATERIALIZED: 1,
    PrefixValidity.REUSABLE: 2,
    PrefixValidity.EXPIRED: 3,
}


class PrefixValidityError(RuntimeError):
    """Raised when an illegal PrefixEntry.validity transition is attempted."""


# ---------------------------------------------------------------------------
# PrefixEntry
# ---------------------------------------------------------------------------

@dataclass
class PrefixEntry:
    """
    One row of the logical prefix database.

    est_saving_us / measured_saving_us are kept STRICTLY SEPARATE:
    est_saving_us is a plain, freely-assignable field (a heuristic estimate);
    measured_saving_us can ONLY be set via record_measured() (enforced: it is
    backed by a private field with a read-only property, not a plain
    attribute) after a real cold/warm A/B on a lane. Use best_saving_us()
    to read either — it documents its own measured>est precedence and is the
    ONLY sanctioned way to get "the" saving; never read the two fields and
    fall back between them ad hoc elsewhere.

    physical_handles is RUNTIME-ONLY (dict lane_id -> handle) — never
    persisted by PrefixRegistry.save().
    """
    prefix_id: str
    token_ids_hash: Optional[str]
    token_count: int
    kind: PrefixKind
    compat: PrefixCompat
    validity: PrefixValidity = PrefixValidity.UNMATERIALIZED
    est_saving_us: Optional[float] = None
    hits: int = 0
    last_used: Optional[float] = None
    physical_handles: dict[str, str] = field(default_factory=dict)
    _measured_saving_us: Optional[float] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError(
                f"PrefixEntry: token_count must be >= 0, got {self.token_count}"
            )

    # ------------------------------------------------------------------
    # est_/measured_ split
    # ------------------------------------------------------------------

    @property
    def measured_saving_us(self) -> Optional[float]:
        """Read-only — set exclusively via record_measured()."""
        return self._measured_saving_us

    def record_measured(self, saving_us: float) -> None:
        """
        Record a MEASURED prefill-time saving from an actual cold/warm A/B on
        a lane. This is the ONLY way
        measured_saving_us is ever set.
        """
        self._measured_saving_us = float(saving_us)

    def best_saving_us(self) -> Optional[float]:
        """Precedence: measured_saving_us if set, else est_saving_us, else None."""
        if self._measured_saving_us is not None:
            return self._measured_saving_us
        return self.est_saving_us

    # ------------------------------------------------------------------
    # Validity transitions (guarded)
    # ------------------------------------------------------------------

    def transition(self, target: PrefixValidity) -> None:
        """
        Guarded validity transition. Legal chain: UNMATERIALIZED ->
        MATERIALIZED -> REUSABLE -> EXPIRED. EXPIRED is terminal (no legal
        move out). Any other move — including a same-state no-op, skipping a
        step, or moving backward — raises PrefixValidityError.
        """
        legal_next = _LEGAL_VALIDITY_MOVES.get(self.validity)
        if legal_next is None or target is not legal_next:
            raise PrefixValidityError(
                f"illegal prefix validity transition: {self.validity.value} -> "
                f"{target.value} (prefix_id={self.prefix_id!r})"
            )
        self.validity = target

    # ------------------------------------------------------------------
    # Reuse stats
    # ------------------------------------------------------------------

    def mark_used(self, now: float) -> None:
        """Record one reuse: increments hits and sets last_used=now."""
        self.hits += 1
        self.last_used = now


# ---------------------------------------------------------------------------
# JSON (de)serialization helpers — physical_handles intentionally excluded.
# ---------------------------------------------------------------------------

def _compat_to_dict(compat: PrefixCompat) -> dict[str, str]:
    return {
        "model_id": compat.model_id,
        "tokenizer_hash": compat.tokenizer_hash,
        "chat_template_hash": compat.chat_template_hash,
        "quant_hash": compat.quant_hash,
        "backend_id": compat.backend_id,
    }


def _compat_from_dict(d: dict) -> PrefixCompat:
    return PrefixCompat(
        model_id=d["model_id"],
        tokenizer_hash=d["tokenizer_hash"],
        chat_template_hash=d["chat_template_hash"],
        quant_hash=d["quant_hash"],
        backend_id=d["backend_id"],
    )


def _entry_to_dict(entry: PrefixEntry) -> dict:
    return {
        "prefix_id": entry.prefix_id,
        "token_ids_hash": entry.token_ids_hash,
        "token_count": entry.token_count,
        "kind": entry.kind.value,
        "compat": _compat_to_dict(entry.compat),
        "validity": entry.validity.value,
        "est_saving_us": entry.est_saving_us,
        "measured_saving_us": entry.measured_saving_us,
        "hits": entry.hits,
        "last_used": entry.last_used,
        # physical_handles are runtime-only and intentionally omitted.
        # Cache validity must be revalidated against on-disk sync.k/v files.
    }


def _entry_from_dict(d: dict) -> PrefixEntry:
    entry = PrefixEntry(
        prefix_id=d["prefix_id"],
        token_ids_hash=d["token_ids_hash"],
        token_count=d["token_count"],
        kind=PrefixKind(d["kind"]),
        compat=_compat_from_dict(d["compat"]),
        validity=PrefixValidity(d["validity"]),
        est_saving_us=d.get("est_saving_us"),
        hits=d.get("hits", 0),
        last_used=d.get("last_used"),
    )
    measured = d.get("measured_saving_us")
    if measured is not None:
        entry.record_measured(measured)
    return entry


def _default_revalidate(_entry: PrefixEntry) -> bool:
    """
    Default revalidate() callback for PrefixRegistry.load(): ALWAYS fails
    (conservative-by-default — no real sync.k/v file probing is hardcoded
    here). Callers who want load()
    to actually trust a materialized snapshot must inject a real check.
    """
    return False


def _merge_entries(dst: PrefixEntry, src: PrefixEntry) -> None:
    """
    Merge src into dst in place when set_token_identity() discovers two
    provisional registrations that resolved to the SAME final identity.

    Merge rules:
      - hits are SUMMED (both entries genuinely observed that many reuses).
      - last_used takes the more recent of the two.
      - validity keeps the HIGHER-ranked (further progressed) of the two —
        see _VALIDITY_RANK. This is conservative in both directions: a
        warm (MATERIALIZED/REUSABLE) entry is never silently demoted by
        merging in a cold duplicate, and if either side is EXPIRED the
        merge result is EXPIRED too (never claim warmth the registry isn't
        sure is still honest).
      - physical_handles are unioned; dst's entries win on key collision.
      - est_saving_us / measured_saving_us: dst's value wins if already set,
        else src's value is adopted.
    """
    dst.hits += src.hits
    if src.last_used is not None and (dst.last_used is None or src.last_used > dst.last_used):
        dst.last_used = src.last_used
    if _VALIDITY_RANK[src.validity] > _VALIDITY_RANK[dst.validity]:
        dst.validity = src.validity
    merged_handles = dict(src.physical_handles)
    merged_handles.update(dst.physical_handles)
    dst.physical_handles = merged_handles
    if dst.est_saving_us is None and src.est_saving_us is not None:
        dst.est_saving_us = src.est_saving_us
    if dst.measured_saving_us is None and src.measured_saving_us is not None:
        dst.record_measured(src.measured_saving_us)


# ---------------------------------------------------------------------------
# PrefixRegistry
# ---------------------------------------------------------------------------

class PrefixRegistry:
    """
    Logical prefix database implemented in Python, with no board or engine
    dependency. Supports LRU capacity limits and TTL eviction.

    register() is a get-or-create keyed by (raw_text, compat) — or directly
    by (compat, token_ids_hash) when token_ids_hash is supplied up front. On
    a repeat call for an existing identity, the EXISTING entry is returned
    unchanged; a differing `kind`/`token_count` argument on that repeat call
    is NOT compared or re-applied (design decision: callers must not vary
    `kind` for the same (raw_text, compat) pair).
    """

    def __init__(
        self,
        *,
        lru_capacity: Optional[int] = None,
        ttl_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if lru_capacity is not None and lru_capacity < 1:
            raise ValueError(
                f"PrefixRegistry: lru_capacity must be >= 1, got {lru_capacity}"
            )
        if ttl_s is not None and ttl_s <= 0:
            raise ValueError(f"PrefixRegistry: ttl_s must be > 0, got {ttl_s}")
        self._lru_capacity = lru_capacity
        self._ttl_s = ttl_s
        self._clock = clock
        self._by_id: dict[str, PrefixEntry] = {}
        self._alias: dict[tuple[str, PrefixCompat], str] = {}
        # LRU touch order shared by register()/record_use()/set_token_identity()
        # re-keying — see module docstring "Eviction model".
        self._lru: "OrderedDict[str, None]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, prefix_id: str) -> bool:
        return prefix_id in self._by_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require(self, prefix_id: str) -> PrefixEntry:
        entry = self._by_id.get(prefix_id)
        if entry is None:
            raise KeyError(f"PrefixRegistry: no entry registered for prefix_id={prefix_id!r}")
        return entry

    def _forget_from_by_id_and_aliases(self, prefix_id: str) -> None:
        self._by_id.pop(prefix_id, None)
        stale_aliases = [k for k, v in self._alias.items() if v == prefix_id]
        for k in stale_aliases:
            del self._alias[k]

    def _touch_lru(self, prefix_id: str) -> None:
        if prefix_id in self._lru:
            self._lru.move_to_end(prefix_id)
        else:
            self._lru[prefix_id] = None
        if self._lru_capacity is not None and len(self._lru) > self._lru_capacity:
            evicted_id, _ = self._lru.popitem(last=False)  # evict least-recently-touched
            self._forget_from_by_id_and_aliases(evicted_id)

    def _rekey_lru(self, old_id: str, new_id: str) -> None:
        self._lru.pop(old_id, None)
        self._touch_lru(new_id)

    # ------------------------------------------------------------------
    # Registration + lookup
    # ------------------------------------------------------------------

    def register(
        self,
        raw_text: str,
        compat: PrefixCompat,
        kind: PrefixKind,
        *,
        token_ids_hash: Optional[str] = None,
        token_count: Optional[int] = None,
    ) -> PrefixEntry:
        """
        Get-or-create a PrefixEntry for (raw_text, compat). Idempotent on
        same identity (see class docstring).

        If token_ids_hash is given (the caller already has TOK results), the
        entry is registered directly under its FINAL identity. Otherwise a
        PROVISIONAL identity is used (see module docstring) until
        set_token_identity() finalizes it.

        token_ids_hash/token_count are validated when supplied; see
        _validate_token_identity().
        """
        _validate_token_identity(token_count, token_ids_hash)

        alias_key = (raw_text, compat)
        existing_id = self._alias.get(alias_key)
        if existing_id is not None and existing_id in self._by_id:
            self._touch_lru(existing_id)
            return self._by_id[existing_id]

        if token_ids_hash is not None:
            prefix_id = _final_prefix_id(compat, token_ids_hash)
        else:
            prefix_id = _provisional_prefix_id(compat, raw_text)

        entry = self._by_id.get(prefix_id)
        if entry is None:
            entry = PrefixEntry(
                prefix_id=prefix_id,
                token_ids_hash=token_ids_hash,
                token_count=token_count or 0,
                kind=kind,
                compat=compat,
            )
            self._by_id[prefix_id] = entry

        self._alias[alias_key] = prefix_id
        self._touch_lru(prefix_id)
        return entry

    def lookup(self, prefix_id: str) -> Optional[PrefixEntry]:
        """Lookup by prefix_id (provisional OR final — whatever is current)."""
        return self._by_id.get(prefix_id)

    def lookup_alias(self, raw_text: str, compat: PrefixCompat) -> Optional[PrefixEntry]:
        """Lookup by the (raw_text, compat) alias — see module docstring."""
        prefix_id = self._alias.get((raw_text, compat))
        if prefix_id is None:
            return None
        return self._by_id.get(prefix_id)

    def set_token_identity(
        self,
        raw_text: str,
        compat: PrefixCompat,
        token_ids_hash: str,
        token_count: int,
    ) -> PrefixEntry:
        """
        Finalize identity from the worker's TOK reply.
        Re-keys the provisional alias registered by an earlier
        register(raw_text, compat, kind) call (which MUST have happened
        first — raises KeyError otherwise). Collision on the same final
        identity (another entry already finalized to it) MERGES the two,
        keeping stats — see _merge_entries().

        token_ids_hash/token_count are validated up front;
        see _validate_token_identity(). This runs BEFORE the "no registered
        entry" KeyError below, so a malformed hash/count is reported
        precisely rather than masked by the unrelated KeyError case.
        Unlike register() (whose identity args are OPTIONAL), this method
        finalizes identity: None is rejected outright because runtime type
        hints do not protect this boundary.
        """
        if token_ids_hash is None or token_count is None:
            raise ValueError(
                "PrefixRegistry.set_token_identity: token_count and "
                "token_ids_hash must be concrete (got None) -- identity "
                "finalization cannot be partial"
            )
        _validate_token_identity(token_count, token_ids_hash)

        alias_key = (raw_text, compat)
        provisional_id = self._alias.get(alias_key)
        if provisional_id is None or provisional_id not in self._by_id:
            raise KeyError(
                "PrefixRegistry.set_token_identity: no registered entry for "
                "this (raw_text, compat) -- call register() first"
            )
        entry = self._by_id[provisional_id]
        final_id = _final_prefix_id(compat, token_ids_hash)

        if final_id == provisional_id:
            # Already finalized under this exact id (idempotent re-call).
            entry.token_ids_hash = token_ids_hash
            entry.token_count = token_count
            self._touch_lru(final_id)
            return entry

        existing_final = self._by_id.get(final_id)
        if existing_final is not None and existing_final is not entry:
            _merge_entries(dst=existing_final, src=entry)
            del self._by_id[provisional_id]
            self._alias[alias_key] = final_id
            self._rekey_lru(provisional_id, final_id)
            return existing_final

        # No collision: re-key the provisional entry in place.
        del self._by_id[provisional_id]
        entry.prefix_id = final_id
        entry.token_ids_hash = token_ids_hash
        entry.token_count = token_count
        self._by_id[final_id] = entry
        self._alias[alias_key] = final_id
        self._rekey_lru(provisional_id, final_id)
        return entry

    # ------------------------------------------------------------------
    # Physical handles (RUNTIME-ONLY — see PrefixEntry docstring)
    # ------------------------------------------------------------------

    def attach_handle(self, prefix_id: str, lane_id: str, handle: str) -> None:
        self._require(prefix_id).physical_handles[lane_id] = handle

    def detach_handle(self, prefix_id: str, lane_id: str) -> None:
        self._require(prefix_id).physical_handles.pop(lane_id, None)

    # ------------------------------------------------------------------
    # Reuse stats + eviction
    # ------------------------------------------------------------------

    def record_use(self, prefix_id: str, *, now: Optional[float] = None) -> None:
        """Record a cache hit: bumps hits/last_used and touches the LRU order."""
        entry = self._require(prefix_id)
        entry.mark_used(now if now is not None else self._clock())
        self._touch_lru(prefix_id)

    def evict_stale(self, *, now: Optional[float] = None) -> list[str]:
        """
        TTL eviction (see module docstring "Eviction model"): physically
        remove entries whose last_used age exceeds ttl_s. No-op (returns [])
        if ttl_s is None (disabled). Never-used entries (last_used is None)
        are immune. Returns the evicted prefix_ids.
        """
        if self._ttl_s is None:
            return []
        t = now if now is not None else self._clock()
        stale_ids = [
            pid for pid, entry in self._by_id.items()
            if entry.last_used is not None and (t - entry.last_used) > self._ttl_s
        ]
        for pid in stale_ids:
            self._lru.pop(pid, None)
            self._forget_from_by_id_and_aliases(pid)
        return stale_ids

    # ------------------------------------------------------------------
    # Snapshot persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        JSON snapshot. physical_handles are RUNTIME-ONLY and are NEVER
        written.
        """
        data = {
            "version": 1,
            "entries": [_entry_to_dict(e) for e in self._by_id.values()],
            "aliases": [
                {"raw_text": rt, "compat": _compat_to_dict(compat), "prefix_id": pid}
                for (rt, compat), pid in self._alias.items()
            ],
            "lru_order": list(self._lru.keys()),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        revalidate: Optional[Callable[[PrefixEntry], bool]] = None,
        lru_capacity: Optional[int] = None,
        ttl_s: Optional[float] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> "PrefixRegistry":
        """
        Load a JSON snapshot written by save(). physical_handles are always
        empty on load (never persisted). Every entry whose PERSISTED
        validity is MATERIALIZED or REUSABLE is passed to `revalidate(entry)`
        (default: _default_revalidate — ALWAYS False, i.e. no real sync.k/v
        file probing is hardcoded here; see module docstring). If revalidate
        returns False, the entry is demoted to UNMATERIALIZED (this demotion
        deliberately bypasses PrefixEntry.transition()'s guard — it models
        restoring a snapshot, not a live transition) and its (already-empty)
        physical_handles are left empty. UNMATERIALIZED/EXPIRED entries are
        loaded as-is (nothing to revalidate).

        LRU/capacity: every loaded entry is placed under normal LRU control
        (entries missing from the snapshot's lru_order count as
        least-recent), and ``lru_capacity`` is enforced during the load —
        an over-capacity snapshot keeps only its most-recent entries, the
        same outcome live registration would have produced.
        """
        if revalidate is None:
            revalidate = _default_revalidate
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ctor_kwargs: dict = {}
        if lru_capacity is not None:
            ctor_kwargs["lru_capacity"] = lru_capacity
        if ttl_s is not None:
            ctor_kwargs["ttl_s"] = ttl_s
        if clock is not None:
            ctor_kwargs["clock"] = clock
        registry = cls(**ctor_kwargs)

        for entry_dict in data.get("entries", []):
            entry = _entry_from_dict(entry_dict)
            if entry.validity in (PrefixValidity.MATERIALIZED, PrefixValidity.REUSABLE):
                if not revalidate(entry):
                    entry.validity = PrefixValidity.UNMATERIALIZED
                    entry.physical_handles = {}
            registry._by_id[entry.prefix_id] = entry

        for alias_dict in data.get("aliases", []):
            compat = _compat_from_dict(alias_dict["compat"])
            registry._alias[(alias_dict["raw_text"], compat)] = alias_dict["prefix_id"]

        # Rebuild the LRU through the SAME machinery live code uses
        # (_touch_lru), so lru_capacity is enforced at load time and EVERY
        # loaded entry ends up under LRU control.  Entries absent from the
        # snapshot's lru_order (hand-edited file, older writer) are touched
        # FIRST — treated as least-recent — then the recorded order is
        # replayed oldest -> newest so recorded recency wins.  An
        # over-capacity snapshot is trimmed here exactly as live
        # registration would have trimmed it; the direct-write alternative
        # left such entries in _by_id but outside _lru, unreachable by
        # eviction forever.  (Aliases are loaded above so _touch_lru's
        # eviction path also drops the evicted entries' aliases.)
        recorded = [pid for pid in data.get("lru_order", []) if pid in registry._by_id]
        recorded_set = set(recorded)
        for pid in list(registry._by_id.keys()):
            if pid not in recorded_set:
                registry._touch_lru(pid)
        for pid in recorded:
            registry._touch_lru(pid)

        return registry

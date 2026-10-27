"""
TABM Core: Admission layer (TokenFlowPolicy).

Decides ACCEPT / DELAY / DROP for each incoming request and, on ACCEPT,
acquires a slot from SlotPool and stamps it with the caller-
supplied admission timestamp.

BANDWIDTH BUDGET NOTE
---------------------
``max_inflight`` is a CONCURRENCY CAP, not a measured-bandwidth gate.
This policy does not read memory-bandwidth counters. Bounding the number
of concurrent decode lanes is a proxy for limiting LPDDR weight-stream contention:
fewer simultaneous decoders → fewer streams competing for memory
bandwidth.  It is an approximation; treat it as such.

Purity contract
---------------
``decide()`` is PURE: given the same ``RequestCost`` and
``AdmissionState`` it returns the same ``(Decision, reason)`` and
performs no mutation and no I/O.  All side effects (slot acquisition,
timestamp injection) live exclusively in ``admit()``.
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from RK3588.edgeflow_rk3588.runtime.tabm.slot import Slot, SlotPool


def _valid_count(value: object, minimum: int = 0) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum


def _finite_number(value: object) -> bool:
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def count_raw_prompt_tokens(
    text: str,
    *,
    tokenize_prefix: Callable[[str], tuple[int, str]] | None = None,
    explicit_count: int | None = None,
    image_placeholder: bool = False,
) -> int:
    """Count raw text with the matching tokenizer, never a whitespace heuristic.

    ``TOK`` does not apply a chat template. For a single image, count the
    pre/post segments separately, excluding the placeholder and visual cells.
    This is NOT the native worker's templated/full-prefill length. The caller
    must keep the tokenizer backend idle. Without it, an explicit count from
    the matching tokenizer (with these same segmentation rules) is required.
    """
    if not isinstance(text, str):
        raise ValueError("prompt text must be a string")
    if image_placeholder and text.count("<image>") != 1:
        raise ValueError("prompt must contain exactly one <image> placeholder")
    if explicit_count is not None:
        if not _valid_count(explicit_count):
            raise ValueError("explicit token count must be a nonnegative integer")
        if tokenize_prefix is not None:
            raise ValueError("supply either a tokenizer or an explicit token count, not both")
        return int(explicit_count)
    if not callable(tokenize_prefix):
        raise ValueError("matching tokenizer unavailable; supply an explicit raw token count")
    total = 0
    for segment in text.split("<image>") if image_placeholder else (text,):
        count, _ = tokenize_prefix(segment)
        if not _valid_count(count):
            raise ValueError("tokenizer token count must be a nonnegative integer")
        total += int(count)
    return total


# ---------------------------------------------------------------------------
# Public enums and dataclasses
# ---------------------------------------------------------------------------

class Decision(enum.Enum):
    ACCEPT = "ACCEPT"
    DELAY  = "DELAY"
    DROP   = "DROP"


@dataclass
class RequestCost:
    """
    Per-request resource description passed to the admission policy.

    ``prompt_len`` counts raw text tokens from the matching model tokenizer,
    not words. For VLM requests it is the sum of the separately tokenized
    raw pre/post-image segments; ``n_vis_tokens`` counts visual cells
    separately. Embeds-only requests have ``prompt_len=0``; text-only
    requests have ``n_vis_tokens=0``. Chat-template overhead is not included:
    this raw-text cap is NOT a total context-window or prefill-length gate.

    prefix_id / prefix_len (prefix-cache affinity)
    -------------------------------------------------------------------
    These two fields let a shareable-prompt-prefix identity ride along with
    the request descriptor so the LANE-DISPATCH layer (dispatch.py) can
    route towards a lane that may already have this prefix warm in the
    decode engine's cache.  ``TokenFlowPolicy.decide()`` / ``admit()``
    do not use them for affinity in v1; only count consistency is validated.
    Prefix affinity is a downstream lane-selection concern.

    prefix_id  : caller-assigned identity for the request's shareable
                 prompt prefix (e.g. a hash of a shared system prompt).
                 None = no known/shareable prefix (the common case today).
    prefix_len : number of prompt tokens covered by ``prefix_id`` (a subset
                 of ``prompt_len``).  0 when ``prefix_id`` is None.
    """
    n_vis_tokens:      int
    hidden:            int
    prompt_len:        int
    max_decode_tokens: int
    deadline:          Optional[float] = None  # epoch seconds; None = no deadline
    prefix_id:         Optional[str] = None
    prefix_len:        int = 0


@dataclass
class AdmissionState:
    """
    Snapshot of system state at the moment of an admission decision.

    Constructed by the caller (lane layer / scheduler); passed into
    ``TokenFlowPolicy.decide()`` so that decide() itself remains pure.

    free_slots : slots currently in SlotState.FREE in the pool.
    inflight   : number of decode lanes currently active (owned by
                 the lane-dispatch layer).
    now        : caller-supplied epoch time (allows deterministic tests).
    """
    free_slots: int
    inflight:   int
    now:        float


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class TokenFlowPolicy:
    """
    Admission policy for TABM decode lanes.

    Parameters
    ----------
    max_inflight : int
        Maximum number of concurrent decode lanes.  This is a concurrency
        cap used as a proxy for LPDDR bandwidth contention (see module
        docstring).  It is NOT a measured-bandwidth gate.
    max_prompt_len : int
        Hard cap on raw text tokens, excluding chat-template overhead and
        visual cells. This is not a full-context capacity guarantee.
    max_decode_tokens : int
        Hard cap on the max_decode_tokens field of a request.
    max_vis_tokens : int
        Hard cap on the number of vision tokens (must fit in slot embedding
        buffer).
    expected_hidden : int
        The model/pool hidden size.  A request whose ``hidden`` differs cannot
        fit this pool's fixed slot geometry and is unconditionally dropped.
    """

    def __init__(
        self,
        max_inflight:      int,
        max_prompt_len:    int,
        max_decode_tokens: int,
        max_vis_tokens:    int,
        expected_hidden:   int,
    ) -> None:
        for name, value, minimum in (
            ("max_inflight", max_inflight, 1),
            ("max_prompt_len", max_prompt_len, 0),
            ("max_decode_tokens", max_decode_tokens, 1),
            ("max_vis_tokens", max_vis_tokens, 0),
            ("expected_hidden", expected_hidden, 1),
        ):
            if not _valid_count(value, minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self._max_inflight      = max_inflight
        self._max_prompt_len    = max_prompt_len
        self._max_decode_tokens = max_decode_tokens
        self._max_vis_tokens    = max_vis_tokens
        self._expected_hidden   = expected_hidden

    # ------------------------------------------------------------------
    # Pure decision function
    # ------------------------------------------------------------------

    def decide(
        self, cost: RequestCost, state: AdmissionState
    ) -> tuple[Decision, str]:
        """
        Return ``(Decision, human-readable reason)``.

        PURE: no mutation, no I/O.  Call as many times as needed with the
        same arguments; the result is always identical and the inputs are
        never modified.

        Decision order (first match wins)
        ----------------------------------
        Invalid scheduler snapshots raise ValueError without side effects.
        1. DROP  — invalid cost, any cap exceeded, or deadline expired.
        2. DELAY — transient capacity: no free slot or inflight budget full.
        3. ACCEPT.
        """
        for name in ("free_slots", "inflight"):
            if not _valid_count(getattr(state, name)):
                raise ValueError(f"{name} must be a nonnegative integer")
        if not _finite_number(state.now):
            raise ValueError("now must be a finite number")
        for name, minimum in (
            ("n_vis_tokens", 0), ("hidden", 1), ("prompt_len", 0),
            ("max_decode_tokens", 1), ("prefix_len", 0),
        ):
            if not _valid_count(getattr(cost, name), minimum):
                return Decision.DROP, f"{name} must be an integer >= {minimum}"
        if cost.prefix_len > cost.prompt_len or (cost.prefix_id is None and cost.prefix_len != 0):
            return Decision.DROP, "prefix_len must be within prompt_len and zero without prefix_id"
        if cost.deadline is not None and not _finite_number(cost.deadline):
            return Decision.DROP, "deadline must be a finite number or None"

        # ---- DROP (hard infeasibility) ---------------------------------
        if cost.prompt_len > self._max_prompt_len:
            return (
                Decision.DROP,
                f"prompt_len {cost.prompt_len} exceeds max_prompt_len {self._max_prompt_len}",
            )
        if cost.max_decode_tokens > self._max_decode_tokens:
            return (
                Decision.DROP,
                f"max_decode_tokens {cost.max_decode_tokens} exceeds cap {self._max_decode_tokens}",
            )
        if cost.n_vis_tokens > self._max_vis_tokens:
            return (
                Decision.DROP,
                f"n_vis_tokens {cost.n_vis_tokens} exceeds max_vis_tokens {self._max_vis_tokens}",
            )
        if cost.hidden != self._expected_hidden:
            return (
                Decision.DROP,
                f"hidden {cost.hidden} != model hidden {self._expected_hidden}",
            )
        if cost.deadline is not None and cost.deadline <= state.now:
            return (
                Decision.DROP,
                f"deadline {cost.deadline} already expired (now={state.now})",
            )

        # ---- DELAY (transient capacity) --------------------------------
        if state.free_slots <= 0:
            return (
                Decision.DELAY,
                "no free slots available (pool exhausted)",
            )
        if state.inflight >= self._max_inflight:
            return (
                Decision.DELAY,
                f"inflight {state.inflight} >= max_inflight {self._max_inflight} "
                f"(bandwidth budget cap reached)",
            )

        # ---- ACCEPT ---------------------------------------------------
        return (Decision.ACCEPT, "within all caps and capacity available")

    # ------------------------------------------------------------------
    # Admit: applies the decision (side effects here only)
    # ------------------------------------------------------------------

    def admit(
        self,
        pool: "SlotPool",
        cost: RequestCost,
        request_id: int,
        state: AdmissionState,
    ) -> "tuple[Decision, Optional[Slot]]":
        """
        Apply the admission decision against the slot pool.

        On ACCEPT:
          - Acquires a slot from ``pool``.
          - Overwrites ``slot.t_admit = state.now`` (caller-controlled
            timestamp, not a fresh wall-clock reading).
          - Converts the deadline's budget relative to ``state.now`` once
            to a host monotonic deadline, including subsequent queue wait.
          - Returns ``(ACCEPT, slot)``.

        On DELAY or DROP:
          - No slot is acquired; pool is not mutated.
          - Returns ``(decision, None)``.

        Stale-snapshot edge:
          ``AdmissionState`` is a SNAPSHOT taken before this call.  TABM is
          single-scheduler-thread (see SlotPool's threading contract), so the
          race is NOT another thread — it is a stale snapshot: ``state.free_slots``
          said a slot was available, but an earlier acquire in the SAME scheduler
          loop has since taken the last one.  If decide() returns ACCEPT yet
          ``pool.acquire()`` returns None, the call returns ``(DELAY, None)``
          rather than raising.
        """
        decision, reason = self.decide(cost, state)

        if decision is not Decision.ACCEPT:
            return (decision, None)

        deadline_monotonic = (
            None if cost.deadline is None
            else time.monotonic() + (cost.deadline - state.now)
        )
        # Side effect: acquire slot
        slot = pool.acquire(request_id)
        if slot is None:
            # Race: pool was full despite AdmissionState saying free_slots > 0.
            return (Decision.DELAY, None)

        # Stamp with caller-supplied timestamp for deterministic accounting.
        slot.t_admit = state.now
        slot.deadline_monotonic = deadline_monotonic
        return (Decision.ACCEPT, slot)

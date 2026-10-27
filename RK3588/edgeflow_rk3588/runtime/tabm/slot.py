"""
TABM Core: Slot ABI + preallocated slot pool.

ABI version: TABM_SLOT_ABI_V1

Lifecycle state machine (enforced; illegal transitions raise SlotTransitionError):
    FREE → VISION_WRITING → READY → OWNED → DONE → FREE

release() additionally accepts VISION_WRITING, READY, and OWNED to support
admission-DROP and failure paths (any in-flight slot can be forcibly freed).
release() from FREE raises SlotTransitionError.

NO KV region is stored in the slot — KV is the decoder's private state
(MNN is black-box). Slots hold: metadata header + embedding region +
prompt/token region only.
"""

from __future__ import annotations

import enum
import time
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# ABI version sentinel
# ---------------------------------------------------------------------------

TABM_SLOT_ABI_VERSION = "TABM_SLOT_ABI_V1"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SlotState(enum.Enum):
    FREE            = "FREE"
    VISION_WRITING  = "VISION_WRITING"
    READY           = "READY"
    OWNED           = "OWNED"
    DONE            = "DONE"


class LaneOwner(enum.Enum):
    NONE = "NONE"
    CPU  = "CPU"
    GPU  = "GPU"


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class SlotTransitionError(RuntimeError):
    """Raised when an illegal slot state transition is attempted."""


# ---------------------------------------------------------------------------
# Slot
# ---------------------------------------------------------------------------

class Slot:
    """
    A single TABM slot (TABM_SLOT_ABI_V1).

    The embedding buffer is preallocated once at construction and NEVER
    reallocated.  write_embedding() copies data into the front of this fixed
    buffer and updates accounting fields.

    Embedding-region ABI (TABM_SLOT_ABI_V1, Tier-1)
    -----------------------------------------------
    The embedding region is a ``float32`` numpy buffer of shape
    ``[max_vis_tokens, hidden]``. Explicit 64-byte alignment is not guaranteed.
    The default pool is in-process and copies embeddings via
    ``write_embedding``; it does not provide zero-copy device mapping.

    Do not construct directly — use SlotPool.acquire().
    """

    def __init__(
        self,
        max_vis_tokens: int,
        hidden: int,
        buffer: Optional[np.ndarray] = None,
    ) -> None:
        # Preallocated, reused embedding buffer — allocated exactly once.
        # If `buffer` is provided it must be a numpy array with shape
        # (max_vis_tokens, hidden) and dtype float32; it is used as-is (e.g. a
        # view into a shared-memory slab injected by ShmSlotPool).
        if buffer is not None:
            if buffer.shape != (max_vis_tokens, hidden) or buffer.dtype != np.float32:
                raise ValueError(
                    f"Slot: provided buffer must have shape "
                    f"({max_vis_tokens}, {hidden}) and dtype float32, "
                    f"got shape={buffer.shape} dtype={buffer.dtype}"
                )
            self._buffer: np.ndarray = buffer
        else:
            self._buffer = np.zeros((max_vis_tokens, hidden), dtype=np.float32)
        self._max_vis_tokens: int = max_vis_tokens
        self._hidden_dim: int = hidden   # internal alias; ABI field is self.hidden

        # ---- ABI metadata header (TABM_SLOT_ABI_V1) ----
        self.request_id: int        = -1
        self.state: SlotState       = SlotState.FREE
        self.lane_owner: LaneOwner  = LaneOwner.NONE

        self.n_vis_tokens: int      = 0
        # hidden is the model's hidden size — fixed pool geometry, constant for
        # every request handled by this slot.  It is intentionally NOT zeroed on
        # release(); only n_vis_tokens is per-request and gets reset.
        self.hidden: int            = hidden

        self.prompt_len: int        = 0
        self.max_decode_tokens: int = 0

        self.embed_bytes: int       = 0        # 0 when FREE

        self.t_admit: float         = 0.0
        self.t_ready: float         = 0.0
        self.t_done: float          = 0.0

        # Host-only request budget; never part of the shared-memory descriptor.
        self.deadline_monotonic: Optional[float] = None

        # Copy-chain fields (fail-closed accounting)
        self.copy_count: int        = 0
        self.copy_bytes: int        = 0

        # Incremented on each re-acquisition; KEPT across reset_header()
        self.reuse_count: int       = 0

        # ---- Prompt / token region ----
        self.prompt_ids: list[int]  = []
        self.output_ids: list[int]  = []
        # Text prompt for decoder backends; empty by default.
        self.prompt_text: str       = ""

    # ------------------------------------------------------------------
    # Embedding region
    # ------------------------------------------------------------------

    @property
    def embedding(self) -> np.ndarray:
        """The preallocated embedding buffer.  Never reallocated."""
        if self._buffer is None:
            raise RuntimeError("slot buffer released — pool closed")
        return self._buffer

    def write_embedding(self, arr: np.ndarray) -> None:
        """
        Copy arr (shape [n, hidden], n <= max_vis_tokens) into the front of the
        slot's preallocated buffer.

        Sets n_vis_tokens and embed_bytes.
        Bumps copy_count by 1 and copy_bytes by n * hidden * 4 (float32).

        Raises RuntimeError if the backing buffer has been released (pool closed).
        Raises ValueError if n > max_vis_tokens.
        """
        if self._buffer is None:
            raise RuntimeError("slot buffer released — pool closed")
        if arr.ndim != 2 or arr.shape[1] != self._hidden_dim:
            raise ValueError(
                f"write_embedding: expected shape (n, {self._hidden_dim}), "
                f"got {arr.shape!r}"
            )
        n = arr.shape[0]
        if n > self._max_vis_tokens:
            raise ValueError(
                f"write_embedding: n={n} exceeds max_vis_tokens={self._max_vis_tokens}"
            )
        self._buffer[:n] = arr
        self.n_vis_tokens = n
        self.embed_bytes  = n * self._hidden_dim * 4
        self.copy_count  += 1
        self.copy_bytes  += n * self._hidden_dim * 4

    # ------------------------------------------------------------------
    # Internal reset (called by SlotPool.release)
    # ------------------------------------------------------------------

    def _reset_header(self) -> None:
        """
        Reset all ABI header fields to FREE-state defaults.

        KEEPS:  reuse_count, _buffer (preallocated — never reallocated)
        RESETS: everything else
        """
        self.request_id        = -1
        self.state             = SlotState.FREE
        self.lane_owner        = LaneOwner.NONE
        self.n_vis_tokens      = 0
        # self.hidden is fixed (pool geometry); not reset
        self.prompt_len        = 0
        self.max_decode_tokens = 0
        self.embed_bytes       = 0
        self.t_admit           = 0.0
        self.t_ready           = 0.0
        self.t_done            = 0.0
        self.deadline_monotonic = None
        self.copy_count        = 0
        self.copy_bytes        = 0
        self.prompt_ids        = []
        self.output_ids        = []
        self.prompt_text       = ""
        # Harness-set AD-HOC attributes (read via getattr by dispatch: the
        # shm/vlm descriptor and prefix fields) must not survive
        # slot reuse — a recycled slot leaking a previous request's prefix_id
        # mis-routes the next request.
        for _adhoc in ("prefix_id", "prefix_len", "shm_descriptor",
                       "prefix_name", "prefix_text", "suffix_text"):
            self.__dict__.pop(_adhoc, None)
        # reuse_count intentionally NOT reset


# ---------------------------------------------------------------------------
# SlotPool
# ---------------------------------------------------------------------------

# States from which release() is permitted
_RELEASE_ALLOWED: frozenset[SlotState] = frozenset({
    SlotState.VISION_WRITING,
    SlotState.READY,
    SlotState.OWNED,
    SlotState.DONE,
})


class SlotPool:
    """
    Preallocated pool of TABM slots with an enforced lifecycle state machine.

    All capacity slots are allocated at construction; no slots are ever created
    or destroyed after that.  The embedding buffer inside each slot is also
    allocated once and reused across requests.

    Threading contract (TABM Tier-1 design)
    ---------------------------------------
    SlotPool is owned by a SINGLE scheduler thread (the one calling
    admit/dispatch/reap); it is NOT thread-safe by design.  Parallelism in
    TABM comes from worker PROCESSES, not threads, so there is deliberately no
    internal lock.  Do not share a SlotPool across threads without external
    locking.

    API:
        acquire(request_id) -> Slot | None
        mark_ready(slot)
        assign(slot, lane)
        complete(slot)
        release(slot)
        in_use_count() -> int
        free_count()   -> int
    """

    def __init__(
        self,
        capacity: int,
        max_vis_tokens: int,
        hidden: int,
        buffer_factory=None,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"SlotPool: capacity must be >= 1, got {capacity}")
        self._capacity: int = capacity
        self._max_vis_tokens: int = max_vis_tokens
        self._hidden: int = hidden
        # Build slots, optionally injecting an external buffer from buffer_factory.
        # buffer_factory, if provided, is a callable (int) -> np.ndarray that returns
        # the backing buffer for slot i (e.g. a view into a shared-memory slab).
        # Default (None) allocates a fresh np.zeros buffer inside Slot.__init__.
        self._slots: list[Slot] = []
        for i in range(capacity):
            buf = buffer_factory(i) if buffer_factory is not None else None
            slot = Slot(max_vis_tokens, hidden, buffer=buf)
            # Stamp with fixed pool index for O(1) release lookup.
            slot._pool_idx = i
            self._slots.append(slot)
        # Indices of slots currently in SlotState.FREE
        self._free_indices: set[int] = set(range(capacity))

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def max_vis_tokens(self) -> int:
        return self._max_vis_tokens

    @property
    def hidden(self) -> int:
        return self._hidden

    def in_use_count(self) -> int:
        """Number of slots NOT in the FREE state."""
        return self._capacity - len(self._free_indices)

    def free_count(self) -> int:
        """Number of slots in the FREE state."""
        return len(self._free_indices)

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def _require_member(self, slot: Slot) -> None:
        index = getattr(slot, "_pool_idx", None)
        if (
            type(index) is not int
            or not 0 <= index < self._capacity
            or self._slots[index] is not slot
        ):
            raise SlotTransitionError("slot does not belong to this pool")

    def acquire(self, request_id: int) -> Optional[Slot]:
        """
        Take a FREE slot and transition it to VISION_WRITING.

        Returns None if the pool is full — full is a normal condition; the
        admission layer handles back-pressure.  Bumps reuse_count on the slot.
        """
        if not self._free_indices:
            return None
        idx = next(iter(self._free_indices))
        self._free_indices.discard(idx)
        slot = self._slots[idx]
        slot.request_id = request_id
        slot.state      = SlotState.VISION_WRITING
        slot.t_admit    = time.time()
        slot.reuse_count += 1
        return slot

    def mark_ready(self, slot: Slot) -> None:
        """
        Transition slot VISION_WRITING → READY.  Sets t_ready.

        Raises SlotTransitionError if slot is not in VISION_WRITING.
        """
        self._require_member(slot)
        if slot.state is not SlotState.VISION_WRITING:
            raise SlotTransitionError(
                f"mark_ready requires VISION_WRITING; slot is {slot.state.value}"
            )
        slot.state   = SlotState.READY
        slot.t_ready = time.time()

    def assign(self, slot: Slot, lane: LaneOwner) -> None:
        """
        Transition slot READY → OWNED, setting lane_owner.

        lane must be LaneOwner.CPU or LaneOwner.GPU.
        Raises SlotTransitionError if slot is not READY.
        Raises ValueError if lane is NONE.
        """
        self._require_member(slot)
        if slot.state is not SlotState.READY:
            raise SlotTransitionError(
                f"assign requires READY; slot is {slot.state.value}"
            )
        if lane not in (LaneOwner.CPU, LaneOwner.GPU):
            raise ValueError(
                f"assign: lane must be CPU or GPU, got {lane!r}"
            )
        slot.state      = SlotState.OWNED
        slot.lane_owner = lane

    def complete(self, slot: Slot) -> None:
        """
        Transition slot OWNED → DONE.  Sets t_done.

        Raises SlotTransitionError if slot is not OWNED.
        """
        self._require_member(slot)
        if slot.state is not SlotState.OWNED:
            raise SlotTransitionError(
                f"complete requires OWNED; slot is {slot.state.value}"
            )
        slot.state  = SlotState.DONE
        slot.t_done = time.time()

    def release(self, slot: Slot) -> None:
        """
        Return slot to FREE, resetting the ABI header (except reuse_count).

        Accepted source states: DONE (happy path), VISION_WRITING, READY,
        OWNED (admission-DROP or failure paths).

        Raises SlotTransitionError if slot is already FREE or in an unknown
        state — double-release is a programming error.
        """
        self._require_member(slot)
        if slot.state not in _RELEASE_ALLOWED:
            raise SlotTransitionError(
                f"release not allowed from {slot.state.value}; "
                "accepted: VISION_WRITING, READY, OWNED, DONE"
            )
        slot._reset_header()           # state -> FREE, header defaults
        self._free_indices.add(slot._pool_idx)

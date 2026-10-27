"""
TABM Core — Chunk B (Tier-1.5): Shared-memory slot ABI.

Claim discipline (read before touching):
    - This is a shared-memory SINGLE-COPY handoff; bounded preallocated footprint.
    - It is NOT zero-copy.  The producer copies the (RKNN host) embedding into the
      shared-memory slab exactly once via write_embedding(); the consumer then reads
      it with no further copy.  The producer→consumer handoff is therefore single-copy.
    - The word "zero-copy" is reserved for the C layer (external device buffer the
      consumer reads directly without any copy at all), which is not implemented here.
    - The retained MNN consumer exists. Its opt-in borrowed input reads this host
      storage directly after the initial prefix; prefix concatenation and any
      backend-internal upload are separate copies, not eliminated by SHM attachment.

API summary:
    SlotBufferDescriptor  — picklable descriptor identifying a slot's shm region.
    ShmSlotPool           — SlotPool subclass backed by a single SharedMemory slab.
    SharedFrameLease     — immutable published frame storage held by all snapshots.
    attach_embedding()    — consumer-side attach; returns a numpy VIEW (no re-copy).
"""

from __future__ import annotations

import dataclasses
import os
import sys
import weakref
import zlib
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np

from RK3588.edgeflow_rk3588.runtime.tabm.slot import Slot, SlotPool

# ---------------------------------------------------------------------------
# SlotBufferDescriptor
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SlotBufferDescriptor:
    """
    Picklable descriptor for a single slot's region within a shared-memory slab.

    Can be sent to a worker process so it can attach the slab and read the
    embedding without Python-object serialization or a private copy.

    Fields
    ------
    buffer_kind : str
        "shm" for this layer.  Future layers will add "numpy" | "opencl_host_alloc"
        | "dmabuf".
    shm_name : str
        The name of the SharedMemory segment (passed to SharedMemory(name=...)).
    offset : int
        Byte offset of this slot's region within the slab.
    nbytes : int
        Byte length of the current valid embedding payload.
    shape : tuple
        (n_vis_tokens, hidden) for the current request.
    dtype : str
        "float32".
    capacity_nbytes : int | None
        Byte capacity of this slot's reusable region.  Older descriptors may
        omit it; consumers then conservatively treat capacity as nbytes.
    owner_pid : int | None
        PID that created the slab.  Used to preserve the owner's single
        resource-tracker registration for same-process attachments.
    resource_tracker_id : str | None
        Stable identity of the producer's resource-tracker pipe.  A spawned
        consumer may share this tracker even though it has a different PID.

    Note: C-layer fields (fd, fence_id) are intentionally absent — leave room,
    do not implement.
    """
    buffer_kind: str    # "shm"
    shm_name: str
    offset: int
    nbytes: int
    shape: tuple
    dtype: str          # "float32"
    capacity_nbytes: Optional[int] = None
    owner_pid: Optional[int] = None
    resource_tracker_id: Optional[str] = None


def _resource_tracker_identity() -> Optional[str]:
    """Identify the active tracker pipe across related Python processes."""
    from multiprocessing import resource_tracker as _rt

    try:
        stat = os.fstat(_rt.getfd())
    except (AttributeError, OSError):
        return None
    return f"{stat.st_dev}:{stat.st_ino}"


# ---------------------------------------------------------------------------
# ShmSlotPool
# ---------------------------------------------------------------------------

class ShmSlotPool(SlotPool):
    """
    SlotPool subclass whose embedding regions are backed by a single preallocated
    POSIX shared-memory slab.

    All capacity slots share one SharedMemory object; each slot owns a contiguous
    float32 subregion of that slab (via a numpy view).  The slab is never
    reallocated after construction.

    Single-copy accounting
    ----------------------
    write_embedding() (inherited from Slot) copies the producer array into the
    slot's shm-backed buffer — copy_count += 1, copy_bytes += n*hidden*4.
    That single copy is the entire producer→slab transfer.  Consumer-side
    attach_embedding() returns a view; it adds no further copy.

    NOT zero-copy: the producer must write into the slab.  Zero-copy is the C
    layer (device-side buffer, not implemented here).

    close() must be called when the pool is no longer needed to unlink the shm
    segment.  Idempotent.
    """

    def __init__(self, capacity: int, max_vis_tokens: int, hidden: int) -> None:
        # (a) Validate ALL dims BEFORE any allocation so bad args get a clear error.
        if capacity < 1:
            raise ValueError(f"ShmSlotPool: capacity must be >= 1, got {capacity}")
        if max_vis_tokens < 1:
            raise ValueError(
                f"ShmSlotPool: max_vis_tokens must be >= 1, got {max_vis_tokens}"
            )
        if hidden < 1:
            raise ValueError(f"ShmSlotPool: hidden must be >= 1, got {hidden}")

        # Store geometry for buffer_descriptor(); needed by the factory closure.
        self._max_vis_tokens = max_vis_tokens
        self._hidden = hidden
        region_bytes = max_vis_tokens * hidden * 4   # float32 = 4 bytes each
        self._region_bytes = region_bytes
        self._mapped_closed = False
        self._unlinked = False
        self._closed = False   # set True by close(); guards acquire/buffer_descriptor

        # (b) Allocate the single shared-memory slab (must precede super() so the
        #     factory closure can reference self._shm).
        total_bytes = capacity * region_bytes
        self._owner_pid = os.getpid()
        self._resource_tracker_id = _resource_tracker_identity()
        self._shm = SharedMemory(create=True, size=total_bytes)

        # (c) Factory: returns a numpy VIEW into slot i's region of the slab.
        #     Offset is in bytes; shape is (max_vis_tokens, hidden), dtype float32.
        def _view_fn(i: int) -> np.ndarray:
            return np.ndarray(
                (max_vis_tokens, hidden),
                dtype=np.float32,
                buffer=self._shm.buf,
                offset=i * region_bytes,
            )

        # Let SlotPool.__init__ build _capacity, _slots (with shm-backed buffers),
        # _pool_idx stamps, and _free_indices — no manual bypass needed.
        super().__init__(capacity, max_vis_tokens, hidden, buffer_factory=_view_fn)

    # ------------------------------------------------------------------
    # Lifecycle (override to guard closed pool)
    # ------------------------------------------------------------------

    def acquire(self, request_id: int) -> Optional[Slot]:
        """
        Acquire a FREE slot.  Raises RuntimeError if the pool has been closed.
        See SlotPool.acquire() for full semantics.
        """
        if self._closed:
            raise RuntimeError(
                "ShmSlotPool.acquire: pool is closed — "
                "call acquire only on a live pool"
            )
        return super().acquire(request_id)

    # ------------------------------------------------------------------
    # Descriptor
    # ------------------------------------------------------------------

    def buffer_descriptor(self, slot: Slot) -> SlotBufferDescriptor:
        """
        Return a picklable descriptor for slot's shm region.

        The descriptor carries everything a consumer process needs to call
        attach_embedding() and obtain a view of the embedding without a copy.

        Raises RuntimeError if the pool has been closed.
        Raises ValueError if slot does not belong to this pool.
        """
        if self._closed:
            raise RuntimeError(
                "ShmSlotPool.buffer_descriptor: pool is closed"
            )
        idx = getattr(slot, "_pool_idx", -1)
        if not (0 <= idx < self._capacity) or self._slots[idx] is not slot:
            raise ValueError(
                "buffer_descriptor: slot does not belong to this pool"
            )
        i = slot._pool_idx
        offset = i * self._region_bytes
        if slot.n_vis_tokens <= 0 or slot.embed_bytes <= 0:
            raise ValueError(
                "buffer_descriptor: slot has no active embedding; "
                "call write_embedding() first"
            )
        return SlotBufferDescriptor(
            buffer_kind="shm",
            shm_name=self._shm.name,
            offset=offset,
            nbytes=slot.embed_bytes,
            shape=(slot.n_vis_tokens, self._hidden),
            dtype="float32",
            capacity_nbytes=self._region_bytes,
            owner_pid=self._owner_pid,
            resource_tracker_id=self._resource_tracker_id,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Close and unlink the SharedMemory slab.  Retry-safe and idempotent.

        The two lifecycle phases (munmap and unlink) are tracked independently
        via _mapped_closed and _unlinked.  If unlink raises an unexpected error
        on the first call, _unlinked stays False so a subsequent close()
        re-attempts the unlink rather than silently no-oping (no silent leak).
        The mapping (close/munmap) is never double-closed.

        Only FileNotFoundError from unlink is silenced (slab already unlinked by
        some external means — idempotent).  All other unexpected exceptions
        propagate with _unlinked still False so the caller can retry.

        Use-after-free safety
        ---------------------
        Before unmapping the slab, every slot's _buffer view is set to None so
        no numpy array holds a reference to the freed memory.  After close(),
        the slot API (write_embedding / embedding) raises RuntimeError instead
        of segfaulting.  Externally-held raw buffer references (i.e. a caller
        that retained a direct reference to slot._buffer rather than using the
        slot API) become dangling after close() and must not be used.
        """
        # Mark closed first so acquire() / buffer_descriptor() raise cleanly
        # rather than touching freed state.
        self._closed = True
        if not self._mapped_closed:
            # Drop all slot views BEFORE unmapping the slab.  This ensures no
            # numpy array holds a reference to the memory being freed, so any
            # post-close access via the slot API hits the None guard in Slot
            # (clean RuntimeError) rather than reading freed memory (segfault /
            # rc -11).
            for slot in self._slots:
                slot._buffer = None
            self._shm.close()
            self._mapped_closed = True
        if not self._unlinked:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass            # already unlinked elsewhere — fine
            self._unlinked = True


class SharedFrameLease:
    """One immutable frame retained by every snapshot/consumer that uses it.

    The bounded stream log owns these leases. Native calls keep their snapshot
    alive until completion, so the final Python reference can release the slab
    without a second per-lane staging pool or a new wire protocol.
    """

    def __init__(self, embedding: np.ndarray) -> None:
        pool = ShmSlotPool(1, embedding.shape[0], embedding.shape[1])
        try:
            slot = pool.acquire(0)
            slot.write_embedding(embedding)
            pool.mark_ready(slot)
            self.descriptor = pool.buffer_descriptor(slot)
            self.crc32 = f"{zlib.crc32(memoryview(slot.embedding).cast('B')) & 0xFFFFFFFF:08x}"
        except BaseException:
            pool.close()
            raise
        self._release = weakref.finalize(self, pool.close)


# ---------------------------------------------------------------------------
# Resource-tracker detach helper  (Python < 3.13, consumer side)
# ---------------------------------------------------------------------------

def _detach_from_resource_tracker(shm: SharedMemory) -> None:
    """
    Unregister *shm* from the process resource_tracker so it will NOT call
    shm_unlink when this process exits.  Used on Python < 3.13 where
    SharedMemory() always registers with resource_tracker.

    If unregister fails the handle is closed and RuntimeError is raised —
    returning an unsafe handle that would unlink the producer's slab on
    consumer process exit is refused.  The caller must not use the handle
    after this function raises.
    """
    from multiprocessing import resource_tracker as _rt
    try:
        _rt.unregister(shm._name, "shared_memory")
    except Exception as exc:
        shm.close()
        raise RuntimeError(
            f"attach_embedding: resource_tracker.unregister failed ({exc!r}); "
            "refusing to return handle — consumer would risk unlinking the "
            "producer's slab on process exit. Close the handle and retry."
        ) from exc


# ---------------------------------------------------------------------------
# attach_embedding  (consumer side)
# ---------------------------------------------------------------------------

def attach_embedding(
    desc: SlotBufferDescriptor,
) -> "tuple[np.ndarray, SharedMemory]":
    """
    Consumer-side attach.  May run in a separate process.

    Opens the SharedMemory segment named by desc.shm_name, returns:
        (view, shm_handle)

    ``view`` is a numpy array of shape desc.shape and dtype desc.dtype whose
    backing buffer IS the shared memory — it is a VIEW, not a copy.  Reading it
    incurs no additional copy beyond the single producer→slab copy already done
    by write_embedding().

    The caller is responsible for closing shm_handle when done:
        shm_handle.close()   # do NOT unlink — the producer owns the slab lifetime

    NOT zero-copy: the producer already wrote once into the slab.  This function
    adds zero further copies; the consumer reads the bytes the producer wrote.

    Resource-tracker safety (IMPORTANT)
    ------------------------------------
    A consumer in an INDEPENDENT process (not a multiprocessing spawn-child)
    must NOT let Python's resource_tracker unlink the segment when the consumer
    exits — that would destroy the producer's slab for everyone.

    On Python >= 3.13 we open with ``track=False`` to skip registration entirely.
    On Python <  3.13, tracker registration is process-group scoped and is not
    reference counted. Same-process and multiprocessing-spawn consumers must
    preserve the producer's registration; consumers with an independent tracker
    unregister only their own registration. The shm_handle is still closed
    (munmap) on consumer exit. ShmSlotPool.close() owns the unlink lifecycle.
    The producer must keep the pool alive until dispatched consumers have
    finished attaching and reading; descriptors are not independent leases.
    In particular, attach and producer unlink must not race.
    """
    # ------------------------------------------------------------------
    # Descriptor validation — fail loud with a clear error before any I/O.
    # ------------------------------------------------------------------
    if desc.buffer_kind != "shm":
        raise ValueError(
            f"attach_embedding: unsupported buffer_kind {desc.buffer_kind!r}; "
            "expected 'shm'"
        )
    if desc.offset < 0:
        raise ValueError(
            f"attach_embedding: offset must be >= 0, got {desc.offset}"
        )
    if desc.nbytes <= 0:
        raise ValueError(
            f"attach_embedding: nbytes must be > 0, got {desc.nbytes}"
        )
    dtype = np.dtype(desc.dtype)
    shape = tuple(desc.shape)
    # F4: enforce 2-D shape ABI — descriptor must describe an embedding matrix.
    if len(shape) != 2:
        raise ValueError(
            f"attach_embedding: expected a 2-D shape (rows=n_vis_tokens, "
            f"cols=hidden), got {len(shape)}-D shape {shape!r}"
        )
    if shape[0] < 1 or shape[1] < 1:
        raise ValueError(
            f"attach_embedding: shape dimensions must each be >= 1, "
            f"got {shape!r}"
        )
    n_elements: int = 1
    for dim in shape:
        n_elements *= dim
    expected_nbytes = n_elements * dtype.itemsize
    if expected_nbytes != desc.nbytes:
        raise ValueError(
            f"attach_embedding: shape {shape} × dtype '{desc.dtype}' "
            f"implies {expected_nbytes} bytes, but nbytes={desc.nbytes}"
        )
    capacity_nbytes = (
        desc.nbytes if desc.capacity_nbytes is None else desc.capacity_nbytes
    )
    if capacity_nbytes < desc.nbytes:
        raise ValueError(
            "attach_embedding: capacity_nbytes must be >= nbytes, got "
            f"capacity_nbytes={capacity_nbytes}, nbytes={desc.nbytes}"
        )

    # ------------------------------------------------------------------
    # Open shared memory.
    # Consumer must NOT resource-track a segment it does not own.
    # ------------------------------------------------------------------
    if sys.version_info >= (3, 13):
        # Python 3.13+ exposes track= to skip resource_tracker registration.
        shm = SharedMemory(name=desc.shm_name, create=False, track=False)
    else:
        # Resolve identity before opening a mapping that a failed lookup would
        # otherwise leave without an owner responsible for closing it.
        current_tracker_id = _resource_tracker_identity()
        if desc.resource_tracker_id is not None and current_tracker_id is not None:
            shares_owner_registration = desc.resource_tracker_id == current_tracker_id
        else:
            shares_owner_registration = desc.owner_pid == os.getpid()
        shm = SharedMemory(name=desc.shm_name, create=False)
        if not shares_owner_registration:
            # Independent consumers must remove their own tracker registration
            # so process exit cannot unlink the producer-owned slab. Legacy
            # descriptors carry no identity and retain this conservative path.
            _detach_from_resource_tracker(shm)

    # ------------------------------------------------------------------
    # Best-effort bounds check (catches corrupt descriptors before SIGSEGV).
    # ------------------------------------------------------------------
    if desc.offset + capacity_nbytes > shm.size:
        shm.close()
        raise ValueError(
            f"attach_embedding: offset {desc.offset} + capacity_nbytes "
            f"{capacity_nbytes} = {desc.offset + capacity_nbytes} exceeds "
            f"segment size {shm.size}"
        )

    view = np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=desc.offset)
    return view, shm

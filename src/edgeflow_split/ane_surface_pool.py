"""Bounded ANE surface lease pool for the surface-backed overlap pipeline.

The pool pre-allocates a fixed number of `ANESurfaceLease` objects up
front. Producer code acquires a lease (blocking if the pool is empty),
produces descriptor bytes into it, wraps it in a
`SurfaceBackedVisualDescriptor`, and pushes the descriptor onto a TABM queue.
For Qwen, the producer binds the merger ANE kernel output directly to the
lease. Staged producers may still call `lease.write_from(...)` explicitly.
When the consumer calls
`descriptor.release()` the pool-owned lease routes back to the pool's
free list instead of freeing the underlying IOSurface, so the lease
can be reused for the next frame.

This is the system-level plumbing for overlapped producer/consumer
execution. It owns lease lifecycle and backpressure; producer-specific
data-plane semantics live in the producer helper.

Synchronization and lifecycle:
- Thread safety uses a single `threading.Condition`. `acquire`,
  `_return_to_pool`, and `close` all operate under the condition lock,
  preventing check-then-act races.
- `close()` wakes every thread currently blocked in `acquire()`. The
  woken waiter re-checks the closed flag and raises `RuntimeError`
  instead of blocking indefinitely. This makes shutdown correct even
  when acquirers were sleeping at the time `close()` fired.
- `acquire(timeout=...)` raises `queue.Empty` when the timeout
  expires and `RuntimeError` when the pool is closed (either at call
  time or while the caller was waiting). Both are catchable; callers
  that need cancellable blocking should poll `acquire` with a small
  timeout while checking their own stop flag.
"""

from __future__ import annotations

import queue as _queue
import threading
import time
from typing import Optional

from edgeflow_split.ane_bridge import ANEBridgeError, ANEBridgeLibrary
from edgeflow_split.ane_surface import ANESurfaceLease


class _PooledLease(ANESurfaceLease):
    """An `ANESurfaceLease` that belongs to a `LeasePool`.

    `release()` returns the lease to the pool's free list without
    freeing the underlying IOSurface. The pool is the sole owner of
    the actual `ane_bridge_free_surface_lease` call — it happens only
    in `LeasePool.close()` via `_actually_free`.
    """

    def __init__(
        self,
        *,
        bridge: ANEBridgeLibrary,
        lease_handle: int,
        pool: "LeasePool",
    ) -> None:
        super().__init__(bridge=bridge, lease_handle=lease_handle)
        self._pool: Optional["LeasePool"] = pool

    def release(self) -> None:
        pool: Optional[LeasePool]
        with self._lock:
            if self._released:
                return
            self._released = True
            pool = self._pool
        if pool is not None:
            pool._return_to_pool(self)
        else:
            # Orphaned lease — pool is gone, free the underlying surface.
            self._actually_free()

    def _reactivate(self) -> None:
        """Mark the lease as usable again after being returned to the
        pool. Called by `LeasePool.acquire()` before handing the lease
        back out to a new producer caller.
        """
        with self._lock:
            self._released = False

    def _actually_free(self) -> None:
        """Bypass the pool-return behaviour and free the underlying
        IOSurface. Only `LeasePool.close()` and the orphan path above
        call this.
        """
        with self._lock:
            handle = self.lease_handle
            self.lease_handle = None
            self._released = True
            self._pool = None
        if handle is not None:
            self.bridge.lib.ane_bridge_free_surface_lease(handle)


class LeasePool:
    """A bounded pool of ANE surface leases.

    Pool lifecycle:
    - constructor: pre-allocates `capacity` leases of `bytes_per_lease`
      bytes each. All leases start in the free list.
    - `acquire(timeout=None)`: blocks until a lease is available, the
      pool is closed, or `timeout` expires.
        - returns a `_PooledLease` (reactivated and ready for reuse)
        - raises `queue.Empty` on timeout
        - raises `RuntimeError` if the pool is closed (at call time or
          while the caller was waiting)
    - `_return_to_pool(lease)`: called from `_PooledLease.release()`.
      If the pool is still open, appends the lease to the free list
      and notifies one waiter. If the pool is closed, the lease's
      underlying IOSurface is freed directly.
    - `close()`: marks the pool closed, drains any leases still in
      the free list, and notifies every waiter in `acquire()` so they
      can raise `RuntimeError` and unwind. Any lease that is still
      out at close time frees itself on the next `release()` via
      `_return_to_pool`'s closed-pool branch. Idempotent.

    Thread safety:
    - All state transitions happen under a single `threading.Condition`
      lock. `acquire`, `_return_to_pool`, and `close` each take the
      lock for the full check-and-act sequence, so there is no
      window where a lease can be queued into a pool that has already
      been closed, nor where an acquire waiter can miss a close
      signal.
    - FFI calls (`ane_bridge_free_surface_lease`) happen OUTSIDE the
      condition lock to avoid holding a Python lock across C calls.
    """

    def __init__(
        self,
        *,
        bridge: ANEBridgeLibrary,
        capacity: int,
        bytes_per_lease: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError("LeasePool capacity must be > 0")
        if bytes_per_lease <= 0:
            raise ValueError("LeasePool bytes_per_lease must be > 0")

        self._bridge = bridge
        self._capacity = int(capacity)
        self._bytes_per_lease = int(bytes_per_lease)
        self._free: list[_PooledLease] = []
        self._all_leases: list[_PooledLease] = []
        self._closed = False
        self._cond = threading.Condition()

        if not hasattr(bridge.lib, "ane_bridge_alloc_surface_lease"):
            raise ANEBridgeError(
                "ANE bridge does not export ane_bridge_alloc_surface_lease"
            )

        try:
            for _ in range(self._capacity):
                handle = bridge.lib.ane_bridge_alloc_surface_lease(
                    self._bytes_per_lease
                )
                if not handle:
                    raise ANEBridgeError(
                        f"ane_bridge_alloc_surface_lease({self._bytes_per_lease}) returned NULL"
                    )
                lease = _PooledLease(
                    bridge=bridge,
                    lease_handle=handle,
                    pool=self,
                )
                self._all_leases.append(lease)
                self._free.append(lease)
        except BaseException:
            # Partial allocation — release whatever we got before propagating.
            for lease in self._all_leases:
                lease._actually_free()
            self._all_leases.clear()
            self._free.clear()
            raise

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def bytes_per_lease(self) -> int:
        return self._bytes_per_lease

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def free_count(self) -> int:
        with self._cond:
            return len(self._free)

    def acquire(self, timeout: Optional[float] = None) -> _PooledLease:
        """Block until a lease is available, the pool is closed, or
        the timeout expires.

        Returns a `_PooledLease` on success. Raises `queue.Empty` on
        timeout. Raises `RuntimeError` if the pool is closed at call
        time OR if the pool is closed while the caller is waiting.
        """
        deadline: Optional[float]
        if timeout is None:
            deadline = None
        else:
            deadline = time.monotonic() + float(timeout)

        with self._cond:
            while True:
                if self._closed:
                    raise RuntimeError("cannot acquire from closed LeasePool")
                if self._free:
                    lease = self._free.pop()
                    lease._reactivate()
                    return lease
                # Wait for either a release or a close.
                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _queue.Empty
                    self._cond.wait(timeout=remaining)

    def _return_to_pool(self, lease: _PooledLease) -> None:
        """Return a lease to the pool. Called from `_PooledLease.release()`.

        If the pool is closed, the lease's underlying IOSurface is
        freed directly instead of being appended to the free list.
        """
        # Hold the condition lock for the entire check-and-act so
        # `close()` cannot race a return into an already-drained pool.
        should_free_directly = False
        with self._cond:
            if self._closed:
                should_free_directly = True
            else:
                self._free.append(lease)
                self._cond.notify()
        if should_free_directly:
            lease._actually_free()

    def close(self) -> None:
        """Mark the pool closed and free every lease in the free list.

        Wakes all threads currently blocked in `acquire()`. They see
        the closed flag and raise `RuntimeError` on the next iteration
        of the while loop. Any lease that is still out at close time
        (i.e., held by a descriptor) will free itself on the next
        `release()` via `_return_to_pool`'s closed-pool branch.

        Idempotent. Safe to call while other threads are acquiring,
        returning, or closing concurrently.
        """
        drained: list[_PooledLease] = []
        with self._cond:
            if self._closed:
                return
            self._closed = True
            # Snapshot the free list so we can free outside the lock.
            drained = list(self._free)
            self._free.clear()
            # Wake all waiters so they can re-check the closed flag.
            self._cond.notify_all()
        # FFI calls happen outside the condition lock.
        for lease in drained:
            lease._actually_free()

    def __enter__(self) -> "LeasePool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

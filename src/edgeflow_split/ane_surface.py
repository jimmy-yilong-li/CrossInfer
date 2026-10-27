from __future__ import annotations

import ctypes
import threading

from edgeflow_split.ane_bridge import ANEBridgeError, ANEBridgeLibrary


class ANESurfaceLease:
    def __init__(self, *, bridge: ANEBridgeLibrary, lease_handle) -> None:
        self.bridge = bridge
        self.lease_handle = lease_handle
        self._released = False
        self._lock = threading.Lock()

    @property
    def released(self) -> bool:
        with self._lock:
            return self._released

    def _handle_ptr(self):
        with self._lock:
            if self._released or self.lease_handle is None:
                raise ANEBridgeError("surface lease has been released")
            return self.lease_handle

    @property
    def surface_id(self) -> int:
        return int(self.bridge.lib.ane_bridge_surface_lease_id(self._handle_ptr()))

    @property
    def surface_bytes(self) -> int:
        return int(self.bridge.lib.ane_bridge_surface_lease_size_bytes(self._handle_ptr()))

    def read_into(self, dst) -> None:
        handle = self._handle_ptr()
        view = memoryview(dst)
        if view.readonly:
            raise ValueError("destination buffer must be writable")
        nbytes = int(view.nbytes)
        if nbytes <= 0:
            return
        c_buf = (ctypes.c_ubyte * nbytes).from_buffer(view)
        self.bridge.lib.ane_bridge_read_surface_lease(
            handle,
            ctypes.cast(c_buf, ctypes.c_void_p),
            nbytes,
        )

    def write_from(self, src) -> None:
        """Stage `src` bytes into the leased surface via a CPU-to-IOSurface
        memcpy.

        This remains available for explicit staged descriptor producers such as
        FastVLM/LLaVA fixtures and bridge round-trip tests. The Qwen
        surface-backed producer does not use this method for final embeddings:
        it binds the merger kernel output directly to the lease instead.
        """
        handle = self._handle_ptr()
        view = memoryview(src)
        nbytes = int(view.nbytes)
        if nbytes <= 0:
            return
        if nbytes > self.surface_bytes:
            raise ValueError(
                f"source nbytes {nbytes} exceeds leased surface size {self.surface_bytes}"
            )
        if view.readonly:
            # ctypes.from_buffer requires a writable buffer; copy through a
            # contiguous bytes blob in that case.
            raw = bytes(view)
            buf = (ctypes.c_ubyte * nbytes).from_buffer_copy(raw)
        else:
            buf = (ctypes.c_ubyte * nbytes).from_buffer(view)
        self.bridge.lib.ane_bridge_write_surface_lease(
            handle,
            ctypes.cast(buf, ctypes.c_void_p),
            nbytes,
        )

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            handle = self.lease_handle
            self._released = True
            self.lease_handle = None
        self.bridge.lib.ane_bridge_free_surface_lease(handle)

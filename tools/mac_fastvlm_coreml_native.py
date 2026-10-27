"""Native CoreML output backing on two existing crane IOSurface leases.

The driver must synchronize the llama consumer before reusing index % 2, and
drain its producer worker and consumers before close(). Descriptor pointers are
diagnostic addresses only: consumers must use lease_handle through the bridge's
scoped read-only callback. Production uses the original lease mapping through
ane_bridge_produce_surface_lease, with no separate surface lookup.
"""

import ctypes
from dataclasses import dataclass, field
import operator
from pathlib import Path
import threading

import numpy as np


class _Prediction(ctypes.Structure):
    _fields_ = [
        ("lease_handle", ctypes.c_void_p),
        ("pointer", ctypes.c_void_p),
        ("generation", ctypes.c_uint64),
        ("nbytes", ctypes.c_size_t),
        ("shape", ctypes.c_int32 * 3),
        ("strides", ctypes.c_int32 * 3),
        ("surface_id", ctypes.c_uint32),
        ("slot", ctypes.c_uint32),
        ("object_matches", ctypes.c_int32),
        ("pointer_matches", ctypes.c_int32),
    ]


@dataclass(frozen=True)
class NativeCoreMLFeatures:
    lease_handle: int
    pointer: int
    shape: tuple
    surface_id: int
    receipt: dict
    _owner: object = field(repr=False, compare=False)


class NativeCoreMLVisual:
    def __init__(self, package, library, width=1536):
        width = operator.index(width)
        if width <= 0 or width > (2**31 - 1) // 256:
            raise ValueError("width must be positive and fit the output strides")
        self.width = width
        self._lock = threading.Lock()
        self._context = None
        self.lib = ctypes.CDLL(str(library))
        self.lib.fv_coreml_error.argtypes = []
        self.lib.fv_coreml_error.restype = ctypes.c_char_p
        self.lib.fv_coreml_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
        self.lib.fv_coreml_init.restype = ctypes.c_void_p
        self.lib.fv_coreml_predict.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.c_int32, ctypes.POINTER(_Prediction),
        ]
        self.lib.fv_coreml_predict.restype = ctypes.c_int
        self.lib.fv_coreml_free.argtypes = [ctypes.c_void_p]
        self.lib.fv_coreml_free.restype = None
        self._context = self.lib.fv_coreml_init(str(Path(package).expanduser()).encode(), width)
        if not self._context:
            raise RuntimeError(self._error())

    def _error(self):
        message = self.lib.fv_coreml_error()
        return message.decode(errors="replace") if message else "native CoreML operation failed"

    def predict(self, pixels, index):
        index = operator.index(index)
        if index < 0:
            raise ValueError("index must be non-negative")
        if not isinstance(pixels, np.ndarray) or pixels.dtype != np.float32:
            raise TypeError("pixels must be a float32 NumPy array")
        if pixels.shape != (1, 3, 1024, 1024):
            raise ValueError("pixels shape must be (1, 3, 1024, 1024)")
        if not pixels.flags.c_contiguous or not pixels.flags.aligned:
            raise ValueError("pixels must be aligned and C-contiguous")
        slot = index % 2
        with self._lock:
            if self._context is None:
                raise RuntimeError("native CoreML visual is closed")
            output = _Prediction()
            status = self.lib.fv_coreml_predict(
                self._context, pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                pixels.size, slot, ctypes.byref(output))
            if status != 0:
                raise RuntimeError(self._error())
            shape = tuple(output.shape)
            strides = tuple(output.strides)
            if (not output.lease_handle or not output.pointer or not output.surface_id
                    or output.object_matches != 1 or output.pointer_matches != 1
                    or output.slot != slot or shape != (1, 256, self.width)
                    or strides != (256 * self.width, self.width, 1)
                    or output.nbytes != 256 * self.width * 4):
                raise RuntimeError("native CoreML output backing contract failed")
            receipt = dict(
                data_plane="coreml_output_backing_iosurface", compute_units="CPU_AND_NE",
                slot=slot, generation=int(output.generation), surface_id=int(output.surface_id),
                object_identity=True, pointer_identity=True, dtype="float32",
                shape=list(shape), strides=list(strides), output_bytes=int(output.nbytes),
                producer_pointer=int(output.pointer), pointer_scope="diagnostic_only",
                producer_scope="ane_bridge_produce_surface_lease",
                coreml_internal_copy_status="not_measured")
            return NativeCoreMLFeatures(int(output.lease_handle), int(output.pointer),
                                        shape, int(output.surface_id), receipt, self)

    def close(self):
        """Caller must drain its worker and synchronize consumers before closing."""
        with self._lock:
            if self._context is not None:
                self.lib.fv_coreml_free(self._context)
                self._context = None

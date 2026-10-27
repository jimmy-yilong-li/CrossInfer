import ctypes
import os
from pathlib import Path

import numpy as np

from edgeflow_split.mil_text import mil_main_func_open


def _resolve_default_ane_bridge_path() -> str:
    """Resolve the default ANE bridge dylib path with the following priority:

    1. `EDGEFLOW_ANE_BRIDGE_PATH` environment variable (caller override)
    2. `crane/src/libane_bridge.dylib` co-located with this repo (fresh,
       maintained alongside the bridge .h/.m sources here)

    The driver must be self-contained for open-source use. Do not fall back to
    a developer-local external checkout; if the repo-local dylib is missing,
    callers should rebuild `crane/`.
    """
    explicit = os.environ.get("EDGEFLOW_ANE_BRIDGE_PATH")
    if explicit:
        return explicit
    repo_local = Path(__file__).resolve().parents[2] / "crane" / "src" / "libane_bridge.dylib"
    return str(repo_local)


DEFAULT_ANE_BRIDGE_PATH = _resolve_default_ane_bridge_path()
_LIBC = ctypes.CDLL(None)
_LIBC.free.argtypes = [ctypes.c_void_p]
_LIBC.free.restype = None

ANE_BRIDGE_OUTPUT_CONSUMER_FN = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
)


def _probe_bridge_symbol(path: str, symbol: str) -> bool:
    if not Path(path).exists():
        return False
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return False
    return getattr(lib, symbol, None) is not None


_HAS_SCOPED_CONSUME = _probe_bridge_symbol(DEFAULT_ANE_BRIDGE_PATH, "ane_bridge_consume_surface_lease")


class ANEBridgeError(RuntimeError):
    pass


class ANEBridgeLibrary:
    def __init__(self, path: str | os.PathLike[str] = DEFAULT_ANE_BRIDGE_PATH) -> None:
        self.path = str(path)
        if not Path(self.path).exists():
            raise ANEBridgeError(f"ANE bridge dylib not found: {self.path}")

        self.lib = ctypes.CDLL(self.path)
        self._bind_symbols()

        if self.lib.ane_bridge_init() != 0:
            raise ANEBridgeError("ane_bridge_init failed")

    def _bind_symbols(self) -> None:
        self.lib.ane_bridge_init.argtypes = []
        self.lib.ane_bridge_init.restype = ctypes.c_int

        self.lib.ane_bridge_compile.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.ane_bridge_compile.restype = ctypes.c_void_p

        self.lib.ane_bridge_compile_multi_weights.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.ane_bridge_compile_multi_weights.restype = ctypes.c_void_p

        self.lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]
        self.lib.ane_bridge_eval.restype = ctypes.c_bool

        self.lib.ane_bridge_eval_batch.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        self.lib.ane_bridge_eval_batch.restype = ctypes.c_bool

        self.lib.ane_bridge_write_input.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.ane_bridge_write_input.restype = ctypes.c_bool

        self.lib.ane_bridge_read_output.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.ane_bridge_read_output.restype = ctypes.c_bool

        self.lib.ane_bridge_get_input_surface_id.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.lib.ane_bridge_get_input_surface_id.restype = ctypes.c_uint32

        self.lib.ane_bridge_get_output_surface_id.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.lib.ane_bridge_get_output_surface_id.restype = ctypes.c_uint32

        get_output_surface_size = getattr(self.lib, "ane_bridge_get_output_surface_size_bytes", None)
        if get_output_surface_size is not None:
            get_output_surface_size.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            get_output_surface_size.restype = ctypes.c_size_t

        prove_output_surface_direct_map = getattr(self.lib, "ane_bridge_prove_output_surface_direct_map", None)
        if prove_output_surface_direct_map is not None:
            prove_output_surface_direct_map.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            prove_output_surface_direct_map.restype = ctypes.c_int

        consume_output_surface = getattr(self.lib, "ane_bridge_consume_output_surface", None)
        if consume_output_surface is not None:
            consume_output_surface.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ANE_BRIDGE_OUTPUT_CONSUMER_FN,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            consume_output_surface.restype = ctypes.c_int

        alloc_surface_lease = getattr(self.lib, "ane_bridge_alloc_surface_lease", None)
        if alloc_surface_lease is not None:
            alloc_surface_lease.argtypes = [ctypes.c_size_t]
            alloc_surface_lease.restype = ctypes.c_void_p

        free_surface_lease = getattr(self.lib, "ane_bridge_free_surface_lease", None)
        if free_surface_lease is not None:
            free_surface_lease.argtypes = [ctypes.c_void_p]
            free_surface_lease.restype = None

        surface_lease_id = getattr(self.lib, "ane_bridge_surface_lease_id", None)
        if surface_lease_id is not None:
            surface_lease_id.argtypes = [ctypes.c_void_p]
            surface_lease_id.restype = ctypes.c_uint32

        surface_lease_size = getattr(self.lib, "ane_bridge_surface_lease_size_bytes", None)
        if surface_lease_size is not None:
            surface_lease_size.argtypes = [ctypes.c_void_p]
            surface_lease_size.restype = ctypes.c_size_t

        bind_output_surface_lease = getattr(self.lib, "ane_bridge_bind_kernel_output_surface_lease", None)
        if bind_output_surface_lease is not None:
            bind_output_surface_lease.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            bind_output_surface_lease.restype = ctypes.c_bool

        consume_surface_lease = getattr(self.lib, "ane_bridge_consume_surface_lease", None)
        if consume_surface_lease is not None:
            consume_surface_lease.argtypes = [
                ctypes.c_void_p,
                ANE_BRIDGE_OUTPUT_CONSUMER_FN,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            consume_surface_lease.restype = ctypes.c_int

        read_surface_lease = getattr(self.lib, "ane_bridge_read_surface_lease", None)
        if read_surface_lease is not None:
            read_surface_lease.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            read_surface_lease.restype = None

        write_surface_lease = getattr(self.lib, "ane_bridge_write_surface_lease", None)
        if write_surface_lease is not None:
            write_surface_lease.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            write_surface_lease.restype = None

        self.lib.ane_bridge_bind_input_surface_id.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint32,
        ]
        self.lib.ane_bridge_bind_input_surface_id.restype = ctypes.c_bool

        self.lib.ane_bridge_bind_output_surface_id.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint32,
        ]
        self.lib.ane_bridge_bind_output_surface_id.restype = ctypes.c_bool

        rebind_surface = getattr(self.lib, "ane_bridge_rebind_surface", None)
        if rebind_surface is not None:
            rebind_surface.argtypes = [
                ctypes.c_void_p,   # dst kernel handle
                ctypes.c_bool,     # dst_is_input
                ctypes.c_int,      # dst_idx
                ctypes.c_void_p,   # src kernel handle
                ctypes.c_bool,     # src_is_input
                ctypes.c_int,      # src_idx
            ]
            rebind_surface.restype = ctypes.c_bool

        self.lib.ane_bridge_free.argtypes = [ctypes.c_void_p]
        self.lib.ane_bridge_free.restype = None

        self.lib.ane_bridge_build_weight_blob.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.ane_bridge_build_weight_blob.restype = ctypes.c_void_p


def _build_dummy_weight_blob(bridge: ANEBridgeLibrary) -> tuple[ctypes.c_void_p, int]:
    dummy_value = (ctypes.c_float * 1)(0.0)
    out_len = ctypes.c_size_t()
    blob_ptr = bridge.lib.ane_bridge_build_weight_blob(dummy_value, 1, 1, ctypes.byref(out_len))
    if not blob_ptr:
        raise ANEBridgeError("ane_bridge_build_weight_blob failed")
    return blob_ptr, int(out_len.value)


def _free_bridge_blob(blob_ptr: ctypes.c_void_p) -> None:
    _LIBC.free(blob_ptr)


def build_dyn_matmul_mil(*, ic: int, oc: int, seq: int) -> str:
    sp = seq + oc
    parts = [
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n",
        mil_main_func_open(f"tensor<fp32, [1, {ic}, 1, {sp}]> x"),
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
        f"        tensor<fp16, [1, {ic}, 1, {sp}]> xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n"
        "        tensor<int32, [4]> ba = const()[name = string(\"ba\"), val = tensor<int32, [4]>([0,0,0,0])];\n"
        f"        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1,{ic},1,{seq}])];\n"
        f"        tensor<fp16, [1,{ic},1,{seq}]> act = slice_by_size(x=xh,begin=ba,size=sa)[name=string(\"act\")];\n"
        f"        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0,0,0,{seq}])];\n"
        f"        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1,{ic},1,{oc}])];\n"
        f"        tensor<fp16, [1,{ic},1,{oc}]> wt = slice_by_size(x=xh,begin=bw,size=sw)[name=string(\"wt\")];\n"
        f"        tensor<int32, [4]> ra = const()[name = string(\"ra\"), val = tensor<int32, [4]>([1,1,{ic},{seq}])];\n"
        f"        tensor<fp16, [1,1,{ic},{seq}]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n"
        "        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0,1,3,2])];\n"
        f"        tensor<fp16, [1,1,{seq},{ic}]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n"
        f"        tensor<int32, [4]> rw = const()[name = string(\"rw\"), val = tensor<int32, [4]>([1,1,{ic},{oc}])];\n"
        f"        tensor<fp16, [1,1,{ic},{oc}]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n"
        "        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n"
        f"        tensor<fp16, [1,1,{seq},{oc}]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"mm\")];\n"
        f"        tensor<fp16, [1,1,{oc},{seq}]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n"
        f"        tensor<int32, [4]> ro = const()[name = string(\"ro\"), val = tensor<int32, [4]>([1,{oc},1,{seq}])];\n"
        f"        tensor<fp16, [1,{oc},1,{seq}]> yr = reshape(shape=ro,x=yt)[name=string(\"yr\")];\n"
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
        f"        tensor<fp32, [1,{oc},1,{seq}]> y = cast(dtype = to32, x = yr)[name = string(\"cout\")];\n"
        "    } -> (y);\n"
        "}\n",
    ]
    return "".join(parts)


def build_baked_linear_mil(*, ic: int, oc: int, seq: int, weight_path: str = "@model_path/weights/weight.bin") -> str:
    parts = [
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n",
        mil_main_func_open(f"tensor<fp32, [1, {ic}, 1, {seq}]> x"),
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        f"        tensor<fp16, [1, {ic}, 1, {seq}]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        f"        tensor<fp16, [{oc}, {ic}, 1, 1]> W = const()[name = string(\"W\"), "
        f"val = tensor<fp16, [{oc}, {ic}, 1, 1]>(BLOBFILE(path = string(\"{weight_path}\"), offset = uint64(64)))];\n"
        f"        tensor<fp16, [1, {oc}, 1, {seq}]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        f"        tensor<fp32, [1,{oc},1,{seq}]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n"
        "}\n",
    ]
    return "".join(parts)


def pack_baked_linear_input(activations: np.ndarray) -> np.ndarray:
    if activations.ndim != 2:
        raise ValueError("activations must be rank-2 (seq, input_channels)")
    seq, ic = activations.shape
    packed = np.empty((1, ic, 1, seq), dtype=np.float32)
    packed[0, :, 0, :] = np.asarray(activations, dtype=np.float32).T
    return packed


def pack_dyn_matmul_input(activations: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if activations.ndim != 2:
        raise ValueError("activations must be rank-2 (seq, input_channels)")
    if weights.ndim != 2:
        raise ValueError("weights must be rank-2 (input_channels, output_channels)")
    seq, ic = activations.shape
    w_ic, oc = weights.shape
    if ic != w_ic:
        raise ValueError("activations.shape[1] must equal weights.shape[0]")

    packed = np.empty((1, ic, 1, seq + oc), dtype=np.float32)
    packed[0, :, 0, :seq] = np.asarray(activations, dtype=np.float32).T
    packed[0, :, 0, seq:] = np.asarray(weights, dtype=np.float32)
    return packed


def _unpack_dyn_matmul_output(output: np.ndarray) -> np.ndarray:
    if output.ndim != 4 or output.shape[0] != 1 or output.shape[2] != 1:
        raise ValueError("ANE output must have shape (1, output_channels, 1, seq)")
    return np.asarray(output[0, :, 0, :], dtype=np.float32).T


def run_dyn_matmul(
    activations: np.ndarray,
    weights: np.ndarray,
    *,
    bridge_path: str | os.PathLike[str] = DEFAULT_ANE_BRIDGE_PATH,
) -> np.ndarray:
    activations_f32 = np.asarray(activations, dtype=np.float32)
    weights_f32 = np.asarray(weights, dtype=np.float32)
    seq, ic = activations_f32.shape
    w_ic, oc = weights_f32.shape
    if ic != w_ic:
        raise ValueError("activations.shape[1] must equal weights.shape[0]")

    packed_input = np.ascontiguousarray(pack_dyn_matmul_input(activations_f32, weights_f32))
    output = np.empty((1, oc, 1, seq), dtype=np.float32)
    input_sizes = (ctypes.c_size_t * 1)(packed_input.nbytes)
    output_sizes = (ctypes.c_size_t * 1)(output.nbytes)
    mil = build_dyn_matmul_mil(ic=ic, oc=oc, seq=seq).encode("utf-8")

    bridge = ANEBridgeLibrary(bridge_path)
    dummy_blob_ptr, dummy_blob_len = _build_dummy_weight_blob(bridge)
    kernel = bridge.lib.ane_bridge_compile(
        mil,
        len(mil),
        dummy_blob_ptr,
        dummy_blob_len,
        1,
        input_sizes,
        1,
        output_sizes,
    )
    if not kernel:
        _free_bridge_blob(dummy_blob_ptr)
        raise ANEBridgeError("ane_bridge_compile failed")

    try:
        if not bridge.lib.ane_bridge_write_input(
            kernel,
            0,
            packed_input.ctypes.data_as(ctypes.c_void_p),
            packed_input.nbytes,
        ):
            raise ANEBridgeError("ane_bridge_write_input failed")
        if not bridge.lib.ane_bridge_eval(kernel):
            raise ANEBridgeError("ane_bridge_eval failed")
        if not bridge.lib.ane_bridge_read_output(
            kernel,
            0,
            output.ctypes.data_as(ctypes.c_void_p),
            output.nbytes,
        ):
            raise ANEBridgeError("ane_bridge_read_output failed")
    finally:
        bridge.lib.ane_bridge_free(kernel)
        _free_bridge_blob(dummy_blob_ptr)

    return _unpack_dyn_matmul_output(output)

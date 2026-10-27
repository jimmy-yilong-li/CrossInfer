"""Model-free regressions for the native Qwen3.5 session extension."""

from pathlib import Path
import ctypes
from dataclasses import replace
import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from edgeflow_split import crossinfer_session as cs
from test_crossinfer_session import _FakeCFunc, _FakeLibrary


@pytest.fixture(scope="module", params=["pinned", "upstream"])
def native_session_test(tmp_path_factory, request):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++17 compiler required")
    root = Path(__file__).resolve().parents[1]
    runtime = root / "tools/edgeflow_llm_runtime"
    llama = runtime / "llama_src"
    if request.param == "upstream":
        external = os.environ.get("EDGEFLOW_QWEN35_LLAMA_ROOT")
        if not external:
            pytest.skip("set EDGEFLOW_QWEN35_LLAMA_ROOT for the newer-header compile check")
        llama = Path(external)
    output = tmp_path_factory.mktemp("qwen35-session") / "session_test"
    subprocess.run(
        [compiler, "-std=c++17", "-O1", "-ffunction-sections", "-fdata-sections",
         "-Wno-deprecated-declarations", "-Wl,-dead_strip",
         "-I", str(runtime / "include"), "-I", str(llama / "include"),
         "-I", str(llama / "src"), "-I", str(llama / "ggml/include"),
         str(root / "tests/native_qwen35_session.cpp"), "-o", str(output)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    return output


@pytest.mark.parametrize("case", ["historical", "current", "ordinary", "recurrent",
                                  "branch-source", "positions", "capacity"])
def test_native_fork_contract(native_session_test, case):
    result = subprocess.run(
        [str(native_session_test), case], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PASS"


@pytest.fixture(scope="module")
def native_surface_test(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("Objective-C++ compiler required")
    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp("qwen35-surface") / "surface_test"
    subprocess.run(
        [compiler, "-std=c++17", "-O1", "-Wl,-dead_strip", "-fobjc-arc",
         "-I", str(root / "tools/edgeflow_llm_runtime/include"),
         "-I", str(root / "crane/src"), "-framework", "Foundation",
         str(root / "tests/native_qwen35_surface.mm"), "-o", str(output)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    return output


@pytest.mark.parametrize("case", ["success", "short", "failure", "alias"])
def test_native_lease_callback(native_surface_test, case):
    result = subprocess.run([str(native_surface_test), case], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PASS"


def test_qwen35_contract_is_explicit_and_paired():
    assert hasattr(cs, "SessionModelFamily"), "missing session Qwen3.5 family"
    config = cs.SessionConfig(
        Path("unused.gguf"), Path("unused.dylib"),
        model_family=cs.SessionModelFamily.QWEN35,
        position_contract=cs.PositionContract.QWEN35_IMROPE,
    )
    assert cs._prepare_session_params(config).model_family == 3
    with pytest.raises(ValueError, match="Qwen3.5"):
        cs._prepare_session_params(replace(config, position_contract=cs.PositionContract.TEXT_1D))
    with pytest.raises(ValueError, match="Qwen3.5"):
        cs._prepare_session_params(replace(config, prompt_contract_version=2))


class _Lease:
    surface_bytes = 144 * 8 * 4
    surface_id = 11

    def _handle_ptr(self):
        return ctypes.c_void_p(123)

    def read_into(self, _):
        raise AssertionError("lease must not be materialized")


@pytest.mark.parametrize("status", [0, 4])
def test_surface_append_borrows_and_releases_descriptor(status):
    assert hasattr(cs.StatefulLLMSession, "append_qwen35_visual"), "missing lease append"
    import threading

    session = object.__new__(cs.StatefulLLMSession)
    session._owner_thread = threading.get_ident()
    session._ctx = ctypes.c_void_p(1)
    session._signature = SimpleNamespace(n_seq_max=3, model_family=cs.SessionModelFamily.QWEN35)
    fake = _FakeLibrary()
    calls = []

    def append(ctx, seq, request_ptr, lease, out_ptr):
        request = request_ptr._obj
        calls.append(seq)
        assert not request.visual_embd
        assert lease.value == 123
        assert tuple(request.image_grid_thw) == (1, 24, 24)
        out = out_ptr._obj
        out.surface_pointer = out.decoder_pointer = 0x10000
        out.payload_bytes = 144 * 8 * 4
        out.next_position = 17
        return status

    fake.edgeflow_llm_session_append_qwen35_visual_from_surface_lease = _FakeCFunc(append)
    fake.edgeflow_llm_surface_append_receipt_sizeof = _FakeCFunc(lambda: 32)
    session._lib = fake
    released = []
    descriptor = SimpleNamespace(
        surface_lease=_Lease(), token_count=144, hidden_size=8, dtype="float32",
        image_grid_thw=(1, 24, 24), released=False,
        data_plane="ane_output_surface", release=lambda: released.append(True),
    )
    try:
        if status:
            with pytest.raises(RuntimeError, match="append"):
                session.append_qwen35_visual(0, descriptor=descriptor)
        else:
            receipt = session.append_qwen35_visual(0, descriptor=descriptor)
            assert receipt.surface_pointer == receipt.decoder_pointer == 0x10000
            assert receipt.payload_bytes == 4608
            assert receipt.next_position == 17
        assert calls == [0] and released == [True]
    finally:
        session._ctx = None

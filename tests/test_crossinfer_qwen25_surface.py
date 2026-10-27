import ctypes
from types import SimpleNamespace

import pytest
from test_crossinfer_root_state import (
    session_factory as session_factory,  # noqa: PLC0414 - pytest fixture re-export
)
from test_crossinfer_session import _FakeCFunc

from edgeflow_split import crossinfer_session as cs


class Lease:
    surface_bytes = 196 * 2048 * 4

    def _handle_ptr(self):
        return ctypes.c_void_p(123)

    def read_into(self, _):
        raise AssertionError("unexpected host materialization")


@pytest.mark.parametrize("status", [0, 9])
def test_surface_native_receipt_and_release(session_factory, status):
    session, lib = session_factory()
    released, calls = [], []

    def append(_, sequence, request_ptr, lease, receipt_ptr):
        request, receipt = request_ptr._obj, receipt_ptr._obj
        assert sequence == 0 and lease.value == 123 and not request.visual_embd
        assert tuple(request.image_grid_thw) == (1, 28, 28)
        receipt.surface_pointer = receipt.decoder_pointer = 0x10000
        receipt.payload_bytes = Lease.surface_bytes
        receipt.next_position = 18
        calls.append(True)
        return status

    lib.edgeflow_llm_session_append_qwen25vl_visual_from_surface_lease = _FakeCFunc(append)
    lib.edgeflow_llm_surface_append_receipt_sizeof = _FakeCFunc(
        lambda: ctypes.sizeof(cs._SurfaceAppendReceipt)
    )
    cs._bind_session_library(lib)
    descriptor = SimpleNamespace(
        surface_lease=Lease(), released=False, dtype="float32", token_count=196,
        hidden_size=2048, image_grid_thw=(1, 28, 28), data_plane="ane_output_surface",
        release=lambda: released.append(True),
    )
    if status:
        with pytest.raises(RuntimeError):
            session.append_qwen25vl_visual_surface(0, descriptor=descriptor)
    else:
        receipt = session.append_qwen25vl_visual_surface(0, descriptor=descriptor)
        assert receipt.surface_pointer == receipt.decoder_pointer == 0x10000
        assert receipt.payload_bytes == Lease.surface_bytes
    assert calls == [True] and released == [True]


def test_packed_surface_cannot_be_used_as_plain_image(session_factory):
    session, _ = session_factory()
    released = []
    descriptor = SimpleNamespace(
        text_prefix_tokens=4, text_suffix_tokens=11,
        release=lambda: released.append(True),
    )
    with pytest.raises(ValueError, match="packed"):
        session.append_qwen25vl_visual_surface(0, descriptor=descriptor)
    assert released == [True]


def test_closed_session_returns_borrowed_surface(session_factory):
    session, _ = session_factory()
    session.close()
    released = []
    descriptor = SimpleNamespace(release=lambda: released.append(True))
    with pytest.raises(RuntimeError, match="closed"):
        session.append_qwen25vl_visual_surface(0, descriptor=descriptor)
    assert released == [True]

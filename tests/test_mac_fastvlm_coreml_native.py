"""Shared-output bridge contracts using a fake C ABI; never load a model."""

import ctypes
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mac_fastvlm_coreml_native_contract", ROOT / "tools/mac_fastvlm_coreml_native.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Function:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


class Library:
    def __init__(self, width=1536):
        self.width = width
        self.calls = []
        self.freed = []
        self.generations = [0, 0]
        self.corrupt = lambda output: None
        self.status = 0
        self.fv_coreml_error = Function(lambda: b"native prediction rejected backing")
        self.fv_coreml_init = Function(lambda path, width: 123)
        self.fv_coreml_free = Function(self.freed.append)
        self.fv_coreml_predict = Function(self.predict)

    def predict(self, context, pixels, count, slot, result):
        assert context == 123 and count == 3 * 1024 * 1024
        self.calls.append((ctypes.cast(pixels, ctypes.c_void_p).value, slot))
        output = ctypes.cast(result, ctypes.POINTER(module._Prediction)).contents
        self.generations[slot] += 1
        output.lease_handle = 100 + slot
        output.pointer = 4096 * (slot + 1)
        output.generation = self.generations[slot]
        output.nbytes = 256 * self.width * 4
        output.shape[:] = (1, 256, self.width)
        output.strides[:] = (256 * self.width, self.width, 1)
        output.surface_id = 42 + slot
        output.slot = slot
        output.object_matches = output.pointer_matches = 1
        self.corrupt(output)
        return self.status


@pytest.fixture
def bridge(monkeypatch):
    library = Library()
    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: library)
    visual = module.NativeCoreMLVisual("visual.mlpackage", "native.dylib")
    yield visual, library
    visual.close()


@pytest.fixture
def pixels():
    return np.empty((1, 3, 1024, 1024), dtype=np.float32)


def test_descriptor_exposes_leases_and_rotates_exactly_two_slots(bridge, pixels):
    visual, library = bridge
    features = [visual.predict(pixels, index) for index in range(3)]
    assert [item.lease_handle for item in features] == [100, 101, 100]
    assert [item.surface_id for item in features] == [42, 43, 42]
    assert [item.receipt["generation"] for item in features] == [1, 1, 2]
    assert features[0].pointer == 4096
    assert features[0].shape == (1, 256, 1536)
    assert features[0].receipt["object_identity"] is True
    assert features[0].receipt["pointer_identity"] is True
    assert features[0].receipt["strides"] == [256 * 1536, 1536, 1]
    assert features[0].receipt["pointer_scope"] == "diagnostic_only"
    assert features[0].receipt["producer_scope"] == "ane_bridge_produce_surface_lease"
    assert all(pointer == pixels.ctypes.data for pointer, _ in library.calls)


@pytest.mark.parametrize("field,value", [
    ("lease_handle", None), ("pointer", None), ("surface_id", 0),
    ("object_matches", 0), ("pointer_matches", 0), ("slot", 1), ("nbytes", 0),
])
def test_invalid_native_receipt_is_not_silently_accepted(bridge, pixels, field, value):
    visual, library = bridge
    library.corrupt = lambda output: setattr(output, field, value)
    with pytest.raises(RuntimeError, match="backing contract"):
        visual.predict(pixels, 0)


@pytest.mark.parametrize("field", ["shape", "strides"])
def test_layout_mismatch_is_rejected(bridge, pixels, field):
    visual, library = bridge
    library.corrupt = lambda output: getattr(output, field).__setitem__(1, 7)
    with pytest.raises(RuntimeError, match="backing contract"):
        visual.predict(pixels, 0)


def test_native_failure_is_reported_without_fallback(bridge, pixels):
    visual, library = bridge
    library.status = -1
    with pytest.raises(RuntimeError, match="native prediction rejected backing"):
        visual.predict(pixels, 0)


def test_invalid_input_is_rejected_before_native_prediction(bridge, pixels):
    visual, library = bridge
    with pytest.raises(ValueError, match="shape"):
        visual.predict(pixels[:, :, :1, :], 0)
    with pytest.raises(TypeError, match="float32"):
        visual.predict(np.empty(pixels.shape, dtype=np.float16), 0)
    with pytest.raises(ValueError, match="contiguous"):
        visual.predict(pixels[..., ::-1], 0)
    with pytest.raises(ValueError, match="index"):
        visual.predict(pixels, -1)
    assert library.calls == []


def test_close_is_idempotent_and_prevents_new_predictions(bridge, pixels):
    visual, library = bridge
    visual.close()
    visual.close()
    assert library.freed == [123]
    with pytest.raises(RuntimeError, match="closed"):
        visual.predict(pixels, 0)


def test_native_source_requires_identity_and_uses_writable_surface_scope():
    source = (ROOT / "tools/mac_fastvlm_coreml_native.mm").read_text()
    assert "ane_bridge_alloc_surface_lease" in source
    assert "ane_bridge_produce_surface_lease" in source
    assert "const_cast<void *>(base_address)" in source
    assert "capacity < prefix_bytes + required_bytes + suffix_bytes" in source
    assert "IOSurfaceLookup" not in source
    assert "IOSurfaceLock(" not in source and "IOSurfaceUnlock(" not in source
    assert "options.outputBackings" in source
    assert "actual != backing" in source
    assert "actual.dataPointer != base" in source
    assert "isEqualToArray:shape" in source
    assert "isEqualToArray:strides" in source
    # Packed prompts copy text rows only; prediction writes visual rows in place.
    assert source.count("std::memcpy(") == 2
    assert "std::memcpy(storage, request.text, prefix_bytes)" in source
    assert "std::memcpy(storage + prefix_bytes + required_bytes," in source
    assert "copy_n(" not in source

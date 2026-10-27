"""Pooled producer ABI and ownership, without loading CoreML models."""

import ctypes

import numpy as np
import pytest


class Function:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


def test_producer_uses_the_acquired_lease_and_original_input(monkeypatch):
    from edgeflow_split import coreml_surface as module

    calls = []
    clock = iter((10.0, 12.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))

    class Lease:
        lease_handle = 456
        releases = 0

        def release(self):
            self.releases += 1

    lease = Lease()

    class Pool:
        def acquire(self, timeout=None):
            calls.append("acquire")
            return lease

    pixels = np.ones((576, 1536), dtype=np.float32)

    def predict(context, pointer, count, handle, result):
        calls.append("predict")
        assert context == 123 and handle == 456 and count == pixels.size
        assert ctypes.cast(pointer, ctypes.c_void_p).value == pixels.ctypes.data
        output = ctypes.cast(result, ctypes.POINTER(module.CoreMLOutput)).contents
        output.lease_handle = 456
        output.pointer = 4096
        output.nbytes = 144 * 1024 * 4
        output.shape[:] = (1, 144, 1024)
        output.strides[:] = (144 * 1024, 1024, 1)
        output.surface_id = 42
        output.generation = 1
        output.object_matches = output.pointer_matches = 1
        return 0

    class Library:
        ef_coreml_init = Function(lambda path, width: 123)
        ef_coreml_predict_into = Function(predict)
        fv_coreml_error = Function(lambda: b"prediction failed")
        fv_coreml_free = Function(lambda context: calls.append("close"))

    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: Library())
    with module.CoreMLSurfaceModel("visual.mlpackage", "runtime.dylib", width=1024,
                                  tokens=144, input_shape=(576, 1536)) as model:
        result = model.predict_to_descriptor(pixels, pool=Pool(), frame_id=3,
                                             grid_thw=(1, 24, 24))
    assert calls == ["acquire", "predict", "close"]
    assert result.surface_lease is lease and lease.releases == 0
    assert result.handoff_receipt["producer_pointer"] == 4096
    assert result.handoff_receipt["application_embedding_copy_bytes"] == 0
    assert result.handoff_receipt["prediction_started_s"] == 10.0
    assert result.handoff_receipt["prediction_finished_s"] == 12.0
    result.release()
    assert lease.releases == 1


def test_native_failure_returns_pool_lease(monkeypatch):
    from edgeflow_split import coreml_surface as module

    class Lease:
        lease_handle = 456
        releases = 0

        def release(self):
            self.releases += 1

    lease = Lease()

    class Pool:
        def acquire(self, timeout=None):
            return lease

    class Library:
        ef_coreml_init = Function(lambda *args: 123)
        ef_coreml_predict_into = Function(lambda *args: -1)
        fv_coreml_error = Function(lambda: b"prediction failed")
        fv_coreml_free = Function(lambda context: None)

    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: Library())
    with module.CoreMLSurfaceModel("p", "l", width=1024, tokens=144,
                                  input_shape=(576, 1536)) as model:
        with pytest.raises(RuntimeError, match="prediction failed"):
            model.predict_to_descriptor(np.ones((576, 1536), np.float32), pool=Pool(),
                                        frame_id=0, grid_thw=(1, 24, 24))
    assert lease.releases == 1


def test_chain_borrows_intermediate_surface_and_retains_it_until_next_prediction():
    from types import SimpleNamespace
    from edgeflow_split.coreml_surface import CoreMLSurfaceChain

    storage = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    calls = []

    class Descriptor:
        def __init__(self, name):
            self.name = name
            self.released = False
            self.token_count, self.hidden_size = 3, 4
            self.handoff_receipt = {"producer_pointer": storage.ctypes.data,
                                    "prediction_started_s": 1, "prediction_finished_s": 2}

        def release(self):
            self.released = True
            calls.append("release_" + self.name)

    first, last = Descriptor("first"), Descriptor("last")

    def predict_first(pixels, **kwargs):
        assert kwargs["pool"] == "intermediate"
        calls.append("first")
        return first

    def predict_last(pixels, **kwargs):
        assert not first.released
        assert pixels.ctypes.data == storage.ctypes.data
        assert pixels.shape == (3, 4)
        assert kwargs["pool"] == "final"
        calls.append("last")
        return last

    chain = CoreMLSurfaceChain([
        SimpleNamespace(predict_to_descriptor=predict_first, input_shape=(2, 6)),
        SimpleNamespace(predict_to_descriptor=predict_last, input_shape=(3, 4)),
    ], intermediate_pool="intermediate")
    result = chain.predict_to_descriptor(np.ones((2, 6), np.float32), pool="final",
                                         frame_id=1, grid_thw=(1, 2, 6))
    assert result is last and not last.released
    assert calls == ["first", "last", "release_first"]
    assert result.handoff_receipt["intermediate_copy_bytes"] == 0


def test_packed_prompt_keeps_visual_output_in_offset_lease_view(monkeypatch):
    from types import SimpleNamespace

    from edgeflow_split import coreml_surface as module

    text = np.arange(12, dtype=np.float32).reshape(3, 4)
    lease = SimpleNamespace(lease_handle=456, release=lambda: None)

    def predict(context, pixels, count, handle, text_pointer, prefix, suffix, result):
        assert handle == 456 and (prefix, suffix) == (2, 1)
        assert ctypes.cast(text_pointer, ctypes.c_void_p).value == text.ctypes.data
        output = ctypes.cast(result, ctypes.POINTER(module.CoreMLOutput)).contents
        output.pointer = 4096 + 2 * 4 * 4
        output.nbytes = 6 * 4 * 4
        output.shape[:] = (1, 6, 4)
        output.strides[:] = (24, 4, 1)
        output.surface_id = 42
        output.object_matches = output.pointer_matches = 1
        return 0

    library = SimpleNamespace(
        ef_coreml_init=Function(lambda *args: 123),
        ef_coreml_predict_into=Function(lambda *args: pytest.fail("unpacked call")),
        ef_coreml_predict_packed=Function(predict),
        fv_coreml_error=Function(lambda: b"error"),
        fv_coreml_free=Function(lambda *args: None),
    )
    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: library)
    with module.CoreMLSurfaceModel("p", "l", width=4, tokens=6, input_shape=(2, 3)) as model:
        descriptor = model.predict_to_descriptor(
            np.ones((2, 3), np.float32), pool=SimpleNamespace(acquire=lambda **kw: lease),
            frame_id=0, grid_thw=(1, 4, 6), text_embeddings=text,
            prefix_tokens=2, prompt="question",
        )
    assert descriptor.text_prefix_tokens == 2 and descriptor.text_suffix_tokens == 1
    assert descriptor.prepared_prompt == "question"
    assert descriptor.handoff_receipt["visual_offset_bytes"] == 32
    assert descriptor.handoff_receipt["lease_base_pointer"] == 4096
    assert descriptor.handoff_receipt["application_visual_embedding_copy_bytes"] == 0
    assert descriptor.handoff_receipt["application_text_embedding_copy_bytes"] == text.nbytes
    assert descriptor.handoff_receipt["prompt_tokens"] == 9
    with pytest.raises(RuntimeError, match="packed"):
        descriptor.materialize_to_batch()

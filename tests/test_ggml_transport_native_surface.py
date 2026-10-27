"""Tests for surface-backed native transport dispatch.

Covers:
- A. Top-level isinstance dispatch (tests 1-3)
- B. Static fallback via feature flag + capability check (tests 4-5)
- C. Surface path success (test 6)
- D. Surface path failure without retry (test 7)
- E. Ownership release in finally on both success and failure (test 8)

Not covered by these model-free tests:
- Real bridge or real llama.cpp
- Lease pool, overlap, multi-descriptor drain
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from edgeflow_split import _ggml_transport_native as native
from edgeflow_split.config import SplitConfig
from edgeflow_split.contracts import ImageEmbedsBatch
from edgeflow_split.model_spec import ModelFamily
from edgeflow_split.surface_backed_descriptor import SurfaceBackedVisualDescriptor


@dataclass
class FakeLease:
    surface_bytes: int = 1024
    fill_value: float = 0.0
    read_into_calls: int = 0
    release_calls: int = 0
    handle_ptr_value: int = 0x1234

    @property
    def released(self) -> bool:
        return self.release_calls > 0

    def read_into(self, dst) -> None:
        self.read_into_calls += 1
        dst.fill(self.fill_value)

    def release(self) -> None:
        self.release_calls += 1

    def _handle_ptr(self):
        return self.handle_ptr_value


@dataclass
class FakeBridgeCall:
    name: str
    args: tuple


@pytest.mark.parametrize("callback_fails", [False, True])
def test_generation_options_deliver_raw_bytes_and_reset(callback_fails):
    import ctypes
    from types import SimpleNamespace

    calls, tokens = [], []
    def setter(ctx, ignore_eog, callback, data):
        calls.append((ignore_eog, callback))
        return 0

    def on_token(token, piece):
        tokens.append((token, piece))
        if callback_fails:
            raise ValueError("consumer failed")

    runtime = SimpleNamespace(ctx=1, lib=SimpleNamespace(edgeflow_llm_set_generation_options=setter))
    def generate():
        with native._generation_options(runtime, on_token, True):
            piece = ctypes.create_string_buffer(b"a\x00b")
            rc = calls[0][1](12, ctypes.addressof(piece), 3, None)
            if rc:
                raise RuntimeError("native aborted")
    if callback_fails:
        with pytest.raises(ValueError, match="consumer failed"):
            generate()
    else:
        generate()
    assert tokens == [(12, b"a\x00b")]
    assert calls[0][0] == 1
    assert calls[-1][0] == 0 and not calls[-1][1]


def test_packed_surface_selects_packed_native_entry(monkeypatch, fake_descriptor):
    import ctypes
    from types import SimpleNamespace

    fake_descriptor.text_prefix_tokens = 4
    fake_descriptor.text_suffix_tokens = 11
    fake_descriptor.prepared_prompt = "question"
    calls = []

    def packed(ctx, req, lease, prefix, suffix, out):
        calls.append((ctx, prefix, suffix))
        assert ctypes.cast(req, ctypes.POINTER(native._EdgeFlowLLMRequest)).contents.num_visual_tokens == 196
        return 0

    runtime = SimpleNamespace(ctx=123, lib=SimpleNamespace(
        edgeflow_llm_generate_from_surface_lease_packed=packed))
    monkeypatch.setattr(native, "_parse_native_out", lambda *args: {"text": "The"})
    request = native._prepare_descriptor_request_for_native(fake_descriptor)
    request.update(prompt="question", max_new_tokens=1)
    assert native._invoke_native_surface_llm(runtime, request)["text"] == "The"
    assert calls == [(123, 4, 11)]
    request["prompt"] = "a different question"
    with pytest.raises(ValueError, match="prepared prompt"):
        native._invoke_native_surface_llm(runtime, request)


@pytest.fixture
def base_config() -> SplitConfig:
    return SplitConfig(
        text_backend="ggml",
        ggml_transport="native",
        ggml_model_path="/tmp/fake-qwen2vl.gguf",
        ggml_library_path="/tmp/libedgeflow_llm.dylib",
    )


@pytest.fixture
def surface_on_config(base_config: SplitConfig) -> SplitConfig:
    return replace(base_config, ggml_native_surface_path_enabled=True)


@pytest.fixture
def surface_off_config(base_config: SplitConfig) -> SplitConfig:
    return replace(base_config, ggml_native_surface_path_enabled=False)


@pytest.fixture
def fake_batch() -> ImageEmbedsBatch:
    return ImageEmbedsBatch(
        image_embeds=np.zeros((4, 8), dtype=np.float32),
        image_grid_thw=(1, 2, 2),
        num_visual_tokens=4,
        hidden_size=8,
        dtype="float32",
    )


@pytest.fixture
def fake_descriptor() -> SurfaceBackedVisualDescriptor:
    # Qwen v1 has no dtype discriminator and consumes only FP32 surfaces.
    # FP16 conversion/refusal is covered by test_qwen_default_handoff.py.
    lease = FakeLease(surface_bytes=196 * 2048 * 4)
    return SurfaceBackedVisualDescriptor(
        descriptor_id=1,
        surface_lease=lease,
        token_count=196,
        hidden_size=2048,
        dtype="float32",
        image_grid_thw=(1, 28, 28),
        frame_id=0,
        timestamp_ms=0.0,
    )


@pytest.fixture
def fake_true_zero_copy_descriptor() -> SurfaceBackedVisualDescriptor:
    lease = FakeLease(surface_bytes=196 * 2048 * 4)
    return SurfaceBackedVisualDescriptor(
        descriptor_id=2,
        surface_lease=lease,
        token_count=196,
        hidden_size=2048,
        dtype="float32",
        image_grid_thw=(1, 28, 28),
        frame_id=0,
        timestamp_ms=0.0,
        data_plane="ane_output_surface",
    )


@pytest.fixture
def bridge_call_log(monkeypatch):
    calls: list[FakeBridgeCall] = []

    def fake_generate_from_surface_lease(ctx_ptr, req_ptr, lease_ptr, out_ptr):
        calls.append(FakeBridgeCall("generate_from_surface_lease", (ctx_ptr, req_ptr, lease_ptr, out_ptr)))
        return 0

    monkeypatch.setattr(
        native,
        "edgeflow_llm_generate_from_surface_lease",
        fake_generate_from_surface_lease,
        raising=False,
    )
    return calls


@pytest.fixture
def fake_runtime(monkeypatch):
    class FakeCFunc:
        def __init__(self, func):
            self._func = func
            self.argtypes = None
            self.restype = None

        def __call__(self, *args, **kwargs):
            return self._func(*args, **kwargs)

    class FakeLib:
        def __init__(self) -> None:
            self.edgeflow_llm_free_result = FakeCFunc(lambda *args: None)

    class FakeRuntime:
        def __init__(self) -> None:
            self.lib = FakeLib()
            self.ctx = object()

    runtime = FakeRuntime()
    monkeypatch.setattr(native, "_load_native_llm_runtime", lambda cfg: runtime, raising=False)
    return runtime


@pytest.fixture
def force_capability_available(monkeypatch):
    monkeypatch.setattr(
        native, "_surface_path_capability_available",
        lambda config: True,
    )


@pytest.fixture
def force_capability_unavailable(monkeypatch):
    monkeypatch.setattr(
        native, "_surface_path_capability_available",
        lambda config: False,
    )


def test_dispatch_batch_routes_to_batch_path(base_config: SplitConfig, fake_batch: ImageEmbedsBatch, monkeypatch) -> None:
    batch_calls: list[tuple] = []
    descriptor_calls: list[tuple] = []

    monkeypatch.setattr(
        native,
        "_invoke_native_generate_from_batch",
        lambda *args, **kwargs: batch_calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        native,
        "_invoke_native_generate_from_descriptor",
        lambda *args, **kwargs: descriptor_calls.append((args, kwargs)),
        raising=False,
    )

    native._run_generate_native_visual_payload("img.png", "prompt", fake_batch, 8, base_config)

    assert len(batch_calls) == 1
    assert len(descriptor_calls) == 0
    args, _kwargs = batch_calls[0]
    assert args[2] is fake_batch


def test_dispatch_descriptor_routes_to_descriptor_path(
    surface_on_config: SplitConfig,
    fake_descriptor: SurfaceBackedVisualDescriptor,
    force_capability_available,
    monkeypatch,
) -> None:
    batch_calls: list[tuple] = []
    descriptor_calls: list[tuple] = []

    monkeypatch.setattr(
        native,
        "_invoke_native_generate_from_batch",
        lambda *args, **kwargs: batch_calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        native,
        "_invoke_native_generate_from_descriptor",
        lambda *args, **kwargs: descriptor_calls.append((args, kwargs)),
        raising=False,
    )

    native._run_generate_native_visual_payload("img.png", "prompt", fake_descriptor, 8, surface_on_config)

    assert len(descriptor_calls) == 1
    assert len(batch_calls) == 0
    args, _kwargs = descriptor_calls[0]
    assert args[2] is fake_descriptor


def test_dispatch_unknown_type_raises_type_error(base_config: SplitConfig) -> None:
    with pytest.raises(TypeError, match="unsupported visual payload type: object"):
        native._run_generate_native_visual_payload("img.png", "prompt", object(), 8, base_config)


def test_flag_off_descriptor_falls_back_via_materialize(
    surface_off_config: SplitConfig,
    fake_descriptor: SurfaceBackedVisualDescriptor,
    force_capability_available,
    monkeypatch,
) -> None:
    batch_calls: list[tuple] = []
    materialize_calls: list[str] = []

    monkeypatch.setattr(
        native,
        "_invoke_native_generate_from_batch",
        lambda *args, **kwargs: batch_calls.append((args, kwargs)),
        raising=False,
    )
    original_materialize = fake_descriptor.materialize_to_batch

    def tracked_materialize():
        materialize_calls.append("materialize")
        return original_materialize()

    monkeypatch.setattr(fake_descriptor, "materialize_to_batch", tracked_materialize)

    native._run_generate_native_visual_payload("img.png", "prompt", fake_descriptor, 8, surface_off_config)

    assert materialize_calls == ["materialize"]
    assert len(batch_calls) == 1
    args, _kwargs = batch_calls[0]
    assert isinstance(args[2], ImageEmbedsBatch)
    assert fake_descriptor.released
    assert fake_descriptor.surface_lease is None


def test_capability_missing_descriptor_falls_back_via_materialize(
    surface_on_config: SplitConfig,
    fake_descriptor: SurfaceBackedVisualDescriptor,
    force_capability_unavailable,
    monkeypatch,
) -> None:
    batch_calls: list[tuple] = []
    materialize_calls: list[str] = []

    monkeypatch.setattr(
        native,
        "_invoke_native_generate_from_batch",
        lambda *args, **kwargs: batch_calls.append((args, kwargs)),
        raising=False,
    )
    original_materialize = fake_descriptor.materialize_to_batch

    def tracked_materialize():
        materialize_calls.append("materialize")
        return original_materialize()

    monkeypatch.setattr(fake_descriptor, "materialize_to_batch", tracked_materialize)

    native._run_generate_native_visual_payload("img.png", "prompt", fake_descriptor, 8, surface_on_config)

    assert materialize_calls == ["materialize"]
    assert len(batch_calls) == 1
    args, _kwargs = batch_calls[0]
    assert isinstance(args[2], ImageEmbedsBatch)
    assert fake_descriptor.released


def test_true_zero_copy_descriptor_fails_loud_without_surface_consumer(
    surface_on_config: SplitConfig,
    fake_true_zero_copy_descriptor: SurfaceBackedVisualDescriptor,
    force_capability_unavailable,
    monkeypatch,
) -> None:
    batch_calls: list[tuple] = []
    materialize_calls: list[str] = []

    monkeypatch.setattr(
        native,
        "_invoke_native_generate_from_batch",
        lambda *args, **kwargs: batch_calls.append((args, kwargs)),
        raising=False,
    )
    original_materialize = fake_true_zero_copy_descriptor.materialize_to_batch

    def tracked_materialize():
        materialize_calls.append("materialize")
        return original_materialize()

    monkeypatch.setattr(fake_true_zero_copy_descriptor, "materialize_to_batch", tracked_materialize)

    with pytest.raises(RuntimeError, match="requires native surface"):
        native._run_generate_native_visual_payload(
            "img.png",
            "prompt",
            fake_true_zero_copy_descriptor,
            8,
            surface_on_config,
        )

    assert materialize_calls == []
    assert batch_calls == []
    assert fake_true_zero_copy_descriptor.released


def test_surface_path_success_invokes_generate_from_surface_lease(
    surface_on_config: SplitConfig,
    fake_descriptor: SurfaceBackedVisualDescriptor,
    force_capability_available,
    bridge_call_log,
    fake_runtime,
    monkeypatch,
) -> None:
    expected_handle = fake_descriptor.surface_lease._handle_ptr()
    monkeypatch.setattr(
        native,
        "_invoke_native_generate_from_batch",
        lambda *args, **kwargs: pytest.fail("batch path should not be used"),
        raising=False,
    )

    result = native._run_generate_native_visual_payload("img.png", "prompt", fake_descriptor, 8, surface_on_config)

    assert isinstance(result, native.GGMLGenerateResult)
    assert len(bridge_call_log) == 1
    assert bridge_call_log[0].name == "generate_from_surface_lease"
    assert bridge_call_log[0].args[2] == expected_handle
    assert fake_descriptor.released


def test_surface_path_failure_raises_without_batch_retry(
    surface_on_config: SplitConfig,
    fake_descriptor: SurfaceBackedVisualDescriptor,
    force_capability_available,
    bridge_call_log,
    fake_runtime,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        native,
        "edgeflow_llm_generate_from_surface_lease",
        lambda *args, **kwargs: 9,
        raising=False,
    )
    monkeypatch.setattr(
        native,
        "_invoke_native_generate_from_batch",
        lambda *args, **kwargs: pytest.fail("batch retry should not run"),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="edgeflow_llm_generate_from_surface_lease failed rc=9"):
        native._run_generate_native_visual_payload("img.png", "prompt", fake_descriptor, 8, surface_on_config)

    assert fake_descriptor.released


def test_descriptor_released_after_successful_surface_path(
    surface_on_config: SplitConfig,
    fake_descriptor: SurfaceBackedVisualDescriptor,
    force_capability_available,
    bridge_call_log,
    fake_runtime,
    monkeypatch,
) -> None:
    result = native._run_generate_native_visual_payload("img.png", "prompt", fake_descriptor, 8, surface_on_config)

    assert isinstance(result, native.GGMLGenerateResult)
    assert result.text == ""
    assert fake_descriptor.released
    assert fake_descriptor.surface_lease is None


def test_fastvlm_descriptor_routes_to_v2_surface_path(monkeypatch) -> None:
    """Explicit FastVLM family + SurfaceBackedVisualDescriptor -> v2 surface dispatch.

    Verifies that _run_generate_native_visual_payload routes FastVLM
    descriptors through _invoke_native_generate_from_descriptor_v2 instead
    of the v1 surface path or the batch fallback. Uses monkeypatch to
    intercept the v2 helper and assert it was called with a descriptor
    carrying the expected shape + fp16 dtype.
    """
    config = SplitConfig(
        model_id="custom-fastvlm-alias-without-hf-substring",
        model_family=ModelFamily.FASTVLM,
        ggml_native_surface_path_enabled=True,
        ggml_transport="native",
        ggml_library_path="/tmp/libedgeflow_llm.dylib",
    )
    assert config.detected_model_family == 1, "explicit FastVLM family should route to native family=1"
    assert config.fastvlm_lease_bytes == 458_752, "FastVLM lease bytes = 256*896*2"

    # Build a FastVLM-shaped descriptor (256 tokens x 896 hidden, fp16).
    lease = FakeLease(surface_bytes=256 * 896 * 2)
    descriptor = SurfaceBackedVisualDescriptor(
        descriptor_id=1,
        surface_lease=lease,
        token_count=256,
        hidden_size=896,
        dtype="float16",
        image_grid_thw=(1, 16, 16),
        frame_id=0,
        timestamp_ms=0.0,
    )

    # Monkeypatch the v2 descriptor dispatch helper to capture the call
    # without invoking the native C runtime. Also stub out v1 descriptor
    # and batch paths so any mis-routing will be caught.
    v2_calls: list[tuple] = []
    v1_calls: list[tuple] = []
    batch_calls: list[tuple] = []

    def fake_v2(image_path, prompt, desc, max_new_tokens, cfg):
        v2_calls.append((image_path, prompt, desc, max_new_tokens, cfg))
        return native.GGMLGenerateResult(
            text="ok",
            first_token_ms=1.0,
            prompt_eval_ms=1.0,
            eval_ms=1.0,
            eval_count=1,
        )

    monkeypatch.setattr(
        native,
        "_invoke_native_generate_from_descriptor_v2",
        fake_v2,
        raising=False,
    )
    monkeypatch.setattr(
        native,
        "_invoke_native_generate_from_descriptor",
        lambda *args, **kwargs: v1_calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        native,
        "_invoke_native_generate_from_batch",
        lambda *args, **kwargs: batch_calls.append((args, kwargs)),
        raising=False,
    )

    result = native._run_generate_native_visual_payload(
        "img.png",
        "describe the image",
        descriptor,
        8,
        config,
    )

    # Routing assertions: v2 called exactly once, v1 and batch untouched.
    assert len(v2_calls) == 1, f"expected v2 dispatch, got v1={len(v1_calls)} batch={len(batch_calls)}"
    assert len(v1_calls) == 0, "v1 descriptor path must not run for FastVLM"
    assert len(batch_calls) == 0, "batch fallback must not run for FastVLM"

    # v2 dispatch was given the right descriptor + config
    image_path, prompt, desc_arg, max_new_tokens, cfg_arg = v2_calls[0]
    assert desc_arg is descriptor
    assert prompt == "describe the image"
    assert max_new_tokens == 8
    assert cfg_arg is config
    assert desc_arg.token_count == 256
    assert desc_arg.hidden_size == 896
    assert desc_arg.dtype == "float16"

    # v2 helper's return value is plumbed back through unchanged.
    assert isinstance(result, native.GGMLGenerateResult)
    assert result.text == "ok"


def test_flag_off_descriptor_releases_lease_when_materialize_raises(
    surface_off_config: SplitConfig,
    fake_descriptor: SurfaceBackedVisualDescriptor,
    force_capability_available,
    monkeypatch,
) -> None:
    """A failing CPU readback must still release the surface lease.

    materialize_to_batch() releases only on SUCCESS; the fallback branch must
    release in a finally like its descriptor-path siblings, or the leaked
    lease permanently starves a capacity-1 LeasePool (the FastVLM adapter's
    configuration).
    """
    lease = fake_descriptor.surface_lease

    def exploding_read_into(dst) -> None:
        raise RuntimeError("simulated readback failure")

    monkeypatch.setattr(lease, "read_into", exploding_read_into)

    with pytest.raises(RuntimeError, match="simulated readback failure"):
        native._run_generate_native_visual_payload(
            "img.png", "prompt", fake_descriptor, 8, surface_off_config
        )

    assert fake_descriptor.released
    assert fake_descriptor.surface_lease is None
    assert lease.release_calls == 1

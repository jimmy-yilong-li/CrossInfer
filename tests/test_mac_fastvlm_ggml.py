"""Compile the real bridge against model-free stubs; never execute an engine."""

import ctypes
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fastvlm_ggml_contract", ROOT / "tools/mac_fastvlm_ggml.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class QueryInfo(ctypes.Structure):
    _fields_ = [("prefill_finished_s", ctypes.c_double),
                ("actual_count", ctypes.c_int32), ("root", ctypes.c_int32),
                ("cleanup_ok", ctypes.c_int32), ("logits_captured", ctypes.c_int32)]


class Lease(ctypes.Structure):
    _fields_ = [("base", ctypes.c_void_p), ("nbytes", ctypes.c_size_t),
                ("fail_before", ctypes.c_int), ("fail_after", ctypes.c_int)]


@pytest.fixture(scope="session")
def compiled_bridge(tmp_path_factory):
    directory = tmp_path_factory.mktemp("fastvlm-native")
    library = directory / "libfastvlm_test.dylib"
    command = ["clang++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
               "-dynamiclib" if sys.platform == "darwin" else "-shared", "-fPIC",
               str(ROOT / "tools/mac_fastvlm_ggml.cpp"),
               str(ROOT / "tests/native_fastvlm_stubs.cpp"),
               "-I", str(ROOT / "tools/edgeflow_llm_runtime/llama_src/include"),
               "-I", str(ROOT / "tools/edgeflow_llm_runtime/llama_src/ggml/include"),
               "-I", str(ROOT / "crane/src"), "-o", str(library)]
    if sys.platform != "darwin":
        command.append("-DCLOCK_UPTIME_RAW=CLOCK_MONOTONIC")
    subprocess.run(command, check=True, capture_output=True, text=True)
    return library


@pytest.fixture
def native(compiled_bridge):
    lib = ctypes.CDLL(str(compiled_bridge))
    lib.fv_test_reset.argtypes = []
    lib.fv_test_reset.restype = None
    lib.fv_test_set.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.fv_test_set.restype = None
    lib.fv_test_get.argtypes = [ctypes.c_char_p]
    lib.fv_test_get.restype = ctypes.c_int
    lib.fv_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.fv_init.restype = ctypes.c_void_p
    lib.fv_free.argtypes = [ctypes.c_void_p]
    lib.fv_free.restype = None
    lib.fv_error.restype = ctypes.c_char_p
    lib.fv_test_reset()
    yield lib
    assert all(lib.fv_test_get(key) == 0 for key in
               (b"live_models", b"live_contexts", b"live_samplers"))


@pytest.fixture
def language(native, compiled_bridge):
    instance = module.GGMLLanguage("unused.gguf", compiled_bridge)
    yield instance
    instance.close()


def append_root(language):
    assert language.append([10, 11], np.ones((2, 3), np.float32)) == 4


def descriptor(features=None, **changes):
    from edgeflow_split.surface_backed_descriptor import SurfaceBackedVisualDescriptor

    features = np.ones((2, 3), np.float32) if features is None else features
    raw = Lease(features.ctypes.data, features.nbytes, 0, 0)
    released = []
    lease = SimpleNamespace(lease_handle=ctypes.addressof(raw), release=lambda: released.append(True))
    value = SurfaceBackedVisualDescriptor(
        descriptor_id=1, surface_lease=lease, token_count=2, hidden_size=3,
        dtype="float32", image_grid_thw=(1, 1, 2), frame_id=0, timestamp_ms=0,
        handoff_receipt={"producer_pointer": features.ctypes.data, "surface_id": 42},
    )
    for name, change in changes.items():
        setattr(value, name, change)
    return value, raw, features, released


def test_legacy_init_keeps_4096_and_single_model(native):
    context = native.fv_init(b"unused", b"unused")
    try:
        assert context
        assert native.fv_test_get(b"capacity") == 4096
        assert native.fv_test_get(b"model_loads") == 1
    finally:
        native.fv_free(context)


def test_greedy_chain_is_owned_once_and_accepts_each_token_once(language, native):
    assert native.fv_test_get(b"chain_inits") == 1
    assert native.fv_test_get(b"chain_adds") == 1
    append_root(language)
    output, _ = language.query([9], 4)
    assert output == [1, 2, 3, 3]
    assert native.fv_test_get(b"accept_calls") == 4


def test_configurable_context_and_model_limit(native, compiled_bridge):
    instance = module.GGMLLanguage("unused", compiled_bridge, n_ctx=32768)
    try:
        assert instance.n_ctx == 32768
        assert instance.vocab_size == 4
        assert native.fv_test_get(b"capacity") == 32768
        assert native.fv_test_get(b"model_loads") == 1
    finally:
        instance.close()
    native.fv_test_set(b"model_limit", 8192)
    with pytest.raises(RuntimeError, match="model.*context"):
        module.GGMLLanguage("unused", compiled_bridge, n_ctx=8193)


@pytest.mark.parametrize("n_ctx", [0, -1, 32769, 2**40, 1.5])
def test_invalid_context_is_rejected_before_model_load(native, compiled_bridge, n_ctx):
    with pytest.raises((ValueError, TypeError), match="n_ctx|integer"):
        module.GGMLLanguage("unused", compiled_bridge, n_ctx=n_ctx)
    assert native.fv_test_get(b"model_loads") == 0


@pytest.mark.parametrize("failure", [b"fail_context", b"fail_sampler"])
def test_initialization_failure_frees_all_owned_resources(native, compiled_bridge, failure):
    native.fv_test_set(failure, 1)
    with pytest.raises(RuntimeError):
        module.GGMLLanguage("unused", compiled_bridge)


@pytest.mark.parametrize("surface", [False, True])
def test_combined_append_preflight_does_not_mutate_kv(language, native, surface):
    features = np.ones((2, 3), np.float32)
    keepalive = descriptor(features)
    with pytest.raises(RuntimeError, match="context"):
        language.append(np.ones(4095, np.int32), keepalive[0] if surface else features)
    assert language.position == native.fv_test_get(b"kv_end") == 0
    assert native.fv_test_get(b"decode_calls") == 0
    assert native.fv_test_get(b"surface_calls") == 0
    assert native.fv_test_get(b"cleanup_calls") == 0


def test_query_preflight_reserves_full_budget_without_kv_mutation(language, native):
    append_root(language)
    before = native.fv_test_get(b"decode_calls")
    with pytest.raises(RuntimeError, match="context"):
        language.query([9], 4092)
    assert language.position == native.fv_test_get(b"kv_end") == 4
    assert native.fv_test_get(b"decode_calls") == before
    assert native.fv_test_get(b"cleanup_calls") == 0
    assert language.last_query["actual_count"] == 0
    assert language.last_query["cleanup_ok"] is True


def test_requested_capacity_not_backend_rounding_is_the_limit(native, compiled_bridge):
    native.fv_test_set(b"round_context", 256)
    instance = module.GGMLLanguage("unused", compiled_bridge, n_ctx=8)
    try:
        assert instance.append([1], np.ones((7, 3), np.float32)) == 8
        with pytest.raises(RuntimeError, match="context"):
            instance.query([1], 1)
    finally:
        instance.close()


def test_exact_query_capacity_and_legacy_fixed_budget(language, native):
    append_root(language)
    before = native.fv_test_get(b"decode_calls")
    output, stamps = language.query(np.ones(4088, np.int32), 4)
    assert output == [1, 2, 3, 3]  # EOG is included and ignored by default.
    assert len(stamps) == 3 and stamps[0] <= stamps[1] <= stamps[2]
    assert language.position == native.fv_test_get(b"kv_end") == 4
    assert native.fv_test_get(b"decode_calls") - before == 8 + 3
    assert native.fv_test_get(b"logits_reads") == 0
    assert language.last_query["root"] == 4
    assert language.last_query["actual_count"] == 4
    assert language.last_query["cleanup_ok"] is True
    assert "first_logits" not in language.last_query


def test_first_logits_are_copied_once_before_sampling_and_eog_stops(language, native):
    append_root(language)
    before = time.monotonic()
    output, stamps = language.query([9], 4, capture_first_logits=True, stop_at_eog=True)
    assert output == [1, 2]
    info = language.last_query
    assert before <= info["prefill_finished_s"] <= stamps[0] <= stamps[1] <= stamps[2]
    assert info["actual_count"] == 2 and info["root"] == 4 and info["cleanup_ok"] is True
    assert info["first_logits"].dtype == np.float32
    np.testing.assert_array_equal(info["first_logits"], [0.25, 1.5, -2, 3.75])
    assert native.fv_test_get(b"logits_reads") == 1
    assert native.fv_test_get(b"decode_calls") == 4  # 2 append, prefill, one decode.
    language.query([9], 1)
    assert "first_logits" not in language.last_query
    np.testing.assert_array_equal(info["first_logits"], [0.25, 1.5, -2, 3.75])


def test_production_descriptor_aliases_without_materializing_or_releasing(language):
    value, raw, storage, released = descriptor()
    value.materialize_to_batch = lambda: pytest.fail("surface was materialized")
    assert language.append([10], value) == 3
    receipt = language.last_handoff
    assert receipt["producer_pointer"] == receipt["consumer_pointer"] == storage.ctypes.data
    assert receipt["producer_consumer_alias"] is True
    assert receipt["application_embedding_copy_bytes"] == 0
    assert receipt["surface_id"] == 42
    assert released == [] and not value.released
    assert receipt["text_ingest_started_s"] <= receipt["text_ingest_finished_s"]
    assert receipt["text_ingest_finished_s"] <= receipt["visual_ingest_started_s"]
    assert receipt["visual_ingest_started_s"] <= receipt["visual_ingest_finished_s"]
    assert receipt["handoff_s"] == pytest.approx(receipt["surface_setup_s"] + receipt["surface_teardown_s"])
    assert receipt["visual_ingest_s"] >= 0.001


def test_early_label_is_unpublished_until_visual_append_and_keeps_alias(language, native):
    append_root(language)
    value, raw, storage, released = descriptor()
    language.begin_append([10, 11], visual_count=2)
    assert language.position == 4
    assert native.fv_test_get(b"kv_end") == 6
    with pytest.raises(RuntimeError, match="pending|unfinished"):
        language.query([9], 1)
    assert language.append(None, value) == 8
    assert native.fv_test_get(b"kv_end") == 8
    timing = language.last_handoff
    assert timing["text_ingest_finished_s"] <= timing["visual_ingest_started_s"]
    assert timing["producer_consumer_alias"] and released == []
    assert timing["application_embedding_copy_bytes"] == 0
    language.query([9], 1)
    assert native.fv_test_get(b"kv_end") == 8


def test_early_label_reserves_whole_frame_before_mutating_kv(language, native):
    append_root(language)
    before = native.fv_test_get(b"decode_calls")
    with pytest.raises(RuntimeError, match="context"):
        language.begin_append([1], visual_count=4092)
    assert language.position == native.fv_test_get(b"kv_end") == 4
    assert native.fv_test_get(b"decode_calls") == before


def test_early_label_can_be_cancelled_when_producer_fails(language, native):
    append_root(language)
    language.begin_append([10], visual_count=2)
    language.cancel_append()
    language.cancel_append()
    assert language.position == native.fv_test_get(b"kv_end") == 4
    assert language.last_handoff is None
    assert language.append([10, 11], np.ones((2, 3), np.float32)) == 8


@pytest.mark.parametrize("failure", ["visual_decode", "surface_lock", "wrong_count"])
def test_staged_visual_failure_restores_root_before_label(language, native, failure):
    append_root(language)
    value, raw, storage, released = descriptor()
    language.begin_append([10], visual_count=2)
    if failure == "visual_decode":
        native.fv_test_set(b"fail_decode", 4)
    elif failure == "surface_lock":
        raw.fail_before = 1
    else:
        value.token_count = 1
    with pytest.raises(RuntimeError):
        language.append(None, value)
    assert language.position == native.fv_test_get(b"kv_end") == 4
    assert native.fv_test_get(b"pending") == 0
    assert language.last_handoff is None


def test_legacy_surface_and_numpy_still_alias(language):
    spec = importlib.util.spec_from_file_location("coreml_contract", ROOT / "tools/mac_fastvlm_coreml_native.py")
    coreml = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coreml)
    value, raw, storage, released = descriptor()
    legacy = coreml.NativeCoreMLFeatures(ctypes.addressof(raw), storage.ctypes.data,
                                       (1, 2, 3), 42, {"producer_pointer": storage.ctypes.data}, value)
    assert language.append([10], legacy) == 3
    assert language.last_handoff["application_embedding_copy_bytes"] == 0
    assert language.append([10], storage) == 6
    assert language.last_handoff["consumer_pointer"] == storage.ctypes.data
    assert language.last_handoff["surface_setup_s"] == 0
    assert language.last_handoff["surface_teardown_s"] == 0


@pytest.mark.parametrize("changes,match", [
    ({"dtype": "float16"}, "float32"),
    ({"text_prefix_tokens": 1}, "packed"),
    ({"surface_lease": None}, "lease"),
    ({"handoff_receipt": {}}, "producer_pointer"),
])
def test_invalid_descriptor_fails_before_append(language, native, changes, match):
    keepalive = descriptor(**changes)
    with pytest.raises((ValueError, RuntimeError), match=match):
        language.append([10], keepalive[0])
    assert native.fv_test_get(b"decode_calls") == 0


def test_alias_mismatch_is_not_reported_as_zero_copy_success(language):
    value, raw, storage, released = descriptor(handoff_receipt={"producer_pointer": 1})
    with pytest.raises(RuntimeError, match="alias"):
        language.append([10], value)
    assert language.position == 3
    assert language.last_handoff["producer_consumer_alias"] is False


@pytest.mark.parametrize("failure", ["decode", "short", "null", "lock", "unlock"])
def test_surface_failure_rolls_back_text_and_visual_suffix(language, native, failure):
    append_root(language)
    value, raw, storage, released = descriptor()
    if failure == "decode":
        native.fv_test_set(b"fail_decode", 4)
    elif failure == "short":
        raw.nbytes = 1
    elif failure == "null":
        raw.base = None
    elif failure == "lock":
        raw.fail_before = 1
    else:
        raw.fail_after = 1
    with pytest.raises(RuntimeError):
        language.append([10], value)
    assert language.position == native.fv_test_get(b"kv_end") == 4
    assert native.fv_test_get(b"pending") == 0
    assert language.last_handoff is None
    assert released == []


@pytest.mark.parametrize("failure", ["prefill", "decode", "logits"])
def test_query_failure_restores_retained_root_and_reports_progress(language, native, failure):
    append_root(language)
    native.fv_test_set(b"null_logits" if failure == "logits" else b"fail_decode",
                       1 if failure == "logits" else 3 if failure == "prefill" else 4)
    with pytest.raises(RuntimeError):
        language.query([9], 4, capture_first_logits=True)
    assert language.position == native.fv_test_get(b"kv_end") == 4
    assert language.last_query["cleanup_ok"] is True
    assert language.last_query["root"] == 4
    assert language.last_query["actual_count"] == (1 if failure == "decode" else 0)
    native.fv_test_set(b"null_logits", 0)
    output, _ = language.query([9], 1)
    assert output == [1]


def test_cleanup_failure_requires_reset_and_does_not_claim_root_restored(language, native):
    append_root(language)
    native.fv_test_set(b"fail_cleanup", 1)
    with pytest.raises(RuntimeError, match="suffix|cleanup"):
        language.query([9], 2)
    assert language.last_query["cleanup_ok"] is False
    assert native.fv_test_get(b"kv_end") > 4
    before = native.fv_test_get(b"decode_calls")
    with pytest.raises(RuntimeError, match="reset"):
        language.query([9], 1)
    with pytest.raises(RuntimeError, match="reset"):
        append_root(language)
    assert native.fv_test_get(b"decode_calls") == before
    native.fv_test_set(b"fail_cleanup", 0)
    language.reset()
    assert language.position == native.fv_test_get(b"kv_end") == 0
    assert language.last_query is None and language.last_handoff is None
    append_root(language)


@pytest.mark.parametrize("budget", [0, -1, 2**40, 1.5])
def test_invalid_query_budget_does_not_allocate_or_mutate(language, native, budget):
    with pytest.raises((ValueError, TypeError), match="budget|integer"):
        language.query([1], budget)
    assert native.fv_test_get(b"decode_calls") == 0


def test_closed_language_rejects_operations(language):
    language.close()
    language.close()
    for operation in (language.reset, lambda: language.query([1], 1),
                      lambda: language.append([1], np.ones((1, 3), np.float32))):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


def test_legacy_query_c_abi_keeps_three_timestamps_and_fixed_budget(language, native):
    append_root(language)
    tokens = (ctypes.c_int32 * 1)(9)
    output = (ctypes.c_int32 * 4)()
    stamps = (ctypes.c_double * 3)()
    assert language.lib.fv_query(language.ctx, tokens, 1, 4, output, stamps) == 4
    assert list(output) == [1, 2, 3, 3]
    assert stamps[0] <= stamps[1] <= stamps[2]
    assert native.fv_test_get(b"decode_calls") == 6
    assert native.fv_test_get(b"logits_reads") == 0
    assert native.fv_test_get(b"kv_end") == 4


@pytest.mark.parametrize("capacity", [0, -1, 32769, 2**31 - 1])
def test_native_init_also_rejects_invalid_capacity_before_load(native, capacity):
    native.fv_init_with_context.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    native.fv_init_with_context.restype = ctypes.c_void_p
    assert not native.fv_init_with_context(b"unused", b"unused", capacity)
    assert b"n_ctx" in native.fv_error()
    assert native.fv_test_get(b"model_loads") == 0


@pytest.mark.parametrize("case", ["context", "tokens", "features", "empty_text",
                                  "empty_visual", "width", "overflow"])
def test_native_append_validation_is_before_kv_mutation(language, native, case):
    tokens = (ctypes.c_int32 * 1)(9)
    features = (ctypes.c_float * 3)(1, 2, 3)
    args = [language.ctx, tokens, 1, features, 1, 3]
    if case in ("context", "tokens", "features"):
        args[{"context": 0, "tokens": 1, "features": 3}[case]] = None
    elif case == "empty_text":
        args[2] = 0
    elif case == "empty_visual":
        args[4] = -1
    elif case == "width":
        args[5] = 4
    else:
        args[2] = args[4] = 2**31 - 1
    assert language.lib.fv_append(*args) == -1
    assert native.fv_test_get(b"decode_calls") == 0
    assert native.fv_test_get(b"cleanup_calls") == 0


@pytest.mark.parametrize("case", ["context", "tokens", "empty_text", "budget",
                                  "output", "times", "info", "short_logits",
                                  "missing_logits", "negative_logits", "stop", "overflow"])
def test_native_query_validation_checks_buffers_and_capacity(language, native, case):
    # Bind independently of the wrapper to test the exact additive ABI layout.
    native.fv_query_ex.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                  ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
                                  ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_float),
                                  ctypes.c_int, ctypes.c_int, ctypes.POINTER(QueryInfo)]
    native.fv_query_ex.restype = ctypes.c_int
    tokens = (ctypes.c_int32 * 1)(9)
    output = (ctypes.c_int32 * 2)(-42, -42)
    stamps = (ctypes.c_double * 3)()
    logits = (ctypes.c_float * 4)(-42, -42, -42, -42)
    info = QueryInfo()
    args = [language.ctx, tokens, 1, 2, output, stamps, logits, 4, 0, ctypes.byref(info)]
    if case in ("context", "tokens", "output", "times", "info", "missing_logits"):
        args[{"context": 0, "tokens": 1, "output": 4, "times": 5,
              "info": 9, "missing_logits": 6}[case]] = None
    elif case in ("empty_text", "budget"):
        args[2 if case == "empty_text" else 3] = 0
    elif case in ("short_logits", "negative_logits"):
        args[7] = 3 if case == "short_logits" else -1
    elif case == "stop":
        args[8] = 2
    else:
        args[2] = args[3] = 2**31 - 1
    assert native.fv_query_ex(*args) == -1
    assert native.fv_test_get(b"decode_calls") == native.fv_test_get(b"cleanup_calls") == 0
    assert list(output) == [-42, -42] and list(logits) == [-42] * 4
    assert info.actual_count == 0 and info.logits_captured == 0
    if case not in ("context", "info"):
        assert info.root == 0 and info.cleanup_ok == 1
        assert np.isnan(info.prefill_finished_s)


@pytest.mark.parametrize("query", [False, True])
def test_failure_after_one_complete_prefill_batch_restores_root(language, native, query):
    append_root(language)
    native.fv_test_set(b"fail_decode", 4)  # Second batch in a 513-token prefill.
    with pytest.raises(RuntimeError, match="llama_decode"):
        if query:
            language.query(np.ones(513, np.int32), 1)
        else:
            language.append(np.ones(513, np.int32), np.ones((2, 3), np.float32))
    assert language.position == native.fv_test_get(b"kv_end") == 4
    assert native.fv_test_get(b"cleanup_calls") == 1
    assert native.fv_test_get(b"pending") == 0


@pytest.mark.parametrize("query", [False, True])
def test_decode_and_cleanup_failure_keep_original_error_and_poison_context(language, native, query):
    append_root(language)
    native.fv_test_set(b"fail_decode", 3)
    native.fv_test_set(b"fail_cleanup", 1)
    with pytest.raises(RuntimeError, match="llama_decode failed;.*reset required"):
        if query:
            language.query([9], 2)
        else:
            language.append([9], np.ones((2, 3), np.float32))
    assert native.fv_test_get(b"cleanup_calls") == 1
    if query:
        assert language.last_query["cleanup_ok"] is False
        assert language.last_query["actual_count"] == 0
        assert language.last_query["prefill_finished_s"] is None
    with pytest.raises(RuntimeError, match="reset required"):
        language.query([9], 1)
    assert native.fv_test_get(b"decode_calls") == 3
    assert native.fv_test_get(b"pending") == 0


@pytest.mark.parametrize("capture,stop,expected,reads", [
    (True, False, [1, 2, 3], 1), (False, True, [1, 2], 0),
])
def test_query_options_are_independent(language, native, capture, stop, expected, reads):
    output, stamps = language.query([9], 3, capture_first_logits=capture, stop_at_eog=stop)
    assert output == expected
    assert native.fv_test_get(b"logits_reads") == reads
    assert native.fv_test_get(b"kv_end") == 0
    assert ("first_logits" in language.last_query) is capture


def test_budget_one_does_not_decode_unused_next_step(language, native):
    output, stamps = language.query([9], 1, capture_first_logits=True, stop_at_eog=True)
    assert output == [1] and stamps[0] == stamps[1]
    assert native.fv_test_get(b"decode_calls") == 1
    assert language.last_query["actual_count"] == 1


def test_released_descriptor_rejected_without_native_work(language, native):
    value, raw, storage, released = descriptor()
    value.release()
    with pytest.raises(RuntimeError, match="lease"):
        language.append([9], value)
    assert native.fv_test_get(b"decode_calls") == 0
    assert released == [True]


def test_failed_capture_never_exposes_uninitialized_logits(language, native):
    native.fv_test_set(b"null_logits", 1)
    with pytest.raises(RuntimeError, match="logits"):
        language.query([9], 1, capture_first_logits=True)
    assert "first_logits" not in language.last_query
    assert language.last_query["prefill_finished_s"] is not None
    assert language.last_query["cleanup_ok"] is True


class LegacyLibrary:
    """Hide only additive symbols; calls still execute the compiled bridge/stubs."""
    def __init__(self, library):
        self.library = library

    def __getattr__(self, name):
        if name in {"fv_init_with_context", "fv_query_ex", "fv_vocab_size", "fv_last_handoff"}:
            raise AttributeError(name)
        return getattr(self.library, name)


@pytest.fixture
def legacy_library(native, monkeypatch):
    library = LegacyLibrary(native)
    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: library)
    return library


def test_archived_runtime_keeps_default_init_append_and_fixed_budget(legacy_library, native):
    instance = module.GGMLLanguage("unused", "archived.dylib")
    try:
        assert instance.n_ctx == native.fv_test_get(b"capacity") == 4096
        assert instance.vocab_size is None
        value, raw, storage, released = descriptor()
        assert instance.append([9], value) == 3
        assert instance.last_handoff["consumer_pointer"] == storage.ctypes.data
        assert instance.last_handoff["application_embedding_copy_bytes"] == 0
        for field in ("handoff_s", "surface_setup_s", "surface_teardown_s",
                      "text_ingest_started_s", "text_ingest_finished_s",
                      "visual_ingest_started_s", "visual_ingest_finished_s",
                      "text_ingest_s", "visual_ingest_s"):
            assert instance.last_handoff[field] is None
        output, stamps = instance.query([9], 4)
        assert output == [1, 2, 3, 3] and len(stamps) == 3
        assert instance.last_query == {"prefill_finished_s": None, "actual_count": 4,
                                      "root": 3, "cleanup_ok": True}
        assert instance.position == native.fv_test_get(b"kv_end") == 3
        assert native.fv_test_get(b"model_loads") == 1
    finally:
        instance.close()


def test_archived_runtime_rejects_larger_context_before_init(legacy_library, native):
    with pytest.raises(RuntimeError, match="n_ctx.*rebuild|rebuild.*n_ctx"):
        module.GGMLLanguage("unused", "archived.dylib", n_ctx=32768)
    assert native.fv_test_get(b"model_loads") == 0


@pytest.mark.parametrize("option", ["capture_first_logits", "stop_at_eog"])
def test_archived_runtime_rejects_new_query_options_before_mutation(legacy_library, native, option):
    instance = module.GGMLLanguage("unused", "archived.dylib")
    try:
        with pytest.raises(RuntimeError, match="rebuild"):
            instance.query([9], 4, **{option: True})
        assert native.fv_test_get(b"decode_calls") == 0
    finally:
        instance.close()


def test_native_timestamps_use_the_python_monotonic_epoch():
    configured = os.environ.get("CROSSINFER_FASTVLM_LIBRARY")
    if not configured:
        pytest.skip("set CROSSINFER_FASTVLM_LIBRARY to check a locally built bridge")
    library = Path(configured).expanduser()
    if not library.is_file():
        pytest.skip("configured bridge is not built")
    native = ctypes.CDLL(str(library))
    native.fv_now.restype = ctypes.c_double
    before = time.monotonic()
    measured = native.fv_now()
    after = time.monotonic()
    assert before <= measured <= after

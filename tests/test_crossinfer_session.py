from __future__ import annotations

import asyncio
import ctypes
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from edgeflow_split import crossinfer_session as cs
from edgeflow_split.model_spec import ModelFamily as SharedModelFamily


class _FakeCFunc:
    def __init__(self, fn):
        self.fn = fn
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.fn(*args)


def test_session_reuses_the_shared_model_family_authority() -> None:
    assert cs.ModelFamily is SharedModelFamily


class _FakeLibrary:
    def __init__(self) -> None:
        self.error = b"fake failure"
        self.free_count = 0
        self.remove_status = 0
        self.sequence = {0: {"active": True, "position": 0, "logits": False}}
        self.params = None
        self.edgeflow_llm_abi_version = _FakeCFunc(lambda: cs.LLM_ABI_VERSION)
        self.edgeflow_llm_request_sizeof = _FakeCFunc(
            lambda: ctypes.sizeof(cs._QwenRequest)
        )
        self.edgeflow_llm_session_abi_version = _FakeCFunc(
            lambda: cs.SESSION_ABI_VERSION
        )
        self.edgeflow_llm_session_params_sizeof = _FakeCFunc(
            lambda: ctypes.sizeof(cs._SessionParams)
        )
        self.edgeflow_llm_session_signature_sizeof = _FakeCFunc(
            lambda: ctypes.sizeof(cs._SessionSignature)
        )
        self.edgeflow_llm_sequence_info_sizeof = _FakeCFunc(
            lambda: ctypes.sizeof(cs._SequenceInfo)
        )
        self.edgeflow_llm_decode_step_result_sizeof = _FakeCFunc(
            lambda: ctypes.sizeof(cs._DecodeStepResult)
        )
        self.edgeflow_llm_session_init = _FakeCFunc(self._init)
        self.edgeflow_llm_session_free = _FakeCFunc(self._free)
        self.edgeflow_llm_session_last_init_error = _FakeCFunc(lambda: self.error)
        self.edgeflow_llm_session_last_error = _FakeCFunc(lambda _ctx: self.error)
        self.edgeflow_llm_session_get_signature = _FakeCFunc(self._get_signature)
        self.edgeflow_llm_session_vocab_size = _FakeCFunc(lambda _ctx: 8)
        self.edgeflow_llm_session_get_sequence_info = _FakeCFunc(
            self._get_sequence_info
        )
        self.edgeflow_llm_session_ingest_text = _FakeCFunc(self._ingest_text)
        self.edgeflow_llm_session_ingest_user_prompt = _FakeCFunc(
            self._ingest_user_prompt
        )
        self.edgeflow_llm_session_ingest_tokens = _FakeCFunc(self._ingest_tokens)
        self.edgeflow_llm_session_ingest_qwen25vl = _FakeCFunc(self._ingest_qwen)
        self.edgeflow_llm_session_ingest_qwen25vl_visual_prefix = _FakeCFunc(
            self._ingest_qwen_visual_prefix
        )
        self.edgeflow_llm_session_ingest_qwen25vl_query = _FakeCFunc(
            self._ingest_qwen_query
        )
        self.edgeflow_llm_session_fork_prefix = _FakeCFunc(self._fork)
        self.edgeflow_llm_session_decode_step = _FakeCFunc(self._decode)
        self.edgeflow_llm_session_token_piece = _FakeCFunc(self._token_piece)
        self.edgeflow_llm_session_remove_sequence = _FakeCFunc(self._remove)

    def _init(self, _model_path, params_ptr):
        self.params = params_ptr._obj
        return 1234

    def _free(self, _ctx) -> None:
        self.free_count += 1

    def _get_signature(self, _ctx, out_ptr) -> int:
        out = out_ptr._obj
        params = self.params
        out.session_abi_version = cs.SESSION_ABI_VERSION
        out.requested_gpu_layers = params.n_gpu_layers
        out.observed_gpu_layers = 37
        out.backend_placement = int(cs.BackendPlacement.GPU_FULL)
        for field in (
            "n_ctx",
            "n_batch",
            "n_ubatch",
            "n_seq_max",
            "n_threads",
            "kv_unified",
            "model_family",
            "kv_type_k",
            "kv_type_v",
            "position_contract",
            "prompt_contract_version",
        ):
            setattr(out, field, getattr(params, field))
        out.model_sha256 = b"a" * 64
        out.runtime_sha256 = b"b" * 64
        out.runtime_source_fingerprint_sha256 = b"c" * 64
        return 0

    def _get_sequence_info(self, _ctx, sequence_id, out_ptr) -> int:
        state = self.sequence.get(
            sequence_id, {"active": False, "position": 0, "logits": False}
        )
        out = out_ptr._obj
        out.active = int(state["active"])
        out.next_position = state["position"]
        out.has_logits = int(state["logits"])
        out.state_bytes = 128 if state["active"] else 0
        return 0

    def _ingest_text(
        self, _ctx, sequence_id, _text, _add_special, _parse_special, count_ptr
    ) -> int:
        self.sequence[sequence_id]["position"] += 3
        self.sequence[sequence_id]["logits"] = True
        count_ptr._obj.value = 3
        return 0

    def _ingest_user_prompt(self, ctx, sequence_id, text, count_ptr) -> int:
        return self._ingest_text(ctx, sequence_id, text, 1, 1, count_ptr)

    def _ingest_tokens(self, _ctx, sequence_id, _tokens, count) -> int:
        self.sequence[sequence_id]["position"] += count
        self.sequence[sequence_id]["logits"] = True
        return 0

    def _ingest_qwen(self, _ctx, sequence_id, request_ptr) -> int:
        request = request_ptr._obj
        assert request.num_visual_tokens == 196
        assert request.hidden_size == 8
        self.sequence[sequence_id]["position"] = 23
        self.sequence[sequence_id]["logits"] = True
        return 0

    def _ingest_qwen_visual_prefix(self, _ctx, sequence_id, request_ptr) -> int:
        request = request_ptr._obj
        assert request.prompt is None
        assert request.num_visual_tokens == 196
        assert request.hidden_size == 8
        self.sequence[sequence_id]["position"] = 17
        self.sequence[sequence_id]["logits"] = False
        self.sequence[sequence_id]["visual_boundary"] = 17
        return 0

    def _ingest_qwen_query(self, _ctx, sequence_id, prompt, count_ptr) -> int:
        assert prompt
        state = self.sequence[sequence_id]
        assert state["position"] == state["visual_boundary"]
        state["position"] += 6
        state["logits"] = True
        count_ptr._obj.value = 6
        return 0

    def _fork(self, _ctx, source, destination, end_position) -> int:
        assert self.sequence[source]["active"]
        self.sequence[destination] = {
            "active": True,
            "position": end_position,
            "logits": end_position == self.sequence[source]["position"],
        }
        if self.sequence[source].get("visual_boundary") == end_position:
            self.sequence[destination]["visual_boundary"] = end_position
        return 0

    def _decode(self, _ctx, sequence_id, out_ptr, logits, capacity) -> int:
        state = self.sequence[sequence_id]
        out = out_ptr._obj
        out.token_id = 5
        out.token_position = state["position"]
        out.is_eog = 0
        out.next_token_id = 6
        out.n_vocab = 8
        state["position"] += 1
        state["logits"] = True
        if logits is not None:
            assert capacity == 8
            for index in range(capacity):
                logits[index] = float(index)
        return 0

    def _token_piece(self, _ctx, token_id, out, capacity, size_ptr) -> int:
        piece = f"token-{token_id}".encode()
        size_ptr._obj.value = len(piece)
        if out is None and capacity == 0:
            return 0
        if capacity < len(piece):
            return 5
        ctypes.memmove(out, piece, len(piece))
        return 0

    def _remove(self, _ctx, sequence_id) -> int:
        if self.remove_status != 0:
            return self.remove_status
        self.sequence[sequence_id] = {"active": False, "position": 0, "logits": False}
        return 0


@pytest.fixture
def fake_session(monkeypatch, tmp_path: Path):
    model = tmp_path / "model.gguf"
    library = tmp_path / "libedgeflow_llm.dylib"
    model.write_bytes(b"model")
    library.write_bytes(b"library")
    fake = _FakeLibrary()
    monkeypatch.setattr(cs.ctypes, "CDLL", lambda _path: fake)
    monkeypatch.setattr(
        cs,
        "_sha256_file",
        lambda path: "a" * 64 if path == model.resolve() else "b" * 64,
    )
    session = cs.StatefulLLMSession(
        cs.SessionConfig(model_path=model, library_path=library)
    )
    try:
        yield session, fake
    finally:
        session.close()


@pytest.mark.parametrize("size", [0, -2])
def test_session_rejects_nonpositive_native_vocabulary(fake_session, size) -> None:
    session, fake = fake_session
    fake.edgeflow_llm_session_vocab_size.fn = lambda _ctx: size
    with pytest.raises(RuntimeError, match="vocabulary size"):
        session._read_vocab_size()


def test_session_binds_complete_compatibility_signature(fake_session) -> None:
    session, fake = fake_session

    assert session.signature == cs.StateCompatibilitySignature(
        session_abi_version=3,
        model_sha256="a" * 64,
        runtime_sha256="b" * 64,
        runtime_source_fingerprint_sha256="c" * 64,
        requested_gpu_layers=999,
        observed_gpu_layers=37,
        backend_placement=cs.BackendPlacement.GPU_FULL,
        n_ctx=4096,
        n_batch=512,
        n_ubatch=512,
        n_seq_max=4,
        n_threads=4,
        kv_unified=True,
        model_family=cs.ModelFamily.QWEN25VL,
        kv_type_k=cs.KVDataType.FP16,
        kv_type_v=cs.KVDataType.FP16,
        position_contract=cs.PositionContract.TEXT_1D,
        prompt_contract_version=1,
    )
    assert fake.params.n_gpu_layers == 999


def test_ingest_fork_decode_and_remove_lifecycle(fake_session) -> None:
    session, _fake = fake_session

    assert session.ingest_text(0, "prefix", add_special=True) == 3
    assert session.sequence_info(0).next_position == 3
    session.fork_prefix(0, 1, 3)
    session.ingest_tokens(1, [1, 2])
    step = session.decode_step(1, capture_logits=True)

    assert step.token_id == 5
    assert step.token_position == 5
    assert step.next_token_id == 6
    assert isinstance(step.logits, np.ndarray)
    assert step.logits.dtype == np.float32 and step.logits.flags.owndata
    assert not step.logits.flags.writeable
    np.testing.assert_array_equal(step.logits, np.arange(8, dtype=np.float32))
    assert session.sequence_info(1).next_position == 6
    session.remove_sequence(1)
    assert not session.sequence_info(1).active


def test_captured_logits_do_not_alias_the_native_output_buffer(fake_session) -> None:
    session, fake = fake_session
    buffers = []

    def capture_buffer(*args):
        buffers.append(args[3])
        return fake._decode(*args)

    fake.edgeflow_llm_session_decode_step.fn = capture_buffer
    session.ingest_text(0, "prefix")
    first = session.decode_step(0, capture_logits=True).logits
    buffers[0][0] = 123.0
    assert first[0] == 0.0
    assert session.decode_step(0).logits is None


def test_user_prompt_ingest_and_token_piece_use_native_contract(fake_session) -> None:
    session, _fake = fake_session

    assert session.ingest_user_prompt(0, "Describe the image briefly.") == 3
    assert session.token_piece_bytes(5) == b"token-5"


def test_token_piece_decoder_preserves_utf8_split_across_tokens() -> None:
    decoder = cs.TokenPieceStreamDecoder()

    assert decoder.feed(b"prefix ") == "prefix "
    assert decoder.feed(b"\xe4") == ""
    assert decoder.feed(b"\xb8") == ""
    assert decoder.feed(b"\xad suffix") == "\u4e2d suffix"
    assert decoder.finish() == ""


def test_token_piece_decoders_isolate_interleaved_response_bytes() -> None:
    left = cs.TokenPieceStreamDecoder()
    right = cs.TokenPieceStreamDecoder()

    assert left.feed(b"\xe4") == ""
    assert right.feed(b"\xe6") == ""
    assert left.feed(b"\xb8") == ""
    assert right.feed(b"\xb0") == ""
    assert left.feed(b"\xad") == "\u4e2d"
    assert right.feed(b"\xb4") == "\u6c34"
    assert left.finish() == ""
    assert right.finish() == ""


def test_session_incarnation_binds_exactly_one_stream(fake_session) -> None:
    session, _fake = fake_session

    assert session.session_incarnation_id
    session.bind_stream("camera-1")
    session.bind_stream("camera-1")
    with pytest.raises(RuntimeError, match="already bound to stream camera-1"):
        session.bind_stream("camera-2")


def test_completed_qwen_image_ingest_uses_explicit_grid(fake_session) -> None:
    np = pytest.importorskip("numpy")
    session, _fake = fake_session
    visual = np.zeros((196, 8), dtype=np.float32)

    session.ingest_qwen25vl(
        0,
        prompt="Describe the image.",
        visual_embeddings=visual,
        image_grid_thw=(1, 28, 28),
    )

    assert session.sequence_info(0).next_position == 23
    assert session.sequence_info(0).has_logits


def test_qwen_visual_prefix_can_fork_two_distinct_queries(fake_session) -> None:
    np = pytest.importorskip("numpy")
    session, _fake = fake_session
    visual = np.zeros((196, 8), dtype=np.float32)

    session.ingest_qwen25vl_visual_prefix(
        0,
        visual_embeddings=visual,
        image_grid_thw=(1, 28, 28),
    )
    prefix = session.sequence_info(0)
    assert prefix.next_position == 17
    assert not prefix.has_logits

    session.fork_prefix(0, 1, prefix.next_position)
    session.fork_prefix(0, 2, prefix.next_position)
    assert session.ingest_qwen25vl_query(1, "What color is most prominent?") == 6
    assert session.ingest_qwen25vl_query(2, "How many people are visible?") == 6

    assert session.sequence_info(1).next_position == 23
    assert session.sequence_info(2).next_position == 23
    assert session.sequence_info(1).has_logits
    assert session.sequence_info(2).has_logits


def test_qwen_query_rejects_embedded_nul_before_native_call(fake_session) -> None:
    session, _fake = fake_session

    with pytest.raises(ValueError, match="NUL"):
        session.ingest_qwen25vl_query(0, "question\0ignored")


def test_visual_append_is_optional_and_has_explicit_abi_binding(fake_session):
    np = pytest.importorskip("numpy")
    session, fake = fake_session
    args = {"visual_embeddings": np.ones((1, 8)), "image_grid_thw": (1, 2, 2)}
    with pytest.raises(NotImplementedError, match="append"):
        session.append_qwen25vl_visual(0, **args)
    calls = []
    fake.edgeflow_llm_session_append_qwen25vl_visual = _FakeCFunc(
        lambda ctx, seq, req: calls.append((ctx, seq, req._obj)) or 0
    )
    cs._bind_session_library(fake)
    session.append_qwen25vl_visual(0, **args)
    assert calls[0][0].value == 1234
    assert calls[0][1] == 0
    assert calls[0][2].num_visual_tokens == 1
    assert tuple(calls[0][2].image_grid_thw) == (1, 2, 2)
    assert fake.edgeflow_llm_session_append_qwen25vl_visual.argtypes == [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(cs._QwenRequest),
    ]


@pytest.mark.parametrize("grid", [(1, 3, 2), (2, 2, 2), (1, 2.5, 2), (1, 2**33, 2)])
def test_visual_append_rejects_invalid_grid_before_native(fake_session, grid):
    np = pytest.importorskip("numpy")
    session, fake = fake_session
    fake.edgeflow_llm_session_append_qwen25vl_visual = _FakeCFunc(
        lambda *args: pytest.fail("invalid input reached native")
    )
    with pytest.raises(ValueError):
        session.append_qwen25vl_visual(
            0, visual_embeddings=np.ones((1, 8)), image_grid_thw=grid
        )


def test_visual_append_rejects_nonfinite_and_mismatched_grid(fake_session):
    np = pytest.importorskip("numpy")
    session, fake = fake_session
    fake.edgeflow_llm_session_append_qwen25vl_visual = _FakeCFunc(
        lambda *args: pytest.fail("invalid input reached native")
    )
    for visual, grid in [
        (np.full((1, 8), np.nan), (1, 2, 2)),
        (np.ones((1, 8)), (1, 4, 4)),
    ]:
        with pytest.raises(ValueError):
            session.append_qwen25vl_visual(
                0, visual_embeddings=visual, image_grid_thw=grid
            )


@pytest.mark.parametrize("sequence_id", [2**32, -(2**32), -1, True, 1.5])
def test_visual_append_sequence_id_cannot_wrap_to_root(fake_session, sequence_id):
    np = pytest.importorskip("numpy")
    session, fake = fake_session
    fake.edgeflow_llm_session_append_qwen25vl_visual = _FakeCFunc(
        lambda *args: pytest.fail("invalid sequence ID reached native")
    )
    with pytest.raises(ValueError, match="sequence_id"):
        session.append_qwen25vl_visual(
            sequence_id, visual_embeddings=np.ones((1, 8)), image_grid_thw=(1, 2, 2)
        )


@pytest.mark.parametrize("failure", [RuntimeError("boom"), asyncio.CancelledError()])
def test_fork_context_removes_sequence_after_failure_or_cancellation(
    fake_session, failure
) -> None:
    session, _fake = fake_session
    session.ingest_text(0, "prefix", add_special=True)

    with pytest.raises(type(failure)), session.forked_sequence(0, 1, 3):
        raise failure

    assert not session.sequence_info(1).active


def test_fork_context_removes_sequence_after_success(fake_session) -> None:
    session, _fake = fake_session
    session.ingest_text(0, "prefix", add_special=True)

    with session.forked_sequence(0, 1, 3):
        assert session.sequence_info(1).active

    assert not session.sequence_info(1).active


def test_fork_context_closes_session_when_success_cleanup_fails(fake_session) -> None:
    session, fake = fake_session
    session.ingest_text(0, "prefix", add_special=True)

    with (
        pytest.raises(RuntimeError, match="remove sequence"),
        session.forked_sequence(0, 1, 3),
    ):
        fake.remove_status = 3

    assert fake.free_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        session.sequence_info(0)


def test_session_rejects_use_from_non_owner_thread(fake_session) -> None:
    session, _fake = fake_session
    errors: list[BaseException] = []

    def use_session() -> None:
        try:
            session.sequence_info(0)
        except RuntimeError as exc:
            errors.append(exc)

    thread = threading.Thread(target=use_session)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert "creating thread" in str(errors[0])


def test_close_is_idempotent(fake_session) -> None:
    session, fake = fake_session

    session.close()
    session.close()

    assert fake.free_count == 1


def test_binding_rejects_struct_size_mismatch() -> None:
    fake = _FakeLibrary()
    fake.edgeflow_llm_session_signature_sizeof = _FakeCFunc(lambda: 1)

    with pytest.raises(RuntimeError, match="signature_sizeof"):
        cs._bind_session_library(fake)


def test_binding_rejects_base_llm_abi_version_mismatch() -> None:
    fake = _FakeLibrary()
    fake.edgeflow_llm_abi_version = _FakeCFunc(lambda: cs.LLM_ABI_VERSION + 1)

    with pytest.raises(RuntimeError, match="native llama.cpp ABI mismatch"):
        cs._bind_session_library(fake)


def test_binding_rejects_previous_session_abi() -> None:
    fake = _FakeLibrary()
    fake.edgeflow_llm_session_abi_version = _FakeCFunc(
        lambda: cs.SESSION_ABI_VERSION - 1
    )

    with pytest.raises(RuntimeError, match="stateful session ABI mismatch"):
        cs._bind_session_library(fake)


def test_binding_rejects_qwen_request_size_mismatch() -> None:
    fake = _FakeLibrary()
    fake.edgeflow_llm_request_sizeof = _FakeCFunc(lambda: 1)

    with pytest.raises(RuntimeError, match="request_sizeof"):
        cs._bind_session_library(fake)


def test_native_error_includes_operation_and_reason(fake_session) -> None:
    session, fake = fake_session
    fake.edgeflow_llm_session_remove_sequence = _FakeCFunc(lambda *_args: 2)
    cs._bind_session_library(fake)

    with pytest.raises(
        RuntimeError, match="remove sequence failed with status 2: fake failure"
    ):
        session.remove_sequence(0)


def test_decode_step_rejects_non_finite_native_logits(fake_session) -> None:
    session, fake = fake_session
    session.ingest_text(0, "prefix", add_special=True)

    def decode_with_nan(ctx, sequence_id, out_ptr, logits, capacity) -> int:
        status = fake._decode(ctx, sequence_id, out_ptr, logits, capacity)
        logits[1] = float("nan")
        return status

    fake.edgeflow_llm_session_decode_step = _FakeCFunc(decode_with_nan)
    cs._bind_session_library(fake)

    with pytest.raises(RuntimeError, match="non-finite"):
        session.decode_step(0, capture_logits=True)


_SEQUENCE_ENTRYPOINTS = {
    "sequence_info": "edgeflow_llm_session_get_sequence_info",
    "ingest_text": "edgeflow_llm_session_ingest_text",
    "ingest_user_prompt": "edgeflow_llm_session_ingest_user_prompt",
    "ingest_tokens": "edgeflow_llm_session_ingest_tokens",
    "ingest_qwen25vl": "edgeflow_llm_session_ingest_qwen25vl",
    "ingest_qwen25vl_visual_prefix": "edgeflow_llm_session_ingest_qwen25vl_visual_prefix",
    "ingest_qwen25vl_query": "edgeflow_llm_session_ingest_qwen25vl_query",
    "decode_step": "edgeflow_llm_session_decode_step",
    "remove_sequence": "edgeflow_llm_session_remove_sequence",
}


def _record_ctypes_calls(monkeypatch, fake, symbol):
    calls = []

    def record(*args):
        calls.append(args)
        return 0

    bound = getattr(fake, symbol)
    callback = ctypes.CFUNCTYPE(ctypes.c_int, *bound.argtypes)(record)
    monkeypatch.setattr(fake, symbol, callback)
    return calls


def _invoke_sequence_entrypoint(session, method, sequence_id):
    if method in {"ingest_text", "ingest_user_prompt", "ingest_qwen25vl_query"}:
        return getattr(session, method)(sequence_id, "question")
    if method == "ingest_tokens":
        return session.ingest_tokens(sequence_id, [1, 2])
    if method in {"ingest_qwen25vl", "ingest_qwen25vl_visual_prefix"}:
        kwargs = {
            "visual_embeddings": np.ones((196, 8), dtype=np.float32),
            "image_grid_thw": (1, 28, 28),
        }
        if method == "ingest_qwen25vl":
            kwargs["prompt"] = "question"
        return getattr(session, method)(sequence_id, **kwargs)
    return getattr(session, method)(sequence_id)


@pytest.mark.parametrize("method", list(_SEQUENCE_ENTRYPOINTS))
@pytest.mark.parametrize("invalid", [2**32, 2**32 + 1, -(2**32), 2**31, 4, -1, True, 1.5, "1"])
def test_sequence_native_integer_is_rejected_before_ctypes(
    fake_session, monkeypatch, method, invalid
):
    session, fake = fake_session
    calls = _record_ctypes_calls(monkeypatch, fake, _SEQUENCE_ENTRYPOINTS[method])
    root_before = fake.sequence[0].copy()

    with pytest.raises(ValueError, match="sequence"):
        _invoke_sequence_entrypoint(session, method, invalid)

    assert calls == []
    assert fake.sequence[0] == root_before
    assert fake.free_count == 0


@pytest.mark.parametrize("field", [0, 1, 2])
@pytest.mark.parametrize("invalid", [2**32, 2**32 + 1, -1, True, 1.5, "1"])
def test_fork_native_integer_is_rejected_before_ctypes(
    fake_session, monkeypatch, field, invalid
):
    session, fake = fake_session
    calls = _record_ctypes_calls(monkeypatch, fake, "edgeflow_llm_session_fork_prefix")
    arguments = [0, 1, 3]
    arguments[field] = invalid

    with pytest.raises(ValueError):
        session.fork_prefix(*arguments)

    assert calls == [] and fake.free_count == 0


@pytest.mark.parametrize("method", ["ingest_tokens", "token_piece_bytes"])
@pytest.mark.parametrize("invalid", [2**32, 2**32 + 1, -1, 8, True, 1.5, "1"])
def test_token_native_integer_is_rejected_before_ctypes(
    fake_session, monkeypatch, method, invalid
):
    session, fake = fake_session
    symbol = (
        "edgeflow_llm_session_ingest_tokens" if method == "ingest_tokens"
        else "edgeflow_llm_session_token_piece"
    )
    calls = _record_ctypes_calls(monkeypatch, fake, symbol)

    with pytest.raises(ValueError, match="token"):
        if method == "ingest_tokens":
            session.ingest_tokens(0, [1, invalid, 2])
        else:
            session.token_piece_bytes(invalid)

    assert calls == [] and fake.free_count == 0
    assert session.sequence_info(0).next_position == 0


@pytest.mark.parametrize("method", ["ingest_qwen25vl", "ingest_qwen25vl_visual_prefix"])
@pytest.mark.parametrize("invalid", [2**32 + 28, 0, -1, True, 28.5, "28"])
def test_grid_native_integer_is_rejected_before_ctypes(
    fake_session, monkeypatch, method, invalid
):
    session, fake = fake_session
    calls = _record_ctypes_calls(monkeypatch, fake, _SEQUENCE_ENTRYPOINTS[method])
    kwargs = {
        "visual_embeddings": np.ones((196, 8), dtype=np.float32),
        "image_grid_thw": (1, invalid, 28),
    }
    if method == "ingest_qwen25vl":
        kwargs["prompt"] = "question"

    with pytest.raises(ValueError, match="grid"):
        getattr(session, method)(0, **kwargs)

    assert calls == [] and fake.free_count == 0


@pytest.mark.parametrize("field, invalid", [
    ("n_gpu_layers", 2**32), ("n_gpu_layers", -(2**31) - 1),
    ("n_ctx", 2**32 + 4096), ("n_ctx", 0), ("n_ctx", True),
    ("n_batch", 2**32 + 512), ("n_batch", 0),
    ("n_ubatch", 2**32 + 512), ("n_ubatch", 0),
    ("n_seq_max", 2**32 + 4), ("n_seq_max", 1),
    ("n_threads", 2**32 + 4), ("n_threads", 1.5),
    ("model_family", 2**32), ("position_contract", 2**32),
    ("kv_type_k", 2**32), ("kv_type_v", 2**32),
    ("prompt_contract_version", 2**32 + 1), ("prompt_contract_version", 0),
])
def test_invalid_session_config_is_rejected_before_library_load(
    fake_session, monkeypatch, field, invalid
):
    session, _fake = fake_session
    config = cs.SessionConfig(
        model_path=session._model_path, library_path=Path("unused.dylib")
    )

    with pytest.raises(ValueError, match=field):
        cs.StatefulLLMSession(replace(config, **{field: invalid}))


def test_valid_integer_scalars_preserve_session_lifecycle(fake_session):
    session, _fake = fake_session
    session.ingest_tokens(np.int64(0), [np.int64(1), np.int64(2)])
    session.fork_prefix(np.int64(0), np.int64(1), np.int64(2))
    assert session.decode_step(np.int64(1)).token_position == 2
    assert session.token_piece_bytes(np.int64(5)) == b"token-5"
    session.remove_sequence(np.int64(1))
    assert not session.sequence_info(np.int64(1)).active
    assert session.sequence_info(0).next_position == 2

from __future__ import annotations

# ruff: noqa: SIM117
# Inspect native receipts before the outer snapshot is released.
import ctypes
import gc
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from test_crossinfer_session import _FakeCFunc, _FakeLibrary

from edgeflow_split import crossinfer_session as cs

_ROOT_SYMBOLS = (
    "edgeflow_llm_session_export_root",
    "edgeflow_llm_session_import_root",
    "edgeflow_llm_root_snapshot_free",
    "edgeflow_llm_root_snapshot_size",
    "edgeflow_llm_session_reset_root",
)


class _RootLibrary(_FakeLibrary):
    def __init__(self, snapshots: dict[int, dict]) -> None:
        super().__init__()
        self.snapshots = snapshots
        self.root_free_count = 0
        self.root_calls: list[str] = []
        self.root_status = {"export": 0, "import": 0, "reset": 0}
        self.peek_status = 0
        self.peek_calls = []
        self.peek_is_eog = False
        self.edgeflow_llm_session_export_root = _FakeCFunc(self._export_root)
        self.edgeflow_llm_session_import_root = _FakeCFunc(self._import_root)
        self.edgeflow_llm_root_snapshot_free = _FakeCFunc(self._free_root)
        self.edgeflow_llm_root_snapshot_size = _FakeCFunc(self._root_size)
        self.edgeflow_llm_root_snapshot_equal = _FakeCFunc(
            lambda a, b: int(self.snapshots[a.value] == self.snapshots[b.value])
        )
        self.edgeflow_llm_session_reset_root = _FakeCFunc(self._reset_root)
        self.edgeflow_llm_session_peek_token = _FakeCFunc(self._peek_token)

    def _get_signature(self, ctx, out_ptr) -> int:
        status = super()._get_signature(ctx, out_ptr)
        if self.params.n_gpu_layers == 0:
            out_ptr._obj.observed_gpu_layers = 0
            out_ptr._obj.backend_placement = int(cs.BackendPlacement.CPU_ONLY)
        return status

    def _check_root_operation(self, operation: str) -> int:
        self.root_calls.append(operation)
        if self.root_status[operation]:
            return self.root_status[operation]
        if any(key != 0 and state["active"] for key, state in self.sequence.items()):
            self.error = b"root operation requires no live response branches"
            return 9
        return 0

    def _export_root(self, _ctx, out_ptr) -> int:
        status = self._check_root_operation("export")
        if status:
            return status
        state = dict(self.sequence[0])
        handle = id(state)
        self.snapshots[handle] = state
        out_ptr._obj.value = handle
        return 0

    def _import_root(self, _ctx, handle) -> int:
        status = self._check_root_operation("import")
        if status:
            return status
        self.sequence = {0: dict(self.snapshots[handle.value])}
        return 0

    def _free_root(self, handle) -> None:
        del self.snapshots[handle.value]
        self.root_free_count += 1

    def _root_size(self, handle) -> int:
        assert handle.value in self.snapshots
        return 2**32 + 128

    def _reset_root(self, _ctx) -> int:
        status = self._check_root_operation("reset")
        if status:
            return status
        self.sequence = {0: {"active": True, "position": 0, "logits": False}}
        return 0

    def _peek_token(self, _ctx, sequence_id, token_ptr, eog_ptr) -> int:
        self.peek_calls.append(sequence_id)
        if self.peek_status:
            return self.peek_status
        token_ptr._obj.value = 5
        eog_ptr._obj.value = int(self.peek_is_eog)
        return 0


@pytest.fixture
def session_factory(monkeypatch, tmp_path: Path):
    model = tmp_path / "model.gguf"
    library = tmp_path / "libedgeflow_llm.dylib"
    model.write_bytes(b"fake model")
    library.write_bytes(b"fake library")
    monkeypatch.setattr(
        cs, "_sha256_file", lambda path: "a" * 64 if path == model else "b" * 64
    )
    snapshots = {}
    sessions = []

    def create(*, root_api=True, missing=(), **config):
        fake = _RootLibrary(snapshots) if root_api else _FakeLibrary()
        for symbol in missing:
            delattr(fake, symbol)
        monkeypatch.setattr(cs.ctypes, "CDLL", lambda _path: fake)
        session = cs.StatefulLLMSession(
            cs.SessionConfig(model_path=model, library_path=library, **config)
        )
        sessions.append(session)
        return session, fake

    yield create
    for session in sessions:
        session.close()


def test_root_functions_bind_the_exact_optional_abi(session_factory) -> None:
    _, fake = session_factory()
    expected = (
        ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int),
        ([ctypes.c_void_p, ctypes.c_void_p], ctypes.c_int),
        ([ctypes.c_void_p], None),
        ([ctypes.c_void_p], ctypes.c_uint64),
        ([ctypes.c_void_p], ctypes.c_int),
    )
    for symbol, (argtypes, restype) in zip(_ROOT_SYMBOLS, expected, strict=True):
        fn = getattr(fake, symbol)
        assert fn.argtypes == argtypes
        assert fn.restype is restype


def test_snapshot_owns_immutable_state_and_outlives_source(session_factory) -> None:
    source, fake = session_factory()
    source.bind_stream("camera-1")
    source.ingest_text(0, "prefix")
    incarnation = source.session_incarnation_id
    snapshot = source.export_root_state()
    assert snapshot.source_signature == source.signature
    assert snapshot.source_session_incarnation_id == incarnation
    assert snapshot.bound_stream_id == "camera-1"
    assert snapshot.next_position == 3
    assert snapshot.size_bytes == 2**32 + 128
    with pytest.raises(FrozenInstanceError):
        snapshot.bound_stream_id = "camera-2"

    source.ingest_tokens(0, [1, 2])
    source.close()
    assert snapshot._lib is fake
    assert fake.free_count == 1
    assert fake.root_free_count == 0
    target, _ = session_factory()
    target.import_root_state(snapshot)
    assert target.sequence_info(0).next_position == 3
    assert snapshot.source_session_incarnation_id == incarnation
    with snapshot as entered:
        assert entered is snapshot
        assert entered.size_bytes == 2**32 + 128
    snapshot.close()
    del snapshot
    gc.collect()
    assert fake.root_free_count == 1
    assert not fake.snapshots


def test_snapshot_equality_uses_native_state_not_size(session_factory) -> None:
    session, fake = session_factory()
    fn = fake.edgeflow_llm_root_snapshot_equal
    assert fn.argtypes == [ctypes.c_void_p, ctypes.c_void_p]
    assert fn.restype is ctypes.c_int
    with session.export_root_state() as empty, session.export_root_state() as same:
        assert empty.same_state(same)
        session.ingest_tokens(0, [1])
        with session.export_root_state() as different:
            assert empty.size_bytes == different.size_bytes
            assert not empty.same_state(different)
        with pytest.raises(RuntimeError, match="snapshot is closed"):
            empty.same_state(different)


def test_snapshot_equality_is_optional(session_factory) -> None:
    session, _ = session_factory(missing=("edgeflow_llm_root_snapshot_equal",))
    with session.export_root_state() as snapshot:
        with pytest.raises(NotImplementedError, match="snapshot comparison"):
            snapshot.same_state(snapshot)


def test_unclosed_snapshot_is_freed_on_collection(session_factory) -> None:
    source, fake = session_factory()
    snapshot = source.export_root_state()
    source.close()
    del snapshot
    gc.collect()
    assert fake.root_free_count == 1


def test_import_allows_cross_placement_and_rotates_identity(session_factory) -> None:
    source, _ = session_factory(n_gpu_layers=0)
    target, fake = session_factory(n_gpu_layers=999, n_threads=2)
    source.bind_stream("camera-1")
    source.ingest_text(0, "prefix")
    target.bind_stream("camera-1")
    before = target.session_incarnation_id
    with source.export_root_state() as snapshot:
        assert snapshot.source_signature != target.signature
        assert (
            snapshot.source_signature.backend_placement == cs.BackendPlacement.CPU_ONLY
        )
        assert target.signature.backend_placement == cs.BackendPlacement.GPU_FULL
        target.import_root_state(snapshot)
        assert target.sequence_info(0).next_position == snapshot.next_position
        assert target.session_incarnation_id not in (
            before,
            snapshot.source_session_incarnation_id,
        )
        first_import = target.session_incarnation_id
        target.import_root_state(snapshot)
        assert target.session_incarnation_id != first_import
    assert fake.root_calls == ["import", "import"]
    target.bind_stream("camera-1")
    with pytest.raises(RuntimeError, match="already bound"):
        target.bind_stream("camera-2")


@pytest.mark.parametrize("source_stream", ["camera-2", None])
def test_import_refuses_overwriting_another_or_unbound_stream(
    session_factory, source_stream
) -> None:
    source, _ = session_factory()
    target, fake = session_factory()
    if source_stream is not None:
        source.bind_stream(source_stream)
    target.bind_stream("camera-1")
    before = target.session_incarnation_id
    with source.export_root_state() as snapshot:
        with pytest.raises(RuntimeError, match="already bound"):
            target.import_root_state(snapshot)
    assert fake.root_calls == []
    assert target.session_incarnation_id == before
    target.bind_stream("camera-1")


def test_reset_clears_binding_and_rotates_identity(session_factory) -> None:
    source, _ = session_factory()
    target, fake = session_factory()
    source.bind_stream("camera-2")
    source.ingest_text(0, "prefix")
    target.bind_stream("camera-1")
    target.ingest_text(0, "old prefix")
    before = target.session_incarnation_id
    target.reset_root()
    after_reset = target.session_incarnation_id
    assert after_reset != before
    assert target.sequence_info(0).next_position == 0
    with source.export_root_state() as snapshot:
        target.import_root_state(snapshot)
    assert target.session_incarnation_id != after_reset
    assert target.sequence_info(0).next_position == 3
    target.bind_stream("camera-2")
    with pytest.raises(RuntimeError, match="already bound"):
        target.bind_stream("camera-1")
    target.reset_root()
    target.bind_stream("camera-3")
    assert fake.root_calls == ["reset", "import", "reset"]


@pytest.mark.parametrize("operation", ["export", "import", "reset"])
def test_live_branch_errors_are_delegated_to_native_status(
    session_factory, operation
) -> None:
    source, _ = session_factory()
    target, fake = session_factory()
    target.bind_stream("camera-1")
    target.ingest_text(0, "prefix")
    target.fork_prefix(0, 1, 3)
    before = target.session_incarnation_id
    source.bind_stream("camera-1")
    with source.export_root_state() as snapshot:
        with pytest.raises(RuntimeError, match="status 9: .*no live response branches"):
            if operation == "export":
                target.export_root_state()
            elif operation == "import":
                target.import_root_state(snapshot)
            else:
                target.reset_root()
    assert fake.root_calls == [operation]
    assert target.session_incarnation_id == before
    assert target.sequence_info(1).active
    target.bind_stream("camera-1")


@pytest.mark.parametrize("operation", ["export", "import", "reset"])
def test_native_format_or_operation_error_preserves_identity(
    session_factory, operation
) -> None:
    source, _ = session_factory()
    target, fake = session_factory()
    fake.root_status[operation] = 17
    fake.error = b"incompatible root state format"
    before = target.session_incarnation_id
    with (
        source.export_root_state() as snapshot,
        pytest.raises(RuntimeError, match="status 17: incompatible root state format"),
    ):
        if operation == "export":
            target.export_root_state()
        elif operation == "import":
            target.import_root_state(snapshot)
        else:
            target.reset_root()
    assert target.session_incarnation_id == before
    assert fake.root_calls == [operation]


def test_closed_snapshot_cannot_be_used(session_factory) -> None:
    source, fake = session_factory()
    target, target_fake = session_factory()
    snapshot = source.export_root_state()
    snapshot.close()
    for operation in (
        lambda: snapshot.size_bytes,
        snapshot.__enter__,
        lambda: target.import_root_state(snapshot),
    ):
        with pytest.raises(RuntimeError, match="snapshot is closed"):
            operation()
    snapshot.close()
    assert fake.root_free_count == 1
    assert target_fake.root_calls == []
    with pytest.raises(TypeError, match="RootStateSnapshot"):
        target.import_root_state(object())


def test_export_position_metadata_is_optional(session_factory) -> None:
    source, fake = session_factory()
    fake.edgeflow_llm_session_get_sequence_info.fn = lambda *_args: 11
    with source.export_root_state() as snapshot:
        assert snapshot.next_position is None


def test_successful_native_export_must_return_a_handle(session_factory) -> None:
    source, fake = session_factory()
    fake.edgeflow_llm_session_export_root.fn = lambda *_args: 0
    with pytest.raises(RuntimeError, match="null snapshot"):
        source.export_root_state()


@pytest.mark.parametrize("operation", ["export", "import", "reset"])
def test_old_library_works_until_optional_root_api_is_used(
    session_factory, operation
) -> None:
    source, _ = session_factory()
    target, fake = session_factory(root_api=False)
    target.ingest_text(0, "prefix")
    assert target.decode_step(0).token_id == 5
    before = target.session_incarnation_id
    with source.export_root_state() as snapshot:
        with pytest.raises(NotImplementedError, match="does not support"):
            if operation == "export":
                target.export_root_state()
            elif operation == "import":
                target.import_root_state(snapshot)
            else:
                target.reset_root()
    assert target.session_incarnation_id == before
    assert fake.free_count == 0


@pytest.mark.parametrize("missing", _ROOT_SYMBOLS[:1] + _ROOT_SYMBOLS[2:4])
def test_export_requires_all_snapshot_ownership_symbols(
    session_factory, missing
) -> None:
    source, fake = session_factory(missing=(missing,))
    with pytest.raises(NotImplementedError, match=missing):
        source.export_root_state()
    assert fake.root_calls == []
    assert not fake.snapshots


@pytest.mark.parametrize("is_eog", [False, True])
def test_peek_token_is_optional_and_does_not_advance(session_factory, is_eog) -> None:
    session, fake = session_factory()
    peek = fake.edgeflow_llm_session_peek_token
    assert peek.argtypes == [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    assert peek.restype is ctypes.c_int
    session.ingest_text(0, "prefix")
    before = session.sequence_info(0)
    fake.peek_is_eog = is_eog
    assert session.peek_token(0) == (5, is_eog)
    assert session.peek_token(0) == (5, is_eog)
    assert session.sequence_info(0) == before
    assert fake.peek_calls == [0, 0]
    assert session.decode_step(0).token_position == before.next_position
    assert session.sequence_info(0).next_position == before.next_position + 1


def test_missing_peek_does_not_affect_root_or_decode(session_factory) -> None:
    session, _ = session_factory(missing=("edgeflow_llm_session_peek_token",))
    session.ingest_text(0, "prefix")
    with session.export_root_state() as snapshot:
        assert snapshot.next_position == 3
    with pytest.raises(NotImplementedError, match="edgeflow_llm_session_peek_token"):
        session.peek_token(0)
    assert session.decode_step(0).token_position == 3


def test_missing_root_does_not_affect_peek(session_factory) -> None:
    session, _ = session_factory(missing=_ROOT_SYMBOLS)
    session.ingest_text(0, "prefix")
    assert session.peek_token(0) == (5, False)
    with pytest.raises(NotImplementedError):
        session.export_root_state()


def test_peek_propagates_status_and_checks_sequence_id(session_factory) -> None:
    session, fake = session_factory()
    fake.peek_status = 13
    before = session.sequence_info(0)
    with pytest.raises(
        RuntimeError, match="peek token failed with status 13: fake failure"
    ):
        session.peek_token(0)
    assert session.sequence_info(0) == before
    assert fake.peek_calls == [0]
    for invalid in (-1, session.signature.n_seq_max, 2**32, True, 1.5):
        with pytest.raises(ValueError, match="sequence_id"):
            session.peek_token(invalid)
    assert fake.peek_calls == [0]

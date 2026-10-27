import ctypes
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_crossinfer_root_state import (
    session_factory as session_factory,  # noqa: PLC0414 - pytest fixture re-export
)
from test_crossinfer_session import _FakeCFunc

from edgeflow_split import crossinfer_session as cs


def test_ignore_eog_flags_and_errors(session_factory):
    session, lib = session_factory()
    calls = []
    lib.edgeflow_llm_session_set_ignore_eog = _FakeCFunc(
        lambda _, flag: calls.append(flag) or 0
    )
    cs._bind_session_library(lib)
    fn = lib.edgeflow_llm_session_set_ignore_eog
    assert fn.argtypes == [ctypes.c_void_p, ctypes.c_int]
    session.set_ignore_eog(True)
    session.set_ignore_eog(False)
    assert calls == [1, 0]
    with pytest.raises(TypeError):
        session.set_ignore_eog(1)
    with ThreadPoolExecutor(1) as pool, pytest.raises(RuntimeError):
        pool.submit(session.set_ignore_eog, True).result()
    assert calls == [1, 0]
    lib.edgeflow_llm_session_set_ignore_eog = _FakeCFunc(lambda *_: 9)
    with pytest.raises(RuntimeError):
        session.set_ignore_eog(True)


def test_ignore_eog_no_silent_fallback(session_factory):
    session, _ = session_factory()
    with pytest.raises(NotImplementedError, match="set_ignore_eog"):
        session.set_ignore_eog(True)

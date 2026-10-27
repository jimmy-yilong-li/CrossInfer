"""Exercise the production C++ logits helper without models or accelerators."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def logits_test_binary(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required for the native logits unit test")
    runtime = Path(__file__).resolve().parents[1] / "tools/edgeflow_llm_runtime"
    executable = tmp_path_factory.mktemp("session-logits") / "test_logits"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(runtime / "src"),
            str(runtime / "tests/session_logits_test.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return executable


@pytest.mark.parametrize(
    "case", ["argmax", "reuse", "invalidate", "fork", "nonfinite", "clear", "exclude", "exclude-cache"]
)
def test_native_session_logits(logits_test_binary, case):
    result = subprocess.run(
        [str(logits_test_binary), case],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout.strip() == "PASS"

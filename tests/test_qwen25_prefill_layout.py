"""No model needed: packed rows keep Qwen's text and image positions."""

import subprocess
from pathlib import Path


def test_native_packed_positions(tmp_path):
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "layout"
    subprocess.run([
        "c++", "-std=c++17", "-I", str(root / "tools/edgeflow_llm_runtime/src"),
        str(root / "tests/native_qwen25_prefill.cpp"), "-o", str(executable),
    ], check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True)

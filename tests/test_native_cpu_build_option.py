"""Exercise the runtime's CPU cache option without compiling or loading models."""

from pathlib import Path
import re
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("requested,expected", [(None, "OFF"), ("ON", "ON"), ("OFF", "OFF")])
def test_native_cpu_flag_keeps_default_but_accepts_explicit_override(tmp_path, requested, expected):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is not installed")
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools/edgeflow_llm_runtime/CMakeLists.txt").read_text()
    declaration = re.search(r"(?m)^set\(GGML_NATIVE[^\n]*\)", source)
    assert declaration is not None
    script = tmp_path / "cpu-option.cmake"
    script.write_text(declaration.group() + '\nmessage(STATUS "native=${GGML_NATIVE}")\n')
    command = [cmake]
    if requested is not None:
        command.append(f"-DGGML_NATIVE:BOOL={requested}")
    result = subprocess.run([*command, "-P", str(script)], capture_output=True, text=True, check=True)
    assert f"native={expected}" in result.stdout

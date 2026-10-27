import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def native_role_test(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    runtime = root / "tools/edgeflow_llm_runtime"
    library = runtime / "build/libedgeflow_llm.dylib"
    compiler = shutil.which("c++")
    if not compiler or not library.exists():
        pytest.skip("requires a built runtime for RTTI symbols, not a model")
    output = tmp_path_factory.mktemp("role-native") / "test_role"
    llama = runtime / "llama_src"
    result = subprocess.run([
        compiler, "-std=c++17", "-O1", "-Wno-deprecated-declarations", "-Wl,-dead_strip",
        "-I", str(runtime / "include"), "-I", str(llama / "include"),
        "-I", str(llama / "src"), "-I", str(llama / "ggml/include"),
        str(root / "tests/native_role_session.cpp"), str(library),
        f"-Wl,-rpath,{library.parent}", "-o", str(output),
    ], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    return output


@pytest.mark.parametrize("case", ["eog", "surface", "failure", "snapshot"])
def test_production_native_session_without_model(native_role_test, case):
    result = subprocess.run([str(native_role_test), case], capture_output=True, text=True,
                            timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PASS"

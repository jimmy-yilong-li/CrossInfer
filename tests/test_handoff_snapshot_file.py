"""Run the native snapshot writer/reader with actual state structs, no model."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def snapshot_tool(tmp_path_factory):
    native = Path(__file__).resolve().parents[1] / "RK3588/edgeflow_rk3588/runtime/tabm/native"
    shared = (native / "mnn_handoff_state.hpp").read_text()
    structs = shared[shared.index("struct HandoffLayer"):shared.index("static size_t bounded_kv_count")]
    directory = tmp_path_factory.mktemp("kv-snapshot")
    source = directory / "test.cpp"
    source.write_text(
        "#include <cstdint>\n#include <string>\n#include <vector>\n#include <map>\n"
        "#include <stdexcept>\n#include <iostream>\n#include <cstring>\n#include <memory>\n"
        '#include "mnn_kv_transfer.hpp"\n'
        "static void require(bool v, const std::string& e) { if (!v) throw std::runtime_error(e); }\n"
        + structs
        + r'''
#include "handoff_snapshot_file.hpp"
int main(int argc, char** argv) {
    try {
        const std::string mode(argv[1]);
        if (mode == "write" || mode == "prefix" || mode == "prefix_ref") {
            WorkerKVSnapshot file;
            file.stream = "stream_a"; file.model_key = "model_a";
            file.content_digest = std::string(64, 'a');
            file.version = 2; file.hidden = 896; file.prefix = {10, 11};
            file.state.positions = 4;
            for (int i = 0; i < 2; ++i) {
                auto& layer = file.state.layers["attention_" + std::to_string(i)];
                layer.tokens = 4; layer.heads = 1; layer.dimension = 2;
                layer.key = {1,2,3,4,5,6,7,8}; layer.value = {8,7,6,5,4,3,2,1};
            }
            if (mode == "prefix") {
                file.version = 1;
                auto captured = std::move(file.state);
                write_worker_kv_snapshot(argv[2], file, captured, 3);
                require(captured.positions == 4 && captured.layers.begin()->second.key.size() == 8,
                        "prefix export mutated the donor");
            } else {
                if (mode == "prefix_ref") {
                    file.version = 1; file.state.positions = 3;
                    for (auto& item : file.state.layers) {
                        item.second.tokens = 3;
                        item.second.key.resize(6); item.second.value.resize(6);
                    }
                }
                write_worker_kv_snapshot(argv[2], file);
            }
        } else {
            auto file = read_worker_kv_snapshot(argv[2]);
            require(file.stream == "stream_a" && file.model_key == "model_a", "identity");
            require(file.version == 2 && file.hidden == 896 && file.prefix.size() == 2, "metadata");
            require(file.state.positions == 4 && file.state.layers.size() == 2, "layers");
            require(file.state.layers.at("attention_0").key[7] == 8, "key bytes");
            require(file.state.layers.at("attention_1").value[7] == 1, "value bytes");
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what(); return 2;
    }
}
'''
    )
    binary = directory / "snapshot"
    subprocess.run([shutil.which("c++"), "-std=c++17", "-I", str(native),
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary


def test_actual_snapshot_roundtrip_and_no_overwrite(snapshot_tool, tmp_path):
    file = tmp_path / "state.kv"
    subprocess.run([snapshot_tool, "write", file], check=True)
    subprocess.run([snapshot_tool, "read", file], check=True)
    before = file.read_bytes()
    assert subprocess.run([snapshot_tool, "write", file], capture_output=True, check=False).returncode == 2
    assert file.read_bytes() == before


@pytest.mark.parametrize("cut", [0, 8, 24, 80, -1])
def test_truncated_snapshots_fail(snapshot_tool, tmp_path, cut):
    file = tmp_path / "state.kv"
    subprocess.run([snapshot_tool, "write", file], check=True)
    file.write_bytes(file.read_bytes()[:cut])
    assert subprocess.run([snapshot_tool, "read", file], capture_output=True, check=False).returncode == 2


def test_bad_magic_fails_before_import(snapshot_tool, tmp_path):
    file = tmp_path / "state.kv"
    subprocess.run([snapshot_tool, "write", file], check=True)
    file.write_bytes(b"BROKEN!!" + file.read_bytes()[8:])
    assert subprocess.run([snapshot_tool, "read", file], capture_output=True, check=False).returncode == 2


def test_prefix_export_is_byte_identical_without_copying_or_mutating_source(snapshot_tool, tmp_path):
    prefix, reference = tmp_path / "prefix.kv", tmp_path / "ref.kv"
    subprocess.run([snapshot_tool, "prefix", prefix], check=True)
    subprocess.run([snapshot_tool, "prefix_ref", reference], check=True)
    assert prefix.read_bytes() == reference.read_bytes()

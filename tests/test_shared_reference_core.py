"""Model-free import and build-command checks for the shared reference core."""

import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "RK3588/edgeflow_rk3588/runtime/tabm/native"
QWEN = ROOT / "QCS6490/edgeflow_qcs6490/native"


def test_core_imports_without_private_modules():
    script = """
import importlib.abc
import sys
from dataclasses import replace

class BlockPrivate(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if '.scripts' in fullname or fullname.rsplit('.', 1)[-1] in {
            'rknn_vision', 'inference_engine', 'text_backends',
            'visual_replay_serving', 'stateful_serving', 'vlm_serving',
        }:
            raise AssertionError('private import: ' + fullname)

sys.meta_path.insert(0, BlockPrivate())
import numpy as np
from QCS6490.edgeflow_qcs6490 import qnn_vision, qwen_vision
from RK3588.edgeflow_rk3588 import runtime
from RK3588.edgeflow_rk3588.runtime.tabm import retained_lane, visual_kv_cache
from RK3588.edgeflow_rk3588.runtime.tabm.native import patch_mnn_kv_transfer
from RK3588.edgeflow_rk3588.runtime.tabm.visual_context import (
    ProjectedEmbeddingLog, validate_replay_snapshot,
)
assert runtime.__all__ == ['VisualPayload']
assert runtime.VisualPayload
assert retained_lane.validate_replay_snapshot is validate_replay_snapshot
assert visual_kv_cache.validate_replay_snapshot is validate_replay_snapshot
log = ProjectedEmbeddingLog('stream', 'session', 2, 4, 32)
snapshot = log.append('frame', 0, np.ones((1, 2), dtype=np.float32))
validate_replay_snapshot(snapshot)
validate_replay_snapshot(replace(snapshot, _payload=b''.join(snapshot.chunks)))
try:
    validate_replay_snapshot(replace(snapshot, content_digest='0' * 64))
except ValueError:
    pass
else:
    raise AssertionError('invalid snapshot accepted')
assert 'onnxruntime' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script], cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["generic", "qwen", "visual"])
def test_native_build_commands(tmp_path, mode):
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("cmake", "c++"):
        tool = tools / name
        tool.write_text('#!/bin/sh\nprintf "%s\\n" "$0 $*" >> "$CALL_LOG"\n')
        tool.chmod(0o755)
    log = tmp_path / "commands"
    args = ["--qwen-positions"] if mode == "qwen" else []
    if mode == "visual":
        args = [str(QWEN / "qwen_visual_worker.cpp")]
    build, source = tmp_path / "mnn-build", tmp_path / "mnn-source"
    result = subprocess.run(
        ["bash", str(NATIVE / "build.sh"), *args], cwd=tmp_path,
        env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}",
             "MNN_BUILD": str(build), "MNN_SRC": str(source),
             "MNN_BUILD_JOBS": "2", "CALL_LOG": str(log)},
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [shlex.split(line) for line in log.read_text().splitlines()]
    assert commands[0][1:] == [
        "--build", str(build), "--target", "llm", "MNN", "MNN_Express",
        "MNN_CL", "--parallel", "2",
    ]
    compiler = commands[1]
    assert "-fexceptions" in compiler and "-fno-exceptions" not in compiler
    assert ("-DCROSSINFER_RETAINED_KV" in compiler) == (mode != "visual")
    assert ("-DCROSSINFER_QWEN_POSITIONS" in compiler) == (mode == "qwen")
    if mode == "qwen":
        for directory in (QWEN, source / "source", source / "transformers/llm/engine/src"):
            assert f"-I{directory}" in compiler
    expected = QWEN / "qwen_visual_worker.cpp" if mode == "visual" else NATIVE / "tabm_decoder_worker.cpp"
    assert str(expected) in compiler


def test_curated_worker_has_no_marker_export():
    assert "export_marker_embeddings" not in (QWEN / "qwen_mnn_positions.hpp").read_text()
    assert "--qwen-export-markers" not in (NATIVE / "tabm_decoder_worker.cpp").read_text()
    assert not (ROOT / "RK3588/edgeflow_rk3588/scripts/patch_mnn_kv_transfer.py").exists()


def test_prefix_costs_are_caller_supplied():
    from types import SimpleNamespace

    from RK3588.edgeflow_rk3588.runtime.tabm.dispatch import (
        PrefixAffinityConfig, affinity_saving_s,
    )
    from RK3588.edgeflow_rk3588.runtime.tabm.prefix_sched import make_saving_estimator

    lane = SimpleNamespace(supports_prefix_cache=True, has_prefix=lambda _: True)
    assert affinity_saving_s(lane, "prefix", 10, PrefixAffinityConfig()) == 0.0
    assert make_saving_estimator()("prefix", 10) == (0.0, "est")
    # Synthetic input checks the formula, not a device performance claim.
    config = PrefixAffinityConfig(alpha=0.5, per_token_prefill_s=0.003)
    assert affinity_saving_s(lane, "prefix", 10, config) == pytest.approx(0.015)
    estimate = make_saving_estimator(est_alpha=0.5, est_per_token_s=0.003)
    seconds, source = estimate("prefix", 10)
    assert seconds == pytest.approx(0.015) and source == "est"

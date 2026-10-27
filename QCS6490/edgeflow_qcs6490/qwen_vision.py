"""Fixed-420 Qwen2.5-VL-3B image frontend using the trained MNN visual module.

CPU vision only, not HTP or native video. Each RGB image is bilinearly resized
directly to 420x420 (no black padding) and duplicated for temporal patch size 2.
The module includes the trained merger and reverse-window ordering. Markers
must be the raw fp32 [3,2048] export from QwenMnnLlm, not synthetic embeddings.
No model parity, quality, or Rubik performance validation is implied.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import tempfile
import threading
from pathlib import Path
from time import monotonic, perf_counter

import numpy as np
from PIL import Image

from QCS6490.edgeflow_qcs6490.qnn_vision import QNNVisionResult

MARKER_IDS = (151652, 151653, 198)
FEATURE_SHAPE = (225, 2048)
MARKER_SHAPE = (3, 2048)


def qwen_fixed420_metadata() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return int32 positions, float32 masks, int32 window_index as in Omni."""
    positions = np.empty((2, 900), dtype=np.int32)
    for h in range(30):
        for w in range(30):
            index = (h // 2) * 60 + (w // 2) * 4 + (h % 2) * 2 + w % 2
            positions[:, index] = (h, w)

    indices, counts = [], []
    # Omni pads the index grid to 16x16, but never pads the actual image.
    for win_h in range(4):
        for win_w in range(4):
            count = 0
            for i in range(4):
                h = win_h * 4 + i
                if h >= 15:
                    continue
                for j in range(4):
                    w = win_w * 4 + j
                    if w < 15:
                        indices.append(h * 15 + w)
                        count += 1
            counts.append(count)

    mask = np.zeros((2, 1, 900, 900), dtype=np.float32)
    mask[1].fill(np.finfo(np.float32).min)
    start = 0
    for count in counts:
        end = start + count * 4
        mask[1, 0, start:end, start:end] = 0
        start = end
    return positions, mask, np.asarray(indices, dtype=np.int32)


def preprocess_qwen_fixed420(image_path: str | Path, config: dict) -> np.ndarray:
    """Omni's merge-block/channel/temporal/patch order, with Pillow RGB resize."""
    mean = np.asarray(config["image_mean"], dtype=np.float32) / 255.0
    norm = np.asarray(config["image_norm"], dtype=np.float32) * 255.0
    if (
        mean.shape != (3,)
        or norm.shape != (3,)
        or not np.isfinite(mean).all()
        or not np.isfinite(norm).all()
        or np.any(norm <= 0)
    ):
        raise ValueError("image_mean/image_norm must contain three finite RGB values")
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize((420, 420), Image.Resampling.BILINEAR)
    pixels = (np.asarray(image, dtype=np.float32) / 255.0 - mean) * norm
    frames = np.stack([pixels.transpose(2, 0, 1)] * 2)
    patches = frames.reshape(1, 2, 3, 15, 2, 14, 15, 2, 14)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(900, 1176)
    return np.ascontiguousarray(patches, dtype=np.float32)


def assemble_qwen_packet(features: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Copy [start | 225 row-major features | end | newline]; no projection."""
    for value, shape in ((features, FEATURE_SHAPE), (markers, MARKER_SHAPE)):
        if (
            value.shape != shape
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"expected finite float32 {shape}")
    return np.concatenate((markers[:1], features, markers[1:]), axis=0)


class QwenVisualEncoder:
    """One lazy persistent CPU worker with one request in flight and bounded IO.

    ``model`` is visual.mnn with adjacent visual.mnn.weight and llm_config.json.
    ``worker`` is qwen_visual_worker; ``directory`` holds temporary raw inputs
    and output, removed after every request. CPU threads=2, precision=high.
    ``vision_ms`` covers native onForward through readMap, excluding file IO,
    loading and preprocessing. ``projection_ms`` is packet assembly only (the
    trained merger is already counted in vision_ms). Call close on the owner.
    """

    def __init__(
        self,
        model: str | Path,
        worker: str | Path,
        marker_embeddings: str | Path,
        directory: str | Path,
        *,
        timeout_s: float = 180.0,
        backend: str = "cpu",
    ) -> None:
        self.model, self.worker = Path(model).resolve(), Path(worker).resolve()
        if backend not in ("cpu", "opencl"):
            raise ValueError("backend must be cpu or opencl")
        self.backend = backend
        self.directory = Path(directory).resolve()
        if any(c in str(self.directory) for c in "\t\r\n"):
            raise ValueError("directory must not contain protocol delimiters")
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        for path in (self.model, Path(str(self.model) + ".weight"), self.worker):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.config = json.loads((self.model.parent / "llm_config.json").read_text())
        expected = {
            "model_type": "qwen2_5_vl",
            "hidden_size": 2048,
            "image_size": 420,
            "vision_start": MARKER_IDS[0],
            "vision_end": MARKER_IDS[1],
        }
        if any(self.config.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "requires fixed420 Qwen2.5-VL-3B config and matching marker IDs"
            )
        marker_path = Path(marker_embeddings)
        if marker_path.stat().st_size != 3 * 2048 * 4:
            raise ValueError(
                "marker_embeddings must be raw float32 [3,2048] in ID order 151652/151653/198"
            )
        self._markers = np.fromfile(marker_path, dtype=np.float32).reshape(MARKER_SHAPE)
        if not np.isfinite(self._markers).all():
            raise ValueError("marker_embeddings contain non-finite values")
        self._positions, self._mask, self._window_index = qwen_fixed420_metadata()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timeout_s = float(timeout_s)
        self._process = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._closed = False

    def _reply(self) -> str:
        deadline = monotonic() + self.timeout_s
        while True:
            while b"\n" in self._buffer:
                raw, _, rest = self._buffer.partition(b"\n")
                self._buffer = bytearray(rest)
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("FATAL "):
                    raise RuntimeError(line)
                if line.startswith(("READY ", "RES ")):
                    return line
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("Qwen visual worker response timed out")
            fd = self._process.stdout.fileno()
            if not select.select([fd], [], [], remaining)[0]:
                raise TimeoutError("Qwen visual worker response timed out")
            chunk = os.read(fd, 4096)
            if not chunk:
                raise RuntimeError("Qwen visual worker closed stdout")
            self._buffer.extend(chunk)
            if len(self._buffer) > 65536:
                raise RuntimeError("Qwen visual worker stdout line exceeds bound")

    def _start(self) -> None:
        if self._process is not None:
            return
        self._process = subprocess.Popen(
            [str(self.worker), str(self.model), self.backend],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
            cwd=self.worker.parent,
        )
        threads = 65 if self.backend == "opencl" else 2
        if self._reply() != f"READY mnn_{self.backend} {threads} 225 2048":
            raise RuntimeError("unexpected Qwen visual worker CPU/shape handshake")

    def encode(self, image_path: str | Path) -> QNNVisionResult:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Qwen visual encoder already has an in-flight request")
        try:
            if self._closed:
                raise RuntimeError("Qwen visual encoder is closed")
            patches = preprocess_qwen_fixed420(image_path, self.config)
            with tempfile.TemporaryDirectory(
                prefix="qwen-vision-", dir=self.directory
            ) as name:
                paths = [
                    Path(name) / f"{key}.bin"
                    for key in (
                        "patches",
                        "positions",
                        "mask",
                        "window_index",
                        "output",
                    )
                ]
                for path, array in zip(
                    paths, (patches, self._positions, self._mask, self._window_index)
                ):
                    array.tofile(path)
                command = ("ENCODE\t" + "\t".join(map(str, paths)) + "\n").encode()
                if len(command) > 4096:
                    raise ValueError("Qwen visual request paths exceed protocol bound")
                try:
                    self._start()
                    self._process.stdin.write(command)
                    self._process.stdin.flush()
                    reply = self._reply().split()
                    if len(reply) != 2 or reply[0] != "RES":
                        raise RuntimeError("unexpected Qwen visual worker response")
                    vision_ms = float(reply[1])
                    if not np.isfinite(vision_ms) or vision_ms < 0:
                        raise RuntimeError("invalid Qwen visual worker timing")
                    if paths[-1].stat().st_size != 225 * 2048 * 4:
                        raise ValueError("visual output must be raw float32 [225,2048]")
                    features = np.fromfile(paths[-1], dtype=np.float32).reshape(
                        FEATURE_SHAPE
                    )
                    start = perf_counter()
                    packet = assemble_qwen_packet(features, self._markers)
                    projection_ms = (perf_counter() - start) * 1e3
                except BaseException:
                    self._closed = True
                    self._stop()
                    raise
            return QNNVisionResult(packet, vision_ms, projection_ms, f"mnn_{self.backend}")
        finally:
            self._lock.release()

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    process.stdin.write(b"QUIT\n")
                    process.stdin.flush()
                except OSError:
                    pass
            process.stdin.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        finally:
            process.stdout.close()
            self._buffer.clear()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._stop()

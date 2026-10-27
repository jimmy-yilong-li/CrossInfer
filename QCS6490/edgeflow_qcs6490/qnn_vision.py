"""Independent ORT-QNN adapter for the trained OneVision SigLIP fixed384 tower.

HTP uses the installed Rubik QNN libraries, without CPU node/retry fallback.
``cpu`` explicitly selects ORT's CPU reference, not QNN's CPU backend. Both
paths reuse the trained NumPy OneVision projector on CPU; the backend label
describes the vision session only. No hardware or quality validation is implied.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from time import perf_counter, perf_counter_ns
from typing import Any

import numpy as np
from PIL import Image

from RK3588.edgeflow_rk3588.runtime.onevision_fixed384 import (
    OneVisionProjector,
    preprocess_siglip_fixed384,
)

SIGLIP_INPUT_SHAPE = (1, 3, 384, 384)
SIGLIP_FEATURE_SHAPE = (1, 729, 1152)
DEFAULT_HTP_BACKEND_PATH = "/usr/lib/libQnnHtp.so"


def preprocess_fastvlm_fixed1024(image_path):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        side = max(image.size)
        square = Image.new("RGB", (side, side), (0, 0, 0))
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        image = square.resize((1024, 1024), Image.BICUBIC)
    return np.ascontiguousarray(
        (np.asarray(image, dtype=np.float32) / 255).transpose(2, 0, 1)[None]
    )


@dataclass(frozen=True)
class QNNVisionResult:
    """Host wall timings; parts follow session order, not hardware utilization."""

    embedding: np.ndarray
    vision_ms: float
    projection_ms: float
    backend: str
    parts_ms: tuple[float, ...] = ()


class QNNVisualEncoder:
    """Encode a single image with a fixed-shape float32-I/O ONNX tower.

    ``model`` exports the checkpoint-selected OneVision vision_feature_layer,
    not the SigLIP pooled head (QDQ W8A16 for HTP). ``projector`` is its matching
    OneVisionProjector or NPZ, including the trained image_newline.
    ``video_pool2`` yields (197, hidden); ``image`` yields (730, hidden).
    ORT is imported and the session constructed only on the first encode.
    ``session_factory`` is an injection point for stub-only contract tests.
    Vision timing covers session.run only, excluding loading and preprocessing.
    """

    def __init__(
        self,
        model: str | Path,
        projector: OneVisionProjector | str | Path,
        *,
        backend: str = "htp",
        visual_token_mode: str = "video_pool2",
        family: str = "onevision",
        backend_path: str | Path = DEFAULT_HTP_BACKEND_PATH,
        htp_performance_mode: str | None = None,
        cpu_threads: int = 2,
        allow_spinning: bool | None = None,
        htp_threads: int | None = None,
        session_factory: Callable[..., Any] | None = None,
    ) -> None:
        if backend not in ("htp", "cpu", "partitioned"):
            raise ValueError("backend must be htp, cpu, or explicit partitioned mode")
        if family not in ("onevision", "fastvlm"):
            raise ValueError("family must be onevision or fastvlm")
        if visual_token_mode not in ("video_pool2", "image") or (
            family == "fastvlm" and visual_token_mode != "image"
        ):
            raise ValueError(f"unsupported visual_token_mode: {visual_token_mode!r}")
        self.family = family
        self.input_shape = (
            SIGLIP_INPUT_SHAPE if family == "onevision" else (1, 3, 1024, 1024)
        )
        self.feature_shape = (
            SIGLIP_FEATURE_SHAPE if family == "onevision" else (1, 256, 3072)
        )
        if family == "fastvlm" and isinstance(projector, (str, Path)):
            with np.load(projector) as data:
                projector = OneVisionProjector(
                    *(
                        np.asarray(data[key], dtype=np.float32)
                        for key in ("0.weight", "0.bias", "2.weight", "2.bias")
                    ),
                    None,
                )
        self.projector = (
            OneVisionProjector.from_npz(projector)
            if isinstance(projector, (str, Path))
            else projector
        )
        if not isinstance(self.projector, OneVisionProjector):
            raise TypeError(
                "projector must be a matching OneVisionProjector or NPZ path"
            )
        if self.projector.linear_1_weight.shape[1] != self.feature_shape[2]:
            raise ValueError(
                f"projector must accept {self.feature_shape[2]}-wide features"
            )
        if family == "onevision" and self.projector.image_newline is None:
            raise ValueError("projector must include the trained image_newline")
        self.model = Path(model)
        self.backend = backend
        self.visual_token_mode = visual_token_mode
        self.backend_path = str(backend_path)
        self.htp_performance_mode = htp_performance_mode
        self.cpu_threads = cpu_threads
        self.allow_spinning = allow_spinning
        self.htp_threads = htp_threads
        self._session_factory = session_factory
        self._sessions = []

    def _load_session(self) -> None:
        if self._sessions:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required; HTP needs the approved QNN-enabled Rubik build"
            ) from exc

        if self.model.suffix == ".json":
            parts = json.loads(self.model.read_text())["parts"]
            models = [
                (
                    self.model.parent / row["file"],
                    tuple(row["input_shape"]),
                    tuple(row["output_shape"]),
                    row.get("input_dtype", "float"),
                    row.get("output_dtype", "float"),
                    row.get("backend", self.backend),
                )
                for row in parts
            ]
            if (
                not models
                or models[0][1] != self.input_shape
                or models[-1][2] != self.feature_shape
            ):
                raise ValueError(
                    "partition chain must preserve the fixed384 I/O contract"
                )
            if any(
                left[2] != right[1] or left[4] != right[3]
                for left, right in pairwise(models)
            ):
                raise ValueError("partition boundary shapes do not match")
            if models[0][3] != "float" or models[-1][4] != "float":
                raise ValueError("partition chain outer tensors must be float32")
            if self.backend != "partitioned" and any(
                row[5] != self.backend for row in models
            ):
                raise ValueError(
                    "mixed backend parts require explicit partitioned mode"
                )
        else:
            models = [
                (
                    self.model,
                    self.input_shape,
                    self.feature_shape,
                    "float",
                    "float",
                    self.backend,
                )
            ]
        sessions = []
        for (
            model,
            input_shape,
            output_shape,
            input_dtype,
            output_dtype,
            backend,
        ) in models:
            if backend not in ("htp", "cpu"):
                raise ValueError(
                    "partitioned mode requires an explicit backend for every part"
                )
            provider = (
                "QNNExecutionProvider" if backend == "htp" else "CPUExecutionProvider"
            )
            if provider not in ort.get_available_providers():
                raise RuntimeError(
                    f"{provider} is unavailable; no fallback is permitted"
                )
            options = ort.SessionOptions()
            if self.allow_spinning is not None:
                spinning = "1" if self.allow_spinning else "0"
                options.add_session_config_entry(
                    "session.intra_op.allow_spinning", spinning
                )
                options.add_session_config_entry(
                    "session.inter_op.allow_spinning", spinning
                )
            provider_options = [{}]
            if backend == "htp":
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                if self.htp_threads is not None:
                    options.intra_op_num_threads = self.htp_threads
                provider_options = [
                    {
                        "backend_path": self.backend_path,
                        "offload_graph_io_quantization": "0",
                        "htp_graph_finalization_optimization_mode": "1",
                    }
                ]
                if self.htp_performance_mode is not None:
                    provider_options[0]["htp_performance_mode"] = (
                        self.htp_performance_mode
                    )
            elif hasattr(options, "intra_op_num_threads"):
                options.intra_op_num_threads = self.cpu_threads
                options.inter_op_num_threads = 1
            session = self._load_part(model, options, provider, provider_options, ort)
            inputs, outputs = session.get_inputs(), session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError(
                    "fixed384 part requires exactly one input and one output"
                )
            for info, shape, dtype, side in (
                (inputs[0], input_shape, input_dtype, "input"),
                (outputs[0], output_shape, output_dtype, "output"),
            ):
                if (
                    dtype not in ("float", "uint16")
                    or info.type != f"tensor({dtype})"
                    or tuple(info.shape) != shape
                ):
                    raise ValueError(
                        f"fixed384 {side} must be {'float32' if dtype == 'float' else dtype} {shape}; got {info.type} {info.shape}"
                    )
            sessions.append((session, inputs[0].name, outputs[0].name))
        self._sessions = sessions

    def _load_part(self, model, options, provider, provider_options, ort):
        factory = self._session_factory or ort.InferenceSession
        # ORT 1.23's Python constructor otherwise retries initialization on CPU.
        # https://github.com/microsoft/onnxruntime/blob/v1.23.0/onnxruntime/python/onnxruntime_inference_collection.py
        session = factory(
            str(model),
            sess_options=options,
            providers=[provider],
            provider_options=provider_options,
            enable_fallback=False,
        )
        session.disable_fallback()
        registered = session.get_providers()
        if provider not in registered or set(registered) - {
            provider,
            "CPUExecutionProvider",
        }:
            raise RuntimeError(
                f"expected {provider}, got session providers {registered}"
            )
        if provider == "QNNExecutionProvider":
            # ORT registers CPU implicitly; this flag prohibits assigning any nodes to it.
            actual_options = session.get_session_options()
            if (
                actual_options.get_session_config_entry(
                    "session.disable_cpu_ep_fallback"
                )
                != "1"
            ):
                raise RuntimeError("HTP session must disable CPU EP fallback")

        return session

    def encode(self, image_path: str | Path) -> QNNVisionResult:
        self._load_session()
        pixels = (
            preprocess_siglip_fixed384(image_path)[0]
            if self.family == "onevision"
            else preprocess_fastvlm_fixed1024(image_path)
        )
        parts_ms = []
        start = perf_counter()
        features = pixels
        for session, input_name, output_name in self._sessions:
            part_start = perf_counter_ns()
            features = session.run([output_name], {input_name: features})[0]
            parts_ms.append((perf_counter_ns() - part_start) / 1e6)
        vision_ms = (perf_counter() - start) * 1e3
        features = np.asarray(features)
        if features.shape != self.feature_shape or features.dtype != np.float32:
            raise ValueError(
                f"visual features must be float32 {self.feature_shape}; "
                f"got {features.dtype} {features.shape}"
            )
        if not np.isfinite(features).all():
            raise ValueError("SigLIP features contain non-finite values")
        embedding, projection_ms = self.projector.project(
            np.ascontiguousarray(features),
            visual_token_mode=self.visual_token_mode,
        )
        tokens = (
            (197 if self.visual_token_mode == "video_pool2" else 730)
            if self.family == "onevision"
            else 256
        )
        hidden = self.projector.linear_2_weight.shape[0]
        if embedding.shape != (tokens, hidden):
            raise ValueError(f"unexpected projected embedding shape: {embedding.shape}")
        label = "htp+cpu" if self.backend == "partitioned" else self.backend
        return QNNVisionResult(
            embedding, vision_ms, projection_ms, label, parts_ms=tuple(parts_ms)
        )

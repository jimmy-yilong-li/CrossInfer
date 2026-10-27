from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from edgeflow_split.crane.bucket_types import BucketSignature
from edgeflow_split.model_spec import ModelFamily, infer_model_family_from_id


# Keep the current flat dataclass compatible while making ownership explicit.
# This is the seam for a future nested SplitConfig decomposition.
SPLIT_CONFIG_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "framework": (
        "model_id",
        "model_family",
        "processor_use_fast",
    ),
    "visual": (
        "visual_backend",
        "ane_surface_producer_enabled",
    ),
    "resources": (
        "ane_lease_pool_capacity",
        "ane_lease_pool_bytes",
    ),
    "text": (
        "text_backend",
        "mlx_model_source",
        "mlx_prefill_step_size",
        "mlx_max_kv_size",
        "ggml_model_path",
        "ggml_sidecar_url",
        "ggml_timeout_s",
        "ggml_max_new_tokens",
        "ggml_transport",
        "ggml_native_surface_path_enabled",
        "ggml_library_path",
        "ggml_shared_memory_dir",
    ),
    "fastvlm": (
        "fastvlm_visual_backend",
        "fastvlm_ane_manifest_path",
        "fastvlm_ane_artifact_root",
        "fastvlm_ane_allow_uncleared_frontend_parity",
    ),
    "llava": (
        "llava_clip_mlpackage_path",
        "llava_projector_weights_path",
        "llava_text_gguf",
    ),
    "qwen25vl": (
        "qwen25vl_use_crane",
        "qwen25vl_bucket_set",
        "qwen25vl_max_bucket",
        "qwen25vl_visual_hf_dir",
    ),
}


@dataclass(frozen=True)
class SplitFrameworkConfig:
    model_id: str
    model_family: ModelFamily | None
    processor_use_fast: bool


@dataclass(frozen=True)
class SplitVisualConfig:
    backend: str
    surface_producer_enabled: bool


@dataclass(frozen=True)
class SplitResourceConfig:
    ane_lease_pool_capacity: int
    ane_lease_pool_bytes: int


@dataclass(frozen=True)
class SplitTextConfig:
    backend: str
    mlx_model_source: str | None
    mlx_prefill_step_size: int | None
    mlx_max_kv_size: int | None
    ggml_model_path: str | None
    ggml_sidecar_url: str
    ggml_timeout_s: float
    ggml_max_new_tokens: int
    ggml_transport: str
    ggml_native_surface_path_enabled: bool
    ggml_library_path: str | None
    ggml_shared_memory_dir: str


@dataclass(frozen=True)
class FastVLMConfig:
    visual_backend: str
    ane_manifest_path: str | None
    ane_artifact_root: str | None
    ane_allow_uncleared_frontend_parity: bool


@dataclass(frozen=True)
class LLaVAConfig:
    clip_mlpackage_path: str | None
    projector_weights_path: str | None
    text_gguf: str | None


@dataclass(frozen=True)
class Qwen25VLConfig:
    use_crane: bool
    bucket_set: Optional[Tuple[BucketSignature, ...]]
    max_bucket: Optional[BucketSignature]
    visual_hf_dir: Optional[Path]


@dataclass(frozen=True)
class SplitConfig:
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    # Explicit native-runtime family. When unset, detected_model_family keeps
    # the historical model_id substring inference for compatibility.
    model_family: ModelFamily | int | None = None
    mlx_model_source: str | None = None
    mlx_prefill_step_size: int | None = 2048
    mlx_max_kv_size: int | None = None
    ggml_model_path: str | None = None
    # HTTP sidecar endpoint used by the binary transport and the frozen shm rollback path.
    ggml_sidecar_url: str = "http://127.0.0.1:8081"
    # Shared timeout for the HTTP sidecar transports only.
    ggml_timeout_s: float = 30.0
    ggml_max_new_tokens: int = 20
    # Transport selector for the GGML text backend. Keep the default stable.
    ggml_transport: str = "binary"
    # Prefer native surface consumption; the configured dylib must still pass
    # the capability probe. Explicit False disables it, but ANE-output
    # descriptors fail closed rather than falling back to CPU readback.
    ggml_native_surface_path_enabled: bool = True
    # Opt-in selector for the ANE surface-backed producer path.
    ane_surface_producer_enabled: bool = False
    # Number of pre-allocated ANE surface leases in the pool.
    ane_lease_pool_capacity: int = 2
    # Bytes per lease. Default sized for
    # Qwen2.5-VL-3B at 384x384 (196 visual tokens x 2048 hidden x
    # float32 = 1,605,632 bytes). Callers that need a different
    # model or resolution should override explicitly.
    ane_lease_pool_bytes: int = 196 * 2048 * 4
    # Optional override for the transport-specific FFI/native dylib.
    ggml_library_path: str | None = None
    # Shared-memory staging directory used only by the frozen shm transport.
    ggml_shared_memory_dir: str = "/tmp/crossinfer_ggml_shm"
    processor_use_fast: bool = False
    visual_backend: str = "torch"
    text_backend: str = "torch"
    # FastVLM visual backend selector; "cached" selects cached embeddings.
    fastvlm_visual_backend: str = "ane_live"
    fastvlm_ane_manifest_path: str | None = None
    fastvlm_ane_artifact_root: str | None = None
    # Safety valve for the opt-in parity harness only. The default live path
    # must fail closed unless the manifest's parity fixture records an explicit
    # full-frontend live parity clearance.
    fastvlm_ane_allow_uncleared_frontend_parity: bool = False
    # LLaVA-1.5-7B paths (optional; required by LLaVAAdapter when invoked)
    llava_clip_mlpackage_path: str | None = None
    llava_projector_weights_path: str | None = None
    llava_text_gguf: str | None = None
    # Optional CRANE bucket configuration; disabled by default.
    qwen25vl_use_crane: bool = False
    qwen25vl_bucket_set: Optional[Tuple[BucketSignature, ...]] = None
    qwen25vl_max_bucket: Optional[BucketSignature] = None
    qwen25vl_visual_hf_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.model_family is None:
            return
        try:
            ModelFamily(self.model_family)
        except ValueError as exc:
            valid = ", ".join(str(int(member)) for member in ModelFamily)
            raise ValueError(
                f"model_family must be one of {{{valid}}}; got {self.model_family!r}"
            ) from exc

    @property
    def detected_model_family(self) -> int:
        """Return the EdgeFlow native runtime model family.

        Returns the integer enum used by the C ABI:
        - 0 (EDGEFLOW_MODEL_FAMILY_QWEN25VL) for Qwen2.5-VL and unknown families
        - 1 (EDGEFLOW_MODEL_FAMILY_FASTVLM)  for FastVLM / LLaVA-FastViTHD style VLMs
        - 2 (EDGEFLOW_MODEL_FAMILY_LLAVA15)  for LLaVA-1.5 / LLaMA-2 text backbones
        """
        if self.model_family is not None:
            return int(ModelFamily(self.model_family))
        return int(infer_model_family_from_id(self.model_id))

    @property
    def framework(self) -> SplitFrameworkConfig:
        family = None if self.model_family is None else ModelFamily(self.model_family)
        return SplitFrameworkConfig(
            model_id=self.model_id,
            model_family=family,
            processor_use_fast=self.processor_use_fast,
        )

    @property
    def visual(self) -> SplitVisualConfig:
        return SplitVisualConfig(
            backend=self.visual_backend,
            surface_producer_enabled=self.ane_surface_producer_enabled,
        )

    @property
    def resources(self) -> SplitResourceConfig:
        return SplitResourceConfig(
            ane_lease_pool_capacity=self.ane_lease_pool_capacity,
            ane_lease_pool_bytes=self.ane_lease_pool_bytes,
        )

    @property
    def text(self) -> SplitTextConfig:
        return SplitTextConfig(
            backend=self.text_backend,
            mlx_model_source=self.mlx_model_source,
            mlx_prefill_step_size=self.mlx_prefill_step_size,
            mlx_max_kv_size=self.mlx_max_kv_size,
            ggml_model_path=self.ggml_model_path,
            ggml_sidecar_url=self.ggml_sidecar_url,
            ggml_timeout_s=self.ggml_timeout_s,
            ggml_max_new_tokens=self.ggml_max_new_tokens,
            ggml_transport=self.ggml_transport,
            ggml_native_surface_path_enabled=self.ggml_native_surface_path_enabled,
            ggml_library_path=self.ggml_library_path,
            ggml_shared_memory_dir=self.ggml_shared_memory_dir,
        )

    @property
    def fastvlm(self) -> FastVLMConfig:
        return FastVLMConfig(
            visual_backend=self.fastvlm_visual_backend,
            ane_manifest_path=self.fastvlm_ane_manifest_path,
            ane_artifact_root=self.fastvlm_ane_artifact_root,
            ane_allow_uncleared_frontend_parity=(
                self.fastvlm_ane_allow_uncleared_frontend_parity
            ),
        )

    @property
    def llava(self) -> LLaVAConfig:
        return LLaVAConfig(
            clip_mlpackage_path=self.llava_clip_mlpackage_path,
            projector_weights_path=self.llava_projector_weights_path,
            text_gguf=self.llava_text_gguf,
        )

    @property
    def qwen25vl(self) -> Qwen25VLConfig:
        return Qwen25VLConfig(
            use_crane=self.qwen25vl_use_crane,
            bucket_set=self.qwen25vl_bucket_set,
            max_bucket=self.qwen25vl_max_bucket,
            visual_hf_dir=self.qwen25vl_visual_hf_dir,
        )

    @property
    def fastvlm_lease_bytes(self) -> int:
        """Lease byte size for FastVLM-0.5B visual descriptor (256 tokens, 896 hidden, fp16)."""
        return 256 * 896 * 2  # 458_752

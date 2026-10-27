from dataclasses import dataclass

_DTYPE_TO_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
}


def derive_payload_bytes(num_tokens: int, hidden_size: int, dtype: str) -> int:
    """Compute byte size for a 2D token embedding payload."""
    if num_tokens < 0:
        raise ValueError("num_tokens must be >= 0")
    if hidden_size < 0:
        raise ValueError("hidden_size must be >= 0")
    try:
        dtype = dtype.lower()
    except AttributeError as exc:
        raise TypeError("dtype must be a string") from exc
    if dtype not in _DTYPE_TO_BYTES:
        raise ValueError(f"unsupported dtype: {dtype}")
    return num_tokens * hidden_size * _DTYPE_TO_BYTES[dtype]


@dataclass
class ImageEmbedsBatch:
    # Keep this backend-flexible for the Apple MLX path. Current transport and
    # PyTorch-backed execution still normalize to torch.Tensor at the boundary.
    image_embeds: object
    image_grid_thw: tuple[int, int, int]
    num_visual_tokens: int
    hidden_size: int
    dtype: str
    source_image_size: tuple[int, int] | None = None
    model_revision: str | None = None


@dataclass(frozen=True)
class TabmAdmissionDecision:
    """Machine-readable admission decision emitted by TABM consume points."""

    policy_applied: bool
    policy: str
    reason: str
    input_handles: int
    output_handles: int
    input_tokens: int
    output_tokens: int
    dropped_handle_ids: tuple[int, ...] = ()
    reduced_handle_tokens: tuple[tuple[int, int], ...] = ()


class QwenPayloadBoundary:
    """Defines the exact Qwen2.5-VL split boundary for payload transport."""

    # Metadata that must cross producer/consumer transport.
    # `image_embeds` is the LLM-ready visual token payload.
    # `image_grid_thw` is required to rebuild Qwen vision position IDs in the text model.
    TRANSPORT_FIELDS = frozenset({"image_embeds", "image_grid_thw"})

    # Metadata reconstructed locally in the text consumer from prompt preparation.
    # `mm_token_type_ids` and `attention_mask` are produced by
    # `full_model_baseline.prepare_multimodal_inputs` and should not be transported.
    CONSUMER_LOCAL_FIELDS = frozenset({"mm_token_type_ids", "attention_mask"})


@dataclass
class TransportHeader:
    """Transport metadata for the Qwen local image-embedding boundary."""

    num_tokens: int
    hidden_size: int
    dtype: str
    payload_bytes: int

    def __post_init__(self) -> None:
        try:
            self.dtype = self.dtype.lower()
        except AttributeError as exc:
            raise TypeError("dtype must be a string") from exc
        expected = derive_payload_bytes(
            num_tokens=self.num_tokens,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
        )
        if self.payload_bytes != expected:
            raise ValueError(f"payload_bytes must be {expected} for provided shape and dtype")

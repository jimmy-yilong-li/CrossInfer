import json
import struct
from collections.abc import Mapping
from json import JSONDecodeError

import numpy as np
import torch

from edgeflow_split.contracts import ImageEmbedsBatch, TransportHeader, derive_payload_bytes

_HEADER_LEN_PACK = ">I"
_HEADER_LEN_BYTES = struct.calcsize(_HEADER_LEN_PACK)


def _normalize_image_embeds(batch: ImageEmbedsBatch) -> torch.Tensor:
    if not torch.is_tensor(batch.image_embeds):
        array = batch.image_embeds
        try:
            import mlx.core as mx

            if isinstance(array, mx.array):
                tensor = torch.from_numpy(np.array(array.astype(mx.float32)))
                return tensor.to(torch.bfloat16 if batch.dtype == "bfloat16" else tensor.dtype)
        except Exception:
            pass
        tensor = torch.from_numpy(np.asarray(array))
        return tensor.to(torch.bfloat16 if batch.dtype == "bfloat16" else tensor.dtype)
    return batch.image_embeds.detach().contiguous()


def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_str(dtype: str) -> torch.dtype:
    try:
        dtype = dtype.lower()
    except AttributeError as exc:
        raise TypeError("dtype must be a string") from exc
    resolved = getattr(torch, dtype, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"unsupported dtype: {dtype}")
    return resolved


def _serialize_header(header: Mapping[str, object]) -> bytes:
    return json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _deserialize_header(payload: bytes) -> tuple[dict[str, object], bytes]:
    if len(payload) < _HEADER_LEN_BYTES:
        raise ValueError("payload is too short to contain a transport header")
    (header_len,) = struct.unpack(_HEADER_LEN_PACK, payload[:_HEADER_LEN_BYTES])
    start = _HEADER_LEN_BYTES
    end = start + header_len
    header_bytes = payload[start:end]
    if len(header_bytes) != header_len:
        raise ValueError("payload header was truncated")
    try:
        header_text = header_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("transport header is not valid UTF-8") from exc
    try:
        header = json.loads(header_text)
    except JSONDecodeError as exc:
        raise ValueError("transport header is not valid JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("transport header must be a JSON object")
    tensor_bytes = payload[end:]
    return header, tensor_bytes


def encode_batch(batch: ImageEmbedsBatch) -> bytes:
    image_embeds = _normalize_image_embeds(batch)
    dtype_str = _dtype_to_str(image_embeds.dtype)
    batch_dtype = str(batch.dtype).replace("torch.", "").lower()
    num_tokens, hidden_size = image_embeds.shape

    if batch_dtype != dtype_str:
        raise ValueError("batch.dtype does not match image_embeds dtype")
    if batch.num_visual_tokens != num_tokens or batch.hidden_size != hidden_size:
        raise ValueError("batch metadata does not match image_embeds shape")

    header = TransportHeader(
        num_tokens=int(num_tokens),
        hidden_size=int(hidden_size),
        dtype=dtype_str,
        payload_bytes=derive_payload_bytes(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            dtype=dtype_str,
        ),
    )

    body = image_embeds.view(torch.uint8).cpu().numpy().tobytes()
    header_payload = _serialize_header(
        {
            "num_tokens": header.num_tokens,
            "hidden_size": header.hidden_size,
            "dtype": header.dtype,
            "payload_bytes": header.payload_bytes,
            "image_grid_thw": batch.image_grid_thw,
        }
    )
    if len(body) != header.payload_bytes:
        raise ValueError("serialized image_embeds payload size does not match header")
    return struct.pack(_HEADER_LEN_PACK, len(header_payload)) + header_payload + body


def decode_batch(payload: bytes) -> ImageEmbedsBatch:
    header_dict, tensor_bytes = _deserialize_header(payload)
    if not isinstance(header_dict, dict):
        raise ValueError("transport header must be a JSON object")

    try:
        num_tokens = int(header_dict["num_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transport header field 'num_tokens' is required and must be an int") from exc

    try:
        hidden_size = int(header_dict["hidden_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transport header field 'hidden_size' is required and must be an int") from exc

    try:
        dtype_str = str(header_dict["dtype"]).lower()
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("transport header field 'dtype' is required and must be a string") from exc

    try:
        payload_bytes = int(header_dict["payload_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("transport header field 'payload_bytes' is required and must be an int") from exc

    try:
        grid_values = header_dict["image_grid_thw"]
    except KeyError as exc:
        raise ValueError("transport header field 'image_grid_thw' is required and must be a 3-tuple") from exc

    if not isinstance(grid_values, (tuple, list)):
        raise ValueError("transport header field 'image_grid_thw' is required and must be a 3-tuple")
    if len(grid_values) != 3:
        raise ValueError("transport header field 'image_grid_thw' must contain exactly 3 values")
    try:
        image_grid_thw = (int(grid_values[0]), int(grid_values[1]), int(grid_values[2]))
    except (TypeError, ValueError) as exc:
        raise ValueError("transport header field 'image_grid_thw' must contain integer values") from exc

    expected_bytes = derive_payload_bytes(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        dtype=dtype_str,
    )
    if payload_bytes != expected_bytes:
        raise ValueError("payload_bytes in header does not match derived payload size")
    if len(tensor_bytes) != expected_bytes:
        raise ValueError("payload byte length does not match payload_bytes")

    tensor = torch.frombuffer(bytearray(tensor_bytes), dtype=torch.uint8).view(_dtype_from_str(dtype_str))
    image_embeds = tensor.reshape((num_tokens, hidden_size))

    return ImageEmbedsBatch(
        image_embeds=image_embeds,
        image_grid_thw=image_grid_thw,
        num_visual_tokens=num_tokens,
        hidden_size=hidden_size,
        dtype=dtype_str,
        source_image_size=None,
        model_revision=None,
    )

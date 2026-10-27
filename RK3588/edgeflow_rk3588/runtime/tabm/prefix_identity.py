"""
TABM Core: tokenized-identity primitives (hash, kind,
compat keys, id derivation, identity-field validation).

Public imports are available through `...tabm.prefix_registry`.
"""
from __future__ import annotations

import enum
import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# FNV-1a 64-bit over tokenized identity (must match
# native/tabm_decoder_worker.cpp's TOK handler exactly).
# ---------------------------------------------------------------------------

_FNV64_OFFSET_BASIS = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3
_FNV64_MASK = 0xFFFFFFFFFFFFFFFF


def fnv1a64_token_ids(token_ids: Iterable[int]) -> str:
    """
    FNV-1a 64-bit hash over each token id's 4 little-endian bytes, taken in
    sequence over `token_ids`. This is the TOK-op identity hash.
    Returns lowercase hex, always exactly 16 chars.

    An empty `token_ids` returns the raw FNV-1a offset basis hex-formatted
    (no bytes mixed in) — this is a well-defined, non-error result (an empty
    tokenized prefix is a legitimate, if unusual, identity).
    """
    h = _FNV64_OFFSET_BASIS
    for token_id in token_ids:
        as_u32 = int(token_id) & 0xFFFFFFFF
        for byte_index in range(4):
            byte = (as_u32 >> (8 * byte_index)) & 0xFF
            h ^= byte
            h = (h * _FNV64_PRIME) & _FNV64_MASK
    return format(h, "016x")


# ---------------------------------------------------------------------------
# PrefixKind
# ---------------------------------------------------------------------------

class PrefixKind(enum.Enum):
    SYSTEM_PROMPT        = "system_prompt"
    CHAT_TEMPLATE_PREFIX = "chat_template_prefix"
    FEWSHOT_PREFIX       = "fewshot_prefix"
    VLM_PRE_IMAGE_PREFIX = "vlm_pre_image_prefix"


# ---------------------------------------------------------------------------
# PrefixCompat — frozen/hashable compatibility identity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrefixCompat:
    """
    Compatibility identity for a prefix entry: model_id, tokenizer_hash,
    chat_template_hash, quant_hash, and backend_id.
    A mismatch on ANY field means the entries are
    for a different (model, tokenizer, chat template, quant, backend)
    combination and must NEVER be treated as a cache hit for each other —
    identity (prefix_id) is always computed WITH these keys folded in (see
    _final_prefix_id() / _provisional_prefix_id()), so a compat mismatch
    mechanically yields a different identity, never a hit.

    Frozen + hashable so it can be used as (part of) a dict key (the
    (raw_text, compat) alias lookup) and cannot be mutated after construction.
    """
    model_id: str
    tokenizer_hash: str
    chat_template_hash: str
    quant_hash: str
    backend_id: str


def _compat_key_string(compat: PrefixCompat) -> str:
    return "|".join((
        compat.model_id, compat.tokenizer_hash, compat.chat_template_hash,
        compat.quant_hash, compat.backend_id,
    ))


def _final_prefix_id(compat: PrefixCompat, token_ids_hash: str) -> str:
    """
    Stable content identity: sha256 hex over compat keys + token_ids_hash,
    not raw text.
    Domain-separated with a "tok:" tag from _provisional_prefix_id() below.
    """
    payload = f"tok:{_compat_key_string(compat)}:{token_ids_hash}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _provisional_prefix_id(compat: PrefixCompat, raw_text: str) -> str:
    """
    PROVISIONAL identity used ONLY as a lookup alias before TOK results
    arrive (see PrefixRegistry.set_token_identity()).
    Domain-separated ("raw:" vs "tok:") from
    _final_prefix_id() so a provisional id can never be mistaken for (or
    collide in the same namespace as) a finalized, tokenized identity.
    """
    payload = f"raw:{_compat_key_string(compat)}:{raw_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_TOKEN_IDS_HASH_RE = re.compile(r"[0-9a-f]{16}")


def _validate_token_identity(
    token_count: Optional[int], token_ids_hash: Optional[str]
) -> None:
    """
    Shared identity-field validation for PrefixRegistry.register() and
    set_token_identity(): both are entry points into the
    tokenized identity these methods key on, so a garbage token_count/
    token_ids_hash must be rejected the same way at either one. `None`
    values are left unvalidated here -- register()'s identity args are
    optional (not yet known at provisional-registration time); callers that
    must supply a value (set_token_identity()) get None rejected by their
    own signature/call site, not by this helper.

    token_count: must be a real int -- NOT a bool (bool is an int subclass
    in Python, so a plain `isinstance(x, int)` check would silently accept
    True/False as 1/0) -- and >= 0.
    token_ids_hash: must be exactly 16 lowercase hex chars, the shape every
    fnv1a64_token_ids() output has by construction.

    Raises ValueError naming the offending field.
    """
    if token_count is not None:
        if isinstance(token_count, bool) or not isinstance(token_count, int):
            raise ValueError(f"token_count must be an int, got {token_count!r}")
        if token_count < 0:
            raise ValueError(f"token_count must be >= 0, got {token_count!r}")
    if token_ids_hash is not None and not _TOKEN_IDS_HASH_RE.fullmatch(token_ids_hash):
        raise ValueError(
            f"token_ids_hash must be exactly 16 lowercase hex chars, got {token_ids_hash!r}"
        )

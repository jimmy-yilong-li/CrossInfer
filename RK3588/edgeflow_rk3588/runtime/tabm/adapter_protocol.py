"""
TABM Core — Decoder Adapter wire protocol: the STATELESS layer of the
tabm_decoder_worker line protocol (constants + pure helpers), shared by
MNNAdapterWorkerBackend (adapter_worker.py).

Stateful request and lifecycle handling lives in MNNAdapterWorkerBackend.

Wire shapes (see native/tabm_decoder_worker.cpp):
  worker -> "READY <load_s>" | "FATAL <err>" on startup
  worker -> "RES <id> <ok:0|1> <n_prompt> <n_decode> <prefill_us> <decode_us>
             <copied_bytes> <ids_csv|-> <err_b64|->"          (decode requests)
  worker -> "RES <id> TOK <token_count> <token_ids_hash>"     (TOK requests)
  host   -> "REQ <id> prefix <max_tokens> <name> <b64prefix> <b64suffix>"
             (cross-request prefix-cache request; both segments
             tokenized separately + UN-TEMPLATED in the worker — TOK identity
             discipline)
  worker -> "PFXINFO <id> hit=<0|1> c1_prefill_us=<..> c2_prefill_us=<..>
             n_prefix=<..> n_suffix=<..>"  followed by a standard 10-field RES.
             PFXINFO is deliberately NOT "RES "-prefixed: the generic decode
             scanner's prefix filter discards it as junk (backward-harmless);
             only run_prefix_request()'s dedicated scan consumes it.
  worker -> "VINFO <id> nbytes=<..> crc32=<8-lower-hex> crc_us=<..>"
             before a standard RES for embeds/VLM requests. This is an optional
             consumer-boundary receipt; older workers remain compatible.
  worker -> "TINFO <id> n_pre=<..> n_vis=<..> n_post=<..> reset_us=<..>
             prepare_us=<..> stage_us=<..> generate_us=<..> native_wall_us=<..>"
             before a successful VLM RES when EDGEFLOW_TABM_TRACE=1.
             New workers optionally append first_token_mono_us/native_ttft_us;
             both fields use the first token-output flush, not prefill completion.
"""
from __future__ import annotations

import base64
import os
import re
from numbers import Integral

from RK3588.edgeflow_rk3588.runtime.tabm.lane import RequestValidationError, WorkItem

_READY_PREFIX = "READY "
_FATAL_PREFIX = "FATAL "
_RES_PREFIX = "RES "
# "RES" + id, ok, n_prompt, n_decode, prefill_us, decode_us, copied_bytes, ids_csv, err
_RES_NUM_FIELDS = 10
# "RES" + id, "TOK", token_count, token_ids_hash — see tokenize_prefix().
_TOK_TAG = "TOK"
_TOK_RES_NUM_FIELDS = 5

# CTX is an explicit retained-context path, not an automatic WorkItem mode.
RETAINED_MAX_POSITIONS = 8192
RETAINED_MAX_DECODE = 128
RETAINED_MAX_REQUESTS = 1024
RETAINED_MAX_BYTES = 64 * 1024 * 1024
RETAINED_MAX_TEXT_BYTES = 32768
RETAINED_MAX_SNAPSHOT_PATH_BYTES = 4096
_RETAINED_STREAM_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")
_RETAINED_LOGITS_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_RETAINED_SHM_RE = re.compile(r"/?[A-Za-z0-9._-]{1,240}")

# Sentinel distinguishing "field genuinely absent" from "field present but
# None" when duck-typing embeds_desc (dict OR object) — see _embeds_field().
_MISSING = object()


def _embeds_field(desc: object, name: str):
    """
    Duck-typed field lookup for a ``WorkItem.embeds_desc``: works for BOTH a
    dict (e.g. ``dataclasses.asdict(SlotBufferDescriptor(...))``, the current
    harness) AND a raw object with matching attributes (e.g. a
    ``SlotBufferDescriptor`` passed through unconverted).

    Raises ``ValueError`` naming the missing field — never a bare
    KeyError/AttributeError — so a malformed descriptor fails loudly and
    specifically instead of crashing with an opaque exception deeper in
    ``submit()``.
    """
    value = (
        desc.get(name, _MISSING)
        if isinstance(desc, dict)
        else getattr(desc, name, _MISSING)
    )
    if value is _MISSING:
        raise RequestValidationError(
            "MNNAdapterWorkerBackend.submit: embeds_desc is missing required "
            f"field {name!r} (desc={desc!r})"
        )
    return value


# Filename-stem discipline for prefix cache names:
# safe charset AND an explicit ".." ban (chars are individually legal).
_PREFIX_NAME_RE = re.compile(r"(?!.*\.\.)[A-Za-z0-9._-]+")

_PFXINFO_PREFIX = "PFXINFO "
_VINFO_PREFIX = "VINFO "
_TINFO_PREFIX = "TINFO "
_TINFO_KEYS = (
    "n_pre", "n_vis", "n_post", "reset_us", "prepare_us", "stage_us",
    "generate_us", "native_wall_us",
)
_TINFO_TTFT_KEYS = ("first_token_mono_us", "native_ttft_us")


def _validate_request_id(request_id: int) -> None:
    # The worker parses signed long long; negative warmup IDs are intentional.
    if (not isinstance(request_id, Integral) or isinstance(request_id, bool)
            or not -2**63 <= request_id < 2**63):
        raise RequestValidationError("request_id must be a signed int64 integer")


def _retained_integer(value, label, minimum, maximum):
    if (not isinstance(value, Integral) or isinstance(value, bool)
            or not minimum <= value <= maximum):
        raise RequestValidationError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _retained_text(text, label):
    b64 = _request_text_b64(text, label)
    if "\0" in text or len(text.encode("utf-8")) > RETAINED_MAX_TEXT_BYTES:
        raise RequestValidationError(f"{label} must be NUL-free and <= 32768 UTF-8 bytes")
    return b64


def build_retained_req(
    request_id: int, op: str, stream_id: str, version: int, *,
    prefix_text: str | None = None, embeds_desc: object = None,
    suffix_text: str | None = None, max_tokens: int = 8,
    logits_name: str | None = None,
    snapshot_path: str | os.PathLike[str] | None = None,
    content_digest: str | None = None,
    cached_source: tuple[str, int, str] | None = None,
    export_prefix: tuple[int, int] | None = None,
) -> str:
    """Build CTX before I/O. Append compares version and publishes version+1.

    Reset stores raw fixed-prefix tokens for joint ingestion with the first
    frame. Query uses raw private suffix tokens (no implicit chat template).
    Only the current root exists; reset and import can replace its identity.
    """
    _validate_request_id(request_id)
    if cached_source is not None and op != "cache_import":
        raise RequestValidationError("cached_source is only valid for cache_import")
    if export_prefix is not None and op != "export":
        raise RequestValidationError("export_prefix is only valid for export")
    if not isinstance(stream_id, str) or stream_id == "-" or not _RETAINED_STREAM_RE.fullmatch(stream_id):
        raise RequestValidationError("stream_id must match [A-Za-z0-9_-]{1,128}")
    version = _retained_integer(version, "version", 0, 2**63 - 1)
    fields = f"CTX {request_id} {op} {stream_id} {version}"
    if op == "reset":
        if any(value is not None for value in (embeds_desc, suffix_text, logits_name, snapshot_path, content_digest)):
            raise RequestValidationError("reset accepts only prefix_text")
        fields += " " + _retained_text(prefix_text, "prefix_text")
    elif op == "append":
        if any(value is not None for value in (prefix_text, suffix_text, logits_name, snapshot_path, content_digest)):
            raise RequestValidationError("append accepts only embeds_desc")
        if version == 2**63 - 1:
            raise RequestValidationError("append version is exhausted; reset required")
        kind = _embeds_field(embeds_desc, "buffer_kind")
        dtype = _embeds_field(embeds_desc, "dtype")
        shape = _embeds_field(embeds_desc, "shape")
        if kind != "shm" or dtype != "float32":
            raise RequestValidationError("retained append requires a float32 TABM shm descriptor")
        if not isinstance(shape, (tuple, list)) or len(shape) != 2:
            raise RequestValidationError("retained shape must be exactly [n_vis, hidden]")
        n_vis = _retained_integer(shape[0], "n_vis", 1, RETAINED_MAX_POSITIONS)
        hidden = _retained_integer(shape[1], "hidden", 1, 16384)
        nbytes = _retained_integer(_embeds_field(embeds_desc, "nbytes"), "nbytes", 1, RETAINED_MAX_BYTES)
        if nbytes != n_vis * hidden * 4:
            raise RequestValidationError("nbytes does not match float32 shape")
        offset = _retained_integer(_embeds_field(embeds_desc, "offset"), "offset", 0, 2**63 - 1 - nbytes)
        if offset % 4:
            raise RequestValidationError("offset must be float32-aligned")
        capacity = (embeds_desc.get("capacity_nbytes") if isinstance(embeds_desc, dict)
                    else getattr(embeds_desc, "capacity_nbytes", None))
        if capacity is not None:
            _retained_integer(capacity, "capacity_nbytes", nbytes, 2**63 - 1 - offset)
        shm_name = _embeds_field(embeds_desc, "shm_name")
        if (not isinstance(shm_name, str) or not _RETAINED_SHM_RE.fullmatch(shm_name)
                or shm_name in (".", "..", "/.", "/..")):
            raise RequestValidationError("shm_name must be one bounded POSIX shm name")
        fields += f" {shm_name} {offset} {n_vis} {hidden}"
    elif op == "query":
        if any(value is not None for value in (prefix_text, embeds_desc, snapshot_path, content_digest)):
            raise RequestValidationError("query accepts only a private suffix")
        max_tokens = _retained_integer(max_tokens, "max_tokens", 1, RETAINED_MAX_DECODE)
        if logits_name is not None and (
            not isinstance(logits_name, str) or logits_name == "-" or not _RETAINED_LOGITS_RE.fullmatch(logits_name)
        ):
            raise RequestValidationError("logits_name must match [A-Za-z0-9_-]{1,64}")
        fields += f" {max_tokens} {_retained_text(suffix_text, 'suffix_text')} {logits_name or '-'}"
    elif op in ("export", "import", "cache_import"):
        if any(value is not None for value in (prefix_text, embeds_desc, suffix_text, logits_name)):
            raise RequestValidationError(f"{op} accepts only snapshot_path and content_digest")
        if not isinstance(content_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", content_digest):
            raise RequestValidationError("content_digest must be a 64-character lowercase hex identity")
        try:
            path = os.fspath(snapshot_path)
        except TypeError as exc:
            raise RequestValidationError("snapshot_path must be an absolute non-empty UTF-8 path") from exc
        encoded = _retained_text(path, "snapshot_path")
        if not os.path.isabs(path) or len(path.encode("utf-8")) > RETAINED_MAX_SNAPSHOT_PATH_BYTES:
            raise RequestValidationError("snapshot_path must be absolute and <= 4096 UTF-8 bytes")
        fields += f" {encoded} {content_digest}"
        if export_prefix is not None:
            if not isinstance(export_prefix, tuple) or len(export_prefix) != 2:
                raise RequestValidationError("export_prefix requires version and position")
            target = _retained_integer(export_prefix[0], "export version", 1, version)
            end = _retained_integer(export_prefix[1], "export position", 1, RETAINED_MAX_POSITIONS)
            fields += f" {target} {end}"
        if op == "cache_import":
            if not isinstance(cached_source, tuple) or len(cached_source) != 3:
                raise RequestValidationError("cache_import requires source stream/version/digest")
            source_stream, source_version, source_digest = cached_source
            if not isinstance(source_stream, str) or not _RETAINED_STREAM_RE.fullmatch(source_stream):
                raise RequestValidationError("invalid cached source stream")
            source_version = _retained_integer(source_version, "source_version", 1, 2**63 - 1)
            if source_version != version or not isinstance(source_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", source_digest):
                raise RequestValidationError("cached source version/partition or digest mismatch")
            fields += f" {source_stream} {source_version} {source_digest}"
    elif op == "release":
        if any(value is not None for value in (prefix_text, embeds_desc, suffix_text, logits_name, snapshot_path, content_digest)):
            raise RequestValidationError("release accepts only current stream/version")
    else:
        raise RequestValidationError("retained op must be reset, append, query, export, import, or release")
    return fields + "\n"


def parse_retained_res(line: str) -> dict:
    """Strict terminal RES/CTX metadata, never a full-logits stdout payload."""
    fields = line.split()
    if len(fields) != 29 or fields[0] != "RES" or fields[2] != "CTX":
        raise ValueError("malformed CTX reply")
    keys = ("request_id", "version", "root_end", "logical_position", "physical_history",
            "pending_remove", "prefix_tokens", "n_input", "prefill_us", "decode_us",
            "copied_bytes", "first_logits_count", "root_before", "reset_us",
            "prefill_wall_us", "decode_wall_us", "ttft_us", "request_wall_us")
    indices = (1, 6, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 23, 24, 25, 26, 27, 28)
    try:
        result = dict(zip(keys, (int(fields[i]) for i in indices)))
        for key, index in (("ok", 3), ("active", 7), ("valid", 8), ("rollback_verified", 14)):
            if fields[index] not in ("0", "1"):
                raise ValueError("invalid CTX boolean")
            result[key] = fields[index] == "1"
        ids = [] if fields[19] == "-" else [int(token) for token in fields[19].split(",")]
        error = None if fields[22] == "-" else base64.b64decode(fields[22], validate=True).decode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("malformed CTX fields") from exc
    _validate_request_id(result["request_id"])
    op, stream, filename = fields[4], fields[5], fields[21]
    if any(result[key] < 0 for key in keys if key not in ("request_id", "ttft_us")):
        raise ValueError("negative CTX field")
    if result["version"] >= 2**63 or result["ttft_us"] < -1:
        raise ValueError("invalid CTX version or TTFT")
    # Inactive retained roots do not own ordinary REQ history. MNN also defers
    # physical deletion after reset; legacy REQ has a larger context contract.
    retired_legacy = (
        result["root_end"] == result["logical_position"] == 0
        and result["physical_history"] == result["pending_remove"]
        and ((result["ok"] and op in ("reset", "release") and result["n_input"] == 0)
             or (not result["ok"] and result["valid"])
             or (not result["ok"] and not result["active"] and stream == "-"
                 and op == "reset" and result["prefix_tokens"] == result["n_input"] == 0))
    )
    physical_max = 2**31 - 1 if retired_legacy else RETAINED_MAX_POSITIONS
    if (any(result[key] > RETAINED_MAX_POSITIONS for key in (
            "root_before", "root_end", "logical_position", "prefix_tokens", "n_input"))
            or any(result[key] > physical_max for key in ("physical_history", "pending_remove"))):
        raise ValueError("CTX positions exceed 8192")
    if (len(ids) > RETAINED_MAX_DECODE or any(token < 0 or token >= 2**31 for token in ids)
            or result["first_logits_count"] > 1000000
            or result["copied_bytes"] > RETAINED_MAX_BYTES):
        raise ValueError("CTX output exceeds bounds")
    if op not in ("reset", "append", "query", "export", "import", "cache_import", "release"):
        raise ValueError("invalid CTX operation")
    if stream != "-" and not _RETAINED_STREAM_RE.fullmatch(stream):
        raise ValueError("invalid CTX stream")
    if result["active"] and stream == "-":
        raise ValueError("active CTX root requires a stream")
    if filename != "-" and (
        not filename.endswith(".f32") or not _RETAINED_LOGITS_RE.fullmatch(filename[:-4])
        or not result["first_logits_count"]
    ):
        raise ValueError("invalid CTX logits file")
    if result["valid"] and (
        result["logical_position"] != result["root_end"]
        or result["physical_history"] - result["pending_remove"] != result["root_end"]
    ):
        raise ValueError("CTX rollback/root positions disagree")
    if result["ok"]:
        if error is not None or not result["valid"]:
            raise ValueError("successful CTX reply is invalid")
        if op == "query" and (
            not ids or not result["rollback_verified"] or result["ttft_us"] < 0
            or not result["first_logits_count"] or not result["n_input"]
        ):
            raise ValueError("successful CTX query lacks completed output/rollback")
        if op != "query" and ids:
            raise ValueError("CTX control operation generated tokens")
        if op == "export" and (
            result["root_end"] != result["root_before"] or result["n_input"] != 0
        ):
            raise ValueError("CTX export changed root or consumed input")
        if op in ("import", "cache_import") and (
            result["n_input"] != 0 or not result["root_end"] or not result["prefix_tokens"]
            or result["physical_history"] != result["root_end"]
        ):
            raise ValueError("CTX import did not replace root metadata without input")
    elif not error:
        raise ValueError("failed CTX reply lacks error")
    result.update(op=op, stream_id=None if stream == "-" else stream, output_ids=ids,
                  first_logits_file=None if filename == "-" else filename, error=error)
    if result["ttft_us"] == -1:
        result["ttft_us"] = None
    return result


def build_prefix_req(
    request_id: int,
    max_tokens: int,
    name: str,
    prefix_text: str,
    suffix_text: str,
) -> str:
    """
    Format the prefix-mode request line (module docstring wire shape).
    ``name`` is the setPrefixCacheFile stem (MNNCacheFileProvider.handle_for
    or an explicitly attached handle); segments are b64-encoded whole.
    Raises ValueError on empty name/segments — the worker would reject them
    anyway, fail loud on the host side first.
    """
    _validate_request_id(request_id)
    # The name becomes a FILENAME STEM inside the
    # worker's prefix_cache_path (MNNFilePathConcat) — path-shaped values
    # ('/', '..') would escape that directory. attach_handle()'d custom
    # handles flow through here too, so this is the single choke point.
    if not isinstance(name, str) or not _PREFIX_NAME_RE.fullmatch(name):
        raise RequestValidationError(
            "prefix cache name must match [A-Za-z0-9._-]+ with no '/' and "
            f"no '..' (filename stem, not a path): {name!r}"
        )
    b64p = _request_text_b64(prefix_text, "prefix_text")
    b64s = _request_text_b64(suffix_text, "suffix_text")
    return f"REQ {request_id} prefix {max_tokens} {name} {b64p} {b64s}\n"


def _request_text_b64(text: str, label: str) -> str:
    if not isinstance(text, str) or not text:
        raise RequestValidationError(f"{label} must be a non-empty string")
    try:
        return base64.b64encode(text.encode("utf-8")).decode("ascii")
    except UnicodeEncodeError as exc:
        raise RequestValidationError(f"{label} must be valid UTF-8 text") from exc


def build_decode_req(work: WorkItem) -> tuple[str, bool]:
    """Validate and format a decode request before any worker I/O or mutation."""
    _validate_request_id(work.request_id)
    max_tokens = work.max_decode_tokens
    if (
        not isinstance(max_tokens, Integral)
        or isinstance(max_tokens, bool)
        or not 0 < max_tokens <= 2**31 - 1
    ):
        raise RequestValidationError(
            "MNNAdapterWorkerBackend.submit: max_decode_tokens must be > 0 "
            f"and fit int32, got {max_tokens!r}"
        )
    prefix = (work.prefix_name, work.prefix_text, work.suffix_text)
    if any(field is not None for field in prefix):
        if not all(prefix):
            raise RequestValidationError(
                "MNNAdapterWorkerBackend.submit: prefix mode needs ALL of "
                "prefix_name/prefix_text/suffix_text (got a partial set)"
            )
        if work.embeds_desc is not None:
            raise RequestValidationError(
                "MNNAdapterWorkerBackend.submit: prefix mode is text-only "
                "today -- embeds_desc must be None"
            )
        return build_prefix_req(work.request_id, max_tokens, *prefix), False

    desc = work.embeds_desc
    prompt = work.prompt_text or ""
    if desc is None:
        if not prompt:
            raise RequestValidationError(
                "MNNAdapterWorkerBackend.submit: text mode requires a "
                "non-empty prompt_text (no embeds_desc was provided)"
            )
        b64 = _request_text_b64(prompt, "prompt_text")
        return f"REQ {work.request_id} text {max_tokens} {b64}\n", False
    buffer_kind = (
        desc.get("buffer_kind") if isinstance(desc, dict)
        else getattr(desc, "buffer_kind", None)
    )
    if buffer_kind != "shm":
        raise RequestValidationError(
            "MNNAdapterWorkerBackend.submit: embeds_desc.buffer_kind "
            f"must be exactly 'shm'; got {buffer_kind!r}"
        )
    shape = _embeds_field(desc, "shape")
    try:
        n_vis, hidden = shape[0], shape[1]
    except (TypeError, IndexError, KeyError):
        raise RequestValidationError(
            "MNNAdapterWorkerBackend.submit: embeds_desc.shape must "
            f"have at least 2 dims (n_vis, hidden); got {shape!r}"
        ) from None
    shm_name = _embeds_field(desc, "shm_name")
    offset = _embeds_field(desc, "offset")
    if (
        not isinstance(shm_name, str) or not shm_name
        or any(character.isspace() or character == "\0" for character in shm_name)
    ):
        raise RequestValidationError("shm_name must be a non-empty protocol token")
    for label, value, minimum in (("offset", offset, 0), ("n_vis", n_vis, 1), ("hidden", hidden, 1)):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
            raise RequestValidationError(f"{label} must be an integer >= {minimum}")
    mode = "vlm" if prompt else "embeds"
    line = f"REQ {work.request_id} {mode} {max_tokens} {shm_name} {offset} {n_vis} {hidden}"
    if prompt:
        line += " " + _request_text_b64(prompt, "prompt_text")
    return line + "\n", bool(prompt)


def parse_res_numbers(fields: list[str]) -> tuple[int, int, int, int, int, list[int]]:
    """Validate a standard RES payload; request-id correlation is caller-owned."""
    if len(fields) != _RES_NUM_FIELDS or fields[0] != "RES":
        raise ValueError("malformed RES line")
    if fields[2] not in ("0", "1"):
        raise ValueError("malformed RES status: expected 0 or 1")
    try:
        n_prompt, n_decode, prefill_us, decode_us, copied_bytes = map(int, fields[3:8])
        output_ids = [] if fields[8] == "-" else [int(x) for x in fields[8].split(",")]
    except ValueError as exc:
        raise ValueError("malformed RES numeric fields") from exc
    if min(n_prompt, n_decode, prefill_us, decode_us, copied_bytes) < 0 or any(
        token_id < 0 for token_id in output_ids
    ):
        raise ValueError("malformed RES numeric fields: negative value")
    if n_decode != len(output_ids):
        raise ValueError("malformed RES token count: n_decode does not match output_ids")
    # Error replies may have zero counts, partial output, or engine timings.
    # Do not impose success-only requirements on the native ok=0 convention.
    return n_prompt, n_decode, prefill_us, decode_us, copied_bytes, output_ids


def parse_pfxinfo(line: str) -> "dict | None":
    """
    Parse one "PFXINFO <id> k=v ..." line into
    {"id": int, "hit": bool, "c1_prefill_us": int, "c2_prefill_us": int,
     "n_prefix": int, "n_suffix": int}. Returns None if the line is not
    PFXINFO-shaped or any field is malformed — callers treat None as
    "not the info line" (same junk-tolerance as the RES prefix filter).
    """
    if not line.startswith(_PFXINFO_PREFIX):
        return None
    fields = line.split()
    if len(fields) != 7:
        return None
    try:
        out = {"id": int(fields[1])}
        kv = dict(f.split("=", 1) for f in fields[2:])
        if kv["hit"] not in ("0", "1"):
            return None   # hit must be exactly 0|1
        out["hit"] = kv["hit"] == "1"
        for k in ("c1_prefill_us", "c2_prefill_us", "n_prefix", "n_suffix"):
            out[k] = int(kv[k])
            if out[k] < 0:
                return None
    except (ValueError, KeyError):
        return None
    return out


def parse_vinfo(line: str) -> "dict | None":
    """Parse a strict visual-payload consumer receipt, or return ``None``."""
    if not line.startswith(_VINFO_PREFIX):
        return None
    fields = line.split()
    if len(fields) != 5:
        return None
    try:
        request_id = int(fields[1])
        kv = dict(field.split("=", 1) for field in fields[2:])
        nbytes = int(kv["nbytes"])
        crc_us = int(kv["crc_us"])
        crc32 = kv["crc32"]
    except (ValueError, KeyError):
        return None
    if nbytes < 0 or crc_us < 0 or re.fullmatch(r"[0-9a-f]{8}", crc32) is None:
        return None
    return {
        "id": request_id,
        "nbytes": nbytes,
        "crc32": crc32,
        "crc_us": crc_us,
    }


def parse_tinfo(line: str) -> "dict | None":
    """Parse a complete, nonoverlapping native VLM wall trace."""
    if not line.startswith(_TINFO_PREFIX):
        return None
    fields = line.split()
    if len(fields) not in (len(_TINFO_KEYS) + 2, len(_TINFO_KEYS) + len(_TINFO_TTFT_KEYS) + 2):
        return None
    try:
        pairs = [field.split("=", 1) for field in fields[2:]]
        if any(len(pair) != 2 for pair in pairs):
            return None
        values = dict(pairs)
        keys = _TINFO_KEYS if len(fields) == len(_TINFO_KEYS) + 2 else _TINFO_KEYS + _TINFO_TTFT_KEYS
        if len(values) != len(keys) or set(values) != set(keys):
            return None
        out = {key: int(values[key]) for key in keys}
        out["id"] = int(fields[1])
    except ValueError:
        return None
    if min(out[key] for key in _TINFO_KEYS) < 0:
        return None
    if "first_token_mono_us" in out and (
        out["first_token_mono_us"] <= 0 or not 0 <= out["native_ttft_us"] <= out["native_wall_us"]
    ):
        return None
    occupied = sum(
        out[key] for key in ("reset_us", "prepare_us", "stage_us", "generate_us")
    )
    if occupied > out["native_wall_us"]:
        return None
    out["native_unattributed_us"] = out["native_wall_us"] - occupied
    return out


def _decode_generic_error(line: str) -> str:
    """
    Best-effort extraction of the base64-encoded error field from a GENERIC
    (non-TOK) "RES ..." line — used by tokenize_prefix() when a TOK request
    gets the worker's per-request error convention (10-field RES) instead of
    a TOK-shaped reply (see native/tabm_decoder_worker.cpp's TOK handler).
    Falls back to the raw line if it doesn't parse as expected — never
    raises on its own.
    """
    fields = line.split()
    if len(fields) == _RES_NUM_FIELDS and fields[-1] != "-":
        try:
            return base64.b64decode(fields[-1]).decode("utf-8", "replace")
        except Exception:
            pass
    return line

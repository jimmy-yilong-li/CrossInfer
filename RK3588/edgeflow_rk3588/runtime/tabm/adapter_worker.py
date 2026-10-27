"""
TABM Core: Python side of the in-process MNN Decoder Adapter.

``MNNAdapterWorkerBackend`` drives the ``tabm_decoder_worker`` native binary
(``native/tabm_decoder_worker.cpp``) as a ``WorkerBackend``.
The worker loads the MNN Llm ONCE and serves
requests over a line protocol on stdin/stdout; this class is the Python
transport that speaks that protocol so TABM's existing lane/dispatch can use
the in-process decoder with EXACT per-request metrics (not the approximate
wall-clock/whitespace-token metrics of ``MNNDemoWorkerBackend``).

Protocol (EXACT — implement, do not change; see native/tabm_decoder_worker.cpp)
--------------------------------------------------------------------------
Startup: the worker prints logs (MNN pollutes stdout!) and eventually a line
starting ``READY <load_seconds>``.  A line starting ``FATAL `` means startup
failed. Lines starting with ``READY ``, ``RES ``, ``VINFO ``, ``TINFO ``,
``CTXCAP ``, ``CTXTIME ``, or ``FATAL `` are protocol lines; everything else
is prefix-filtered (ignored).

Requests (single in-flight)::

    REQ <id> text   <max_tokens> <base64(prompt_utf8)>
    REQ <id> embeds <max_tokens> <shm_name> <offset> <n_vis> <hidden>
    REQ <id> vlm    <max_tokens> <shm_name> <offset> <n_vis> <hidden> <base64(prompt_utf8)>
    EXIT

Mode routing (``MNNAdapterWorkerBackend.submit()``, decided from
``WorkItem.embeds_desc`` / ``WorkItem.prompt_text`` — see ``submit()``'s
docstring for the exact rule)::

    embeds_desc present + prompt_text non-empty -> vlm     REQ
    embeds_desc present + prompt_text empty     -> embeds  REQ
    embeds_desc None                            -> text    REQ

Note the ``vlm`` REQ's field order: it is the ``embeds`` descriptor fields
(shm_name, offset, n_vis, hidden) with ``base64(prompt_utf8)`` appended as
the LAST token on the line — never leading, never interleaved.

Response::

    RES <id> <ok:0|1> <n_prompt> <n_decode> <prefill_us> <decode_us> \
<copied_bytes> <ids_csv|-> <err_b64|->

TOK op (prefix-registry identity; see prefix_registry.py)::

    TOK <id> <b64text>                          (sent by tokenize_prefix())
    RES <id> TOK <token_count> <token_ids_hash>  (well-formed reply)

``tokenize_prefix()`` is a SEPARATE, synchronous one-shot helper — not part
of the submit()/poll() WorkerBackend decode path — meant for whatever
populates a PrefixRegistry, not the hot decode loop. A malformed TOK line
gets the worker's GENERIC per-request error RES shape (the same one shown
above for REQ), which ``tokenize_prefix()`` surfaces as a RuntimeError —
NOT a process FATAL (native/tabm_decoder_worker.cpp's TOK handler).

Honesty notes
-------------
- ``service_time`` reported here is ``(prefill_us + decode_us) / 1e6`` — the
  EXACT engine time measured inside MNN's ``LlmContext`` for this request. It
  is NOT a wall-clock measurement: wall time additionally includes this
  protocol's overhead (line building/base64, the pipe round trip, process
  scheduling delay).  If wall time is needed, time the ``submit()``..``poll()``
  span externally; the raw engine components are always available via
  ``WorkResult.extra["prefill_us"]`` / ``extra["decode_us"]``.
- ``output_ids`` are the EXACT tokenizer ids returned by the worker (parsed
  from ``ids_csv``) — not an approximation.
- Set ``EDGEFLOW_TABM_TRACE=1`` for an opt-in VLM worker trace. Successful
  requests then require one matching ``TINFO`` before ``RES`` and expose
  nonoverlapping native spans plus submit-to-observed-RES wall time in
  ``WorkResult.extra["worker_trace"]``. The existing engine ``service_time``
  and dispatch policy are unchanged; rejected requests need no trace.
- The synthesized failure paths below (worker died mid-request / malformed or
  mismatched RES) have no real engine timing, so their ``service_time`` is the
  wall-clock elapsed since ``submit()`` — the best available number, but it is
  wall time, not engine time.

Board-coupled: needs the ``tabm_decoder_worker`` native binary + a model
config whose relative paths resolve from the config's directory (mirrors
``llm_demo``'s layout).  This class itself is pure-Python (subprocess) and
import-safe anywhere; it only requires the binary at construction time.
"""
from __future__ import annotations

import base64
import math
import os
import re
import select
import subprocess
import time
from typing import Callable, Optional

from RK3588.edgeflow_rk3588.runtime.tabm.admission import _finite_number
from RK3588.edgeflow_rk3588.runtime.tabm.adapter_protocol import (
    _FATAL_PREFIX,
    _READY_PREFIX,
    _RES_NUM_FIELDS,
    _RES_PREFIX,
    _TOK_RES_NUM_FIELDS,
    _TOK_TAG,
    _decode_generic_error,
    _embeds_field,
    RETAINED_MAX_POSITIONS,
    RETAINED_MAX_REQUESTS,
    build_decode_req,
    build_prefix_req,
    build_retained_req,
    parse_pfxinfo,
    parse_res_numbers,
    parse_retained_res,
    parse_tinfo,
    parse_vinfo,
)
from RK3588.edgeflow_rk3588.runtime.tabm.lane import (
    RequestValidationError,
    WorkerBackend,
    WorkerTerminationError,
    WorkItem,
    WorkResult,
)
from RK3588.edgeflow_rk3588.runtime.tabm.worker_output import WorkerOutput, WorkerOutputError

_MNN_GPU_TOTAL_RE = re.compile(r"^total kernel time =\s*(\d+)\s+us,")
_CTX_TIME_FIELDS = ("request_start_s", "prefill_start_s", "prefill_end_s",
                    "first_token_ready_s", "finish_s")


class _AdapterWorkerOutput(WorkerOutput):
    # Extend the existing continuous, bounded reader without changing other
    # backends' protocol filtering or mutating its shared class-level prefixes.
    _PROTOCOL = WorkerOutput._PROTOCOL + (b"CTXCAP ", b"CTXTIME ", b"CTXKV ")


def _parse_retained_timing(line: str, expected_id: int, op: str) -> dict[str, float]:
    fields = line.split()
    if len(fields) != 8:
        raise ValueError("malformed CTXTIME field count")
    try:
        request_id = int(fields[1])
        values = [float(value) for value in fields[3:]]
    except ValueError as exc:
        raise ValueError("malformed CTXTIME numbers") from exc
    if request_id != expected_id or fields[2] != op:
        raise ValueError("mismatched CTXTIME request id/operation")
    if (not all(math.isfinite(value) for value in values)
            or values[0] < 0 or values[-1] < 0
            or any(value < 0 and value != -1 for value in values[1:-1])):
        raise ValueError("nonfinite or invalid CTXTIME timestamp/sentinel")
    present = [value for value in values if value != -1]
    if present != sorted(present):
        raise ValueError("invalid CTXTIME timestamp ordering")
    _, prefill_start, prefill_end, first_token, _ = values
    if ((prefill_end != -1 and prefill_start == -1)
            or (first_token != -1 and (op != "query" or prefill_end == -1))
            or (op in ("reset", "release") and any(value != -1 for value in values[1:-1]))):
        raise ValueError("inconsistent CTXTIME completed phases")
    return dict(zip(_CTX_TIME_FIELDS, values))


def _parse_retained_kv_transfer(line: str, expected_id: int) -> dict[str, int] | None:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError("malformed CTXKV field count")
    if int(fields[1]) != expected_id:
        return None
    result = {}
    for field, key in zip(fields[2:], ("imported_layers", "capture_us", "snapshot_bytes")):
        name, separator, value = field.partition("=")
        if name != key or not separator or not re.fullmatch(r"[0-9]+", value):
            raise ValueError("malformed CTXKV unsigned field")
        number = int(value)
        if number >= 2**64:
            raise ValueError("CTXKV unsigned field exceeds uint64")
        result[key] = number
    return result


class MNNAdapterWorkerBackend(WorkerBackend):
    """
    Drives ``tabm_decoder_worker`` (native/tabm_decoder_worker.cpp) as a
    ``WorkerBackend``.

    The model is loaded ONCE at construction (blocking until ``READY``).
    Each ``submit()``/``poll()`` pair drives exactly one request; a second
    ``submit()`` before the first resolves raises (single in-flight, mirrors
    ``MNNDemoWorkerBackend``). ``command_prefix`` supports process placement
    such as ``("taskset", "-c", "0-4")`` without changing the worker wire.
    """

    def __init__(
        self,
        worker_bin: str,
        config_path: str,
        *,
        cwd: Optional[str] = None,
        load_timeout: float = 180.0,
        request_timeout: float = 120.0,
        env: Optional[dict] = None,
        health_check: Optional[Callable[[], None]] = None,
        command_prefix: tuple[str, ...] = (),
        retained_context: bool = False,
        retained_diagnostics_dir: Optional[str] = None,
        retained_cpu_fixed_scale: bool = False,
        retained_opencl_quant_gemm: bool = False,
        retained_segmented_replay: bool = False,
        retained_kv_transfer: bool = False,
        retained_qwen_positions: bool = False,
    ) -> None:
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("request_timeout must be positive and finite")
        self.request_timeout = request_timeout
        run_env = dict(os.environ)
        if env:
            run_env.update(env)
        if not isinstance(retained_context, bool):
            raise ValueError("retained_context must be an explicit boolean")
        if not isinstance(retained_cpu_fixed_scale, bool) or (retained_cpu_fixed_scale and not retained_context):
            raise ValueError("retained_cpu_fixed_scale must be an explicit boolean and requires retained_context=True")
        if not isinstance(retained_opencl_quant_gemm, bool) or (retained_opencl_quant_gemm and not retained_context):
            raise ValueError("retained_opencl_quant_gemm must be an explicit boolean and requires retained_context=True")
        if not isinstance(retained_segmented_replay, bool) or (retained_segmented_replay and not retained_context):
            raise ValueError("retained_segmented_replay must be an explicit boolean and requires retained_context=True")
        if not isinstance(retained_kv_transfer, bool) or (retained_kv_transfer and not retained_context):
            raise ValueError("retained_kv_transfer must be an explicit boolean and requires retained_context=True")
        if sum((retained_cpu_fixed_scale, retained_opencl_quant_gemm, retained_segmented_replay)) > 1:
            raise ValueError("retained numerical policies are mutually exclusive")
        self._retained_enabled = retained_context
        self._retained_segmented_replay = retained_segmented_replay
        self._retained_kv_transfer = retained_kv_transfer
        if retained_qwen_positions and not retained_segmented_replay:
            raise ValueError("Qwen positions require retained segmented replay")
        self._retained_qwen_positions = retained_qwen_positions
        self._qwen_position_contract: str | None = None
        self._reference_contract: str | None = None
        self._kv_transfer_contract: str | None = None
        self._visual_prefix_contract: str | None = None
        self._retained_state: Optional[dict] = None
        streams = (env or {}).get("EDGEFLOW_TABM_RETAINED_STREAMS", "1")
        if streams not in ("1", "2") or (streams == "2" and not retained_context):
            raise ValueError("EDGEFLOW_TABM_RETAINED_STREAMS requires 1 or retained_context=True with 2")
        self._retained_streams = int(streams)
        self._shared_weights_contract = None
        self._retained_states: dict[str, dict] = {}
        run_env.pop("EDGEFLOW_TABM_RETAINED_STREAMS", None)
        if streams == "2":
            run_env["EDGEFLOW_TABM_RETAINED_STREAMS"] = streams
        self._retained_requests = 0
        self._retained_diagnostics_dir = None
        # Do not accidentally inherit an opt-in from an unrelated experiment.
        run_env.pop("EDGEFLOW_TABM_RETAINED", None)
        run_env.pop("EDGEFLOW_TABM_RETAINED_DIAGNOSTICS_DIR", None)
        run_env.pop("EDGEFLOW_TABM_RETAINED_CPU_FIXED_SCALE", None)
        run_env.pop("EDGEFLOW_TABM_RETAINED_OPENCL_QUANT_GEMM", None)
        run_env.pop("EDGEFLOW_TABM_RETAINED_SEGMENTED_REPLAY", None)
        run_env.pop("EDGEFLOW_TABM_RETAINED_KV_TRANSFER", None)
        run_env.pop("CROSSINFER_MNN_KV_TRANSFER", None)
        for key, expected in (
            ("MNN_EDGEFLOW_FIXED_QUERY_SCALE", "1" if retained_cpu_fixed_scale else None),
            ("MNN_CROSSINFER_PREFILL_LINEAR_POLICY", "quant_gemm" if retained_opencl_quant_gemm else None),
        ):
            if env is None or key not in env:
                run_env.pop(key, None)
            elif retained_context and env[key] != expected:
                raise ValueError(f"{key} conflicts with retained numerical-policy opt-in")
        if retained_diagnostics_dir is not None:
            directory = os.fspath(retained_diagnostics_dir)
            if not retained_context or not os.path.isabs(directory) or not os.path.isdir(directory):
                raise ValueError("retained diagnostics require an existing absolute experiment directory and opt-in")
            self._retained_diagnostics_dir = os.path.realpath(directory)
            run_env["EDGEFLOW_TABM_RETAINED_DIAGNOSTICS_DIR"] = self._retained_diagnostics_dir
        if retained_context:
            # Native checks actual dump_config().precision BEFORE load, not a
            # host declaration. Legacy low needs its actual native capability;
            # segmented replay instead declares a same-partition reference.
            run_env["EDGEFLOW_TABM_RETAINED"] = "1"
        if retained_cpu_fixed_scale:
            run_env["EDGEFLOW_TABM_RETAINED_CPU_FIXED_SCALE"] = "1"
            run_env["MNN_EDGEFLOW_FIXED_QUERY_SCALE"] = "1"
        if retained_opencl_quant_gemm:
            run_env["EDGEFLOW_TABM_RETAINED_OPENCL_QUANT_GEMM"] = "1"
            run_env["MNN_CROSSINFER_PREFILL_LINEAR_POLICY"] = "quant_gemm"
        if retained_segmented_replay:
            run_env["EDGEFLOW_TABM_RETAINED_SEGMENTED_REPLAY"] = "1"
        if retained_kv_transfer:
            run_env["EDGEFLOW_TABM_RETAINED_KV_TRANSFER"] = "1"
            run_env["CROSSINFER_MNN_KV_TRANSFER"] = "1"
        self._trace_enabled = run_env.get("EDGEFLOW_TABM_TRACE") == "1"
        self._retained_borrow_input = run_env.get("EDGEFLOW_TABM_RETAINED_BORROW_INPUT") == "1"
        self._mnn_gpu_profile_enabled = run_env.get("EDGEFLOW_MNN_GPU_PROFILE") == "1"
        self._mnn_gpu_totals_us: list[int] = []
        # Record the actual backend settings without exporting credentials.
        self._execution_environment = {
            key: value for key, value in run_env.items()
            if key.startswith(("MNN_", "EDGEFLOW_")) or key in (
                "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "CROSSINFER_MNN_KV_TRANSFER",
            )
        }
        run_cwd = cwd if cwd is not None else os.path.dirname(os.path.abspath(worker_bin))
        self._health_check = health_check
        self._proc = subprocess.Popen(
            [*command_prefix, worker_bin, config_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            cwd=run_cwd,
            env=run_env,
        )
        os.set_blocking(self._proc.stdin.fileno(), False)

        self._inflight: Optional[int] = None
        self._deadline: Optional[float] = None
        self._terminated = False
        self._stopping = False
        self.load_s: float = 0.0
        # Separate id space from decode WorkItem.request_id — see
        # tokenize_prefix(); TOK never overlaps a decode request in flight.
        self._next_tok_id: int = 0
        # PFXINFO stash for the ASYNC submit/poll prefix path:
        # _scan_res() parks a matching-id PFXINFO here; _parse_res() attaches
        # it to WorkResult.extra and clears it. Reset on each prefix submit.
        self._pfx_stash = None
        self._vinfo_stash = None
        self._trace_expected = False
        self._trace_stash = None
        self._trace_error: Optional[str] = None
        self._t_start_monotonic = 0.0
        # Public operations still have one scheduler owner. The reader only
        # frames stdout; it never mutates request state or invokes callbacks.
        self._oneshot: Optional[str] = None
        self._output = _AdapterWorkerOutput(self._proc.stdout, profile=self._mnn_gpu_profile_enabled)

        try:
            ready_line = self._wait_for_ready(load_timeout)
        except BaseException:
            self.terminate()
            raise
        if ready_line is None:
            self.terminate()
            raise RuntimeError(
                "tabm_decoder_worker did not print READY within load_timeout "
                f"({load_timeout}s) — check the worker binary / model config"
            )
        if ready_line.startswith(_FATAL_PREFIX):
            self.terminate()
            raise RuntimeError(
                f"tabm_decoder_worker failed to start: {ready_line!r}"
            )
        if self._retained_streams == 2 and self._shared_weights_contract != "shared_weights_2_v1":
            self.terminate()
            raise RuntimeError("two retained streams require CTXCAP shared_weights_2_v1 before READY")
        if retained_segmented_replay and self._reference_contract != "per_frame_v1":
            self.terminate()
            raise RuntimeError("retained segmented replay requires CTXCAP per_frame_v1 before READY")
        if retained_qwen_positions and self._qwen_position_contract != "qwen_image_sequence_v1":
            self.terminate()
            raise RuntimeError("Qwen requires CTXCAP qwen_image_sequence_v1 before READY")
        if retained_kv_transfer and (
            self._reference_contract != "per_frame_v1" or self._kv_transfer_contract != "kv_transfer_v1"
        ):
            self.terminate()
            raise RuntimeError("retained KV transfer requires CTXCAP per_frame_v1 and CTXCAP kv_transfer_v1 before READY")
        # ready_line startswith READY_PREFIX here.
        parts = ready_line.split()
        if len(parts) > 1:
            try:
                self.load_s = float(parts[1])
            except ValueError:
                self.load_s = 0.0

    # ------------------------------------------------------------------
    # low-level IO
    # ------------------------------------------------------------------
    @property
    def execution_environment(self) -> dict[str, str]:
        """Effective child backend/loader/thread settings, not all ambient env."""
        return dict(self._execution_environment)

    @property
    def reference_contract(self) -> str | None:
        """Native handshake declaration, not a numerical/hardware acceptance gate."""
        return self._reference_contract

    @property
    def metadata(self) -> dict:
        """Safe retained-mode receipt; no ambient environment or credentials."""
        return {"retained_context": self._retained_enabled,
                "reference_contract": self.reference_contract,
                "retained_kv_transfer": self._retained_kv_transfer,
                "retained_streams": self._retained_streams,
                "shared_weights_contract": self._shared_weights_contract}

    def _run_health_check(self) -> None:
        if self._health_check is None:
            return
        try:
            self._health_check()
        except BaseException:
            # Synchronous one-shot replies may arrive after the exception. A
            # terminated worker cannot leak that stale reply into a later op.
            self.terminate()
            raise

    def _next_line(self) -> Optional[str]:
        item = self._output.pop()
        if item is None:
            return None
        line, arrived_at = item
        if self._deadline is not None and arrived_at > self._deadline:
            raise TimeoutError("worker response arrived after request timeout")
        return line

    def _write_line(self, line: str, deadline: float, *, reject_expired: bool = False) -> None:
        """Nonblocking fd writes with a total monotonic deadline, including stalls."""
        data = memoryview(line.encode("utf-8"))
        total_bytes = len(data)
        fd = self._proc.stdin.fileno()
        while data:
            self._run_health_check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if reject_expired and len(data) == total_bytes:
                    raise RequestValidationError("request deadline expired before worker I/O")
                raise TimeoutError("worker stdin write exceeded request timeout")
            try:
                written = os.write(fd, data[:16384])
            except BlockingIOError:
                select.select([], [fd], [], min(remaining, 0.05))
                continue
            if written == 0:
                raise BrokenPipeError("worker stdin closed during request")
            data = data[written:]

    def _wait_for_ready(self, timeout: float) -> Optional[str]:
        """
        Block (up to timeout) reading stdout, filtering to the first complete
        line that starts with READY or FATAL — MNN pollutes stdout with other
        logs before READY, so junk lines are discarded (prefix filter).

        Returns that line (with any trailing '\\r' but no '\\n'), or None if
        the timeout elapses, or the process exits, without ever producing one.
        """
        end = time.monotonic() + timeout
        while True:
            self._run_health_check()
            if self._proc.poll() is not None:
                self._output.thread.join(timeout=0.2)
            found = self._scan_ready_fatal()
            if found is not None:
                return found
            remaining = end - time.monotonic()
            if remaining <= 0 or self._output.eof.is_set():
                return self._scan_ready_fatal()
            self._output.wait(min(0.2, remaining))

    def _scan_ready_fatal(self) -> Optional[str]:
        """
        Consume complete lines retained by the stdout reader, never fragments.
        """
        while (line := self._next_line()) is not None:
            if line.startswith("CTXCAP "):
                fields = line.split()
                if self._retained_streams == 2 and fields == ["CTXCAP", "shared_weights_2_v1"]:
                    if self._shared_weights_contract is not None:
                        raise RuntimeError("duplicate CTXCAP shared_weights_2_v1 startup handshake")
                    self._shared_weights_contract = "shared_weights_2_v1"
                    continue
                if self._retained_segmented_replay or self._retained_kv_transfer:
                    fields = line.split()
                    if fields == ["CTXCAP", "per_frame_v1"] and self._reference_contract is None:
                        self._reference_contract = "per_frame_v1"
                    elif (self._retained_qwen_positions and fields == ["CTXCAP", "qwen_image_sequence_v1"]
                          and self._qwen_position_contract is None):
                        self._qwen_position_contract = "qwen_image_sequence_v1"
                    elif (self._retained_kv_transfer and fields == ["CTXCAP", "kv_transfer_v1"]
                          and self._kv_transfer_contract is None):
                        self._kv_transfer_contract = "kv_transfer_v1"
                    elif (self._retained_kv_transfer and fields == ["CTXCAP", "visual_prefix_v1"]
                          and self._visual_prefix_contract is None):
                        self._visual_prefix_contract = "visual_prefix_v1"
                    else:
                        raise RuntimeError("invalid or duplicate CTXCAP startup handshake")
                continue
            if line.startswith(_READY_PREFIX) or line.startswith(_FATAL_PREFIX):
                return line
        return None

    def _scan_res(self) -> Optional[WorkResult]:
        """
        Consume bounded protocol/profile lines framed by the stdout reader.

        A line starting with "RES " is always a protocol line (never further
        discarded as junk) — _parse_res() always returns a concrete
        WorkResult for it, ok=False with a descriptive error if malformed.

        Once _parse_res() completes a request (success,
        failure, OR id-mismatch — it always clears self._inflight), any
        OTHER complete lines already sitting in self._buf are drained: this
        backend serves one request at a time, so a "RES ..." line still
        present at this point cannot belong to a future request (none has
        been submitted yet) — it is a stale leftover that would otherwise be
        mis-read as the response to whatever is submitted next.
        """
        while (line := self._next_line()) is not None:
            info = parse_pfxinfo(line)
            if info is not None:
                # Park the prefix-mode info line for the in-flight
                # request; _parse_res() attaches it to WorkResult.extra.
                if info["id"] == self._inflight:
                    self._pfx_stash = info
                continue
            visual_info = parse_vinfo(line)
            if visual_info is not None:
                if visual_info["id"] == self._inflight:
                    self._vinfo_stash = visual_info
                continue
            if line.startswith("TINFO "):
                trace = parse_tinfo(line)
                if trace is None:
                    self._trace_error = "malformed TINFO"
                elif trace["id"] != self._inflight:
                    self._trace_error = "mismatched TINFO id"
                elif self._trace_stash is not None:
                    self._trace_error = "duplicate TINFO"
                else:
                    self._trace_stash = trace
                continue
            if self._mnn_gpu_profile_enabled and self._inflight is not None:
                profile = _MNN_GPU_TOTAL_RE.match(line)
                if profile is not None:
                    try:
                        self._mnn_gpu_totals_us.append(int(profile.group(1)))
                    except ValueError as exc:
                        raise WorkerOutputError("malformed MNN GPU profile summary") from exc
                    continue
            if not line.startswith(_RES_PREFIX):
                continue  # junk stdout line: ignore (prefix filter)
            result = self._parse_res(line)
            self._drain_stale_res()
            return result
        return None

    def _drain_stale_res(self) -> None:
        """End this operation's stdout ownership and discard stale summaries."""
        self._output.finish()
        self._deadline = None

    def _parse_res(self, line: str) -> WorkResult:
        """
        Parse one complete ``RES ...`` line into a WorkResult.

        Fields: RES <id> <ok:0|1> <n_prompt> <n_decode> <prefill_us>
        <decode_us> <copied_bytes> <ids_csv|-> <err_b64|->
        """
        fields = line.split()
        if len(fields) != _RES_NUM_FIELDS:
            rid = self._inflight
            self._inflight = None
            return WorkResult(
                request_id=rid if rid is not None else -1,
                output_ids=[],
                ok=False,
                error=f"malformed RES line: {line!r}",
                service_time=time.monotonic() - self._t_start_monotonic,
            )
        s_id, s_ok, s_err = fields[1], fields[2], fields[-1]

        try:
            rid = int(s_id)
        except ValueError:
            rid = -1

        if rid != self._inflight:
            # Protocol error: a RES for a request we don't have in flight.
            cur = self._inflight
            self._inflight = None
            return WorkResult(
                request_id=cur if cur is not None else -1,
                output_ids=[],
                ok=False,
                error=f"protocol error: RES id {s_id} does not match inflight {cur}",
                service_time=time.monotonic() - self._t_start_monotonic,
            )

        # A RES with the right FIELD COUNT can still
        # carry a non-numeric value in a numeric slot (or a non-numeric
        # ids_csv entry) — a corrupted/adversarial worker reply must not
        # crash poll(); it must complete the in-flight request as a failure.
        try:
            (n_prompt, n_decode, prefill_us, decode_us,
             copied_bytes, output_ids) = parse_res_numbers(fields)
        except ValueError as exc:
            self._inflight = None
            return WorkResult(
                request_id=rid,
                output_ids=[],
                ok=False,
                error=str(exc),
                service_time=time.monotonic() - self._t_start_monotonic,
            )
        if s_err == "-":
            error = None
        else:
            try:
                error = base64.b64decode(s_err).decode("utf-8", "replace")
            except Exception:
                error = s_err  # defensive: surface the raw token rather than raising

        self._inflight = None
        if self._trace_enabled and self._trace_expected:
            trace_error = self._trace_error
            if (
                trace_error is None
                and s_ok == "1"
                and n_decode > 0
                and self._trace_stash is None
            ):
                trace_error = "missing TINFO for successful VLM request"
            if (
                trace_error is None
                and self._trace_stash is not None
                and sum(self._trace_stash[key] for key in ("n_pre", "n_vis", "n_post"))
                != n_prompt
            ):
                trace_error = "TINFO token counts do not match RES n_prompt"
            if trace_error is not None:
                self._trace_expected = False
                self._trace_stash = None
                self._trace_error = None
                return WorkResult(
                    request_id=rid,
                    output_ids=[],
                    ok=False,
                    error=f"worker trace protocol error: {trace_error}",
                    service_time=time.monotonic() - self._t_start_monotonic,
                )
        extra = {
            "prefill_us": prefill_us,
            "decode_us": decode_us,
            "copied_bytes": copied_bytes,
            "n_prompt": n_prompt,
            "load_s": self.load_s,
        }
        if self._mnn_gpu_profile_enabled and s_ok == "1":
            if not self._mnn_gpu_totals_us:
                return WorkResult(
                    request_id=rid,
                    output_ids=[],
                    ok=False,
                    error="missing MNN GPU profile summaries",
                    service_time=time.monotonic() - self._t_start_monotonic,
                )
            extra["mnn_gpu_kernel_totals_us"] = list(self._mnn_gpu_totals_us)
        self._mnn_gpu_totals_us.clear()
        stash = getattr(self, "_pfx_stash", None)
        if stash is not None and stash["id"] == rid:
            # Prefix-mode requests surface hit/miss + per-call
            # prefill split to the reap/metrics layer via extra.
            extra["pfx_hit"] = stash["hit"]
            extra["pfx_c1_prefill_us"] = stash["c1_prefill_us"]
            extra["pfx_c2_prefill_us"] = stash["c2_prefill_us"]
            extra["pfx_n_prefix"] = stash["n_prefix"]
            self._pfx_stash = None
        visual_stash = getattr(self, "_vinfo_stash", None)
        if visual_stash is not None and visual_stash["id"] == rid:
            extra["visual_nbytes"] = visual_stash["nbytes"]
            extra["visual_crc32"] = visual_stash["crc32"]
            extra["visual_crc_us"] = visual_stash["crc_us"]
        self._vinfo_stash = None
        if self._trace_expected and self._trace_stash is not None:
            trace = dict(self._trace_stash)
            trace.pop("id")
            # Reader arrival decides deadline eligibility, not this public
            # submit-to-poll-observation metric (which includes scheduler lag).
            trace["adapter_observed_wall_us"] = max(
                0, int((time.monotonic() - self._t_start_monotonic) * 1e6)
            )
            extra["worker_trace"] = trace
        self._trace_expected = False
        self._trace_stash = None
        self._trace_error = None
        self._mnn_gpu_totals_us.clear()
        return WorkResult(
            request_id=rid,
            output_ids=output_ids,
            ok=(s_ok == "1" and n_decode > 0),
            error=error,
            service_time=(prefill_us + decode_us) / 1e6,
            # A WELL-FORMED RES (whatever its ok bit) proves the worker is
            # alive and speaking the protocol: any failure it reports is
            # request-level, not a lane fault.  The malformed/mismatch early
            # returns above deliberately keep the conservative default
            # (lane_fault=True) — those are protocol violations.
            lane_fault=False,
            extra=extra,
        )

    # ------------------------------------------------------------------
    # WorkerBackend interface
    # ------------------------------------------------------------------
    def submit(self, work: WorkItem) -> None:
        """
        Build and send exactly one ``REQ`` line for *work* (single in-flight).

        Mode routing: decided from ``work.embeds_desc`` /
        ``work.prompt_text``, in this order:

          1. ``embeds_desc`` present AND ``prompt_text`` non-empty -> ``vlm``
             REQ: the ``embeds`` descriptor fields (shm_name, offset, n_vis,
             hidden) followed by ``base64(prompt_text)`` as the LAST token on
             the line (native/tabm_decoder_worker.cpp requires the prompt to
             contain the literal ``<image>`` placeholder; that is validated
             worker-side, not here — a missing placeholder surfaces as a
             normal ``ok=0`` RES).
          2. ``embeds_desc`` present AND ``prompt_text`` empty -> ``embeds``
             REQ (unchanged from before ``vlm`` existed).
          3. ``embeds_desc`` is ``None`` -> ``text`` REQ; ``prompt_text`` is
             then required (unchanged).
        """
        self._reject_retained_owner()
        if self._inflight is not None:
            raise RuntimeError(
                f"MNNAdapterWorkerBackend already has request {self._inflight} in flight"
            )
        if self._oneshot is not None:
            # Reentrancy/misuse guard: a TOK/prefix
            # one-shot owns the shared stdin/stdout/_buf right now —
            # interleaving a decode REQ would corrupt both streams.
            raise RuntimeError(
                f"MNNAdapterWorkerBackend.submit: a {self._oneshot} one-shot "
                "is in progress -- decode submits must wait"
            )
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"tabm_decoder_worker is not alive (exit {self._proc.returncode})"
            )
        if self._stopping:
            raise WorkerTerminationError("worker shutdown is unconfirmed; new work is refused")
        request_deadline = work.deadline_monotonic
        if request_deadline is not None:
            if not _finite_number(request_deadline):
                raise RequestValidationError("deadline_monotonic must be finite or None")
            if time.monotonic() >= request_deadline:
                raise RequestValidationError("request deadline expired before worker I/O")
        line, is_vlm = build_decode_req(work)
        self._vinfo_stash = None
        self._trace_expected = False
        self._trace_stash = None
        self._trace_error = None
        self._mnn_gpu_totals_us.clear()

        if work.prefix_name is not None:
            self._pfx_stash = None

        self._trace_expected = self._trace_enabled and is_vlm
        self._t_start_monotonic = time.monotonic()
        self._deadline = self._t_start_monotonic + self.request_timeout
        if request_deadline is not None:
            self._deadline = min(self._deadline, request_deadline)
        self._output.begin()
        # Ownership precedes the first byte: a partial write may reach a child
        # still holding shared memory even when submit subsequently raises.
        self._inflight = work.request_id
        try:
            self._write_line(line, self._deadline, reject_expired=request_deadline is not None)
        except RequestValidationError:
            # No byte was sent: keep the healthy worker and release ownership.
            self._inflight = None
            self._drain_stale_res()
            raise
        except BaseException:
            self.terminate()
            self._inflight = None
            raise

    def poll(self) -> Optional[WorkResult]:
        """
        Non-blocking.  Returns None if no result is ready yet.

        service_time on a genuine RES is exact engine time (see module
        docstring); on the synthesized "worker died" failure below it is wall
        time elapsed since submit() — there is no engine measurement for a
        request the worker never finished.
        """
        if self._inflight is None:
            return None
        # A process may write a complete RES and exit before the scheduler next
        # polls it.  Protocol output already delivered through the pipe wins
        # over the later liveness observation; otherwise a valid completion is
        # misreported as "worker died (exit 0)".
        try:
            result = self._scan_res()
            if result is not None:
                return result
            exited = self._proc.poll() is not None
            eof = self._output.eof.is_set()
            expired = self._deadline is not None and time.monotonic() >= self._deadline
            if exited or eof or expired:
                if exited:
                    self._output.thread.join(timeout=0.2)
                # The reader may have queued an on-time reply after the empty
                # scan but before the exit/EOF/deadline observation above.
                result = self._scan_res()
                if result is not None:
                    return result
                if expired:
                    if self._deadline < self._t_start_monotonic + self.request_timeout:
                        raise TimeoutError("request deadline expired")
                    raise TimeoutError(f"request timeout ({self.request_timeout}s)")
                if exited:
                    raise WorkerOutputError(f"tabm_decoder_worker died (exit {self._proc.returncode})")
                raise WorkerOutputError("tabm_decoder_worker stdout EOF without a complete RES")
        except (TimeoutError, WorkerOutputError) as exc:
            rid = self._inflight
            elapsed = time.monotonic() - self._t_start_monotonic
            self.terminate()  # Failure is deliverable only after confirmed exit.
            self._inflight = None
            self._drain_stale_res()
            return WorkResult(rid, [], False, str(exc), elapsed)
        return None

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def terminate(self) -> None:
        if self._terminated:
            return
        # A failed kill must not make this instance reusable, even after a
        # one-shot helper clears its reentrancy guard in finally.
        self._stopping = True
        if self._proc.poll() is None:
            # Never splice EXIT into a partially written request. Even at idle
            # this is a best-effort nonblocking write, not a buffered flush.
            if self._inflight is None and self._oneshot is None:
                try:
                    os.write(self._proc.stdin.fileno(), b"EXIT\n")
                except (OSError, ValueError):
                    pass
            for action, timeout in ((None, 0.2), (self._proc.terminate, 0.3),
                                    (self._proc.kill, 1.0)):
                try:
                    if action is not None and self._proc.poll() is None:
                        action()
                    self._proc.wait(timeout=timeout)
                    break
                except (OSError, subprocess.TimeoutExpired):
                    continue
        if self._proc.poll() is None:
            raise WorkerTerminationError(
                f"worker {self._proc.pid} termination could not be confirmed; "
                "shared-memory ownership must be retained"
            )
        self._output.close()
        for stream in (self._proc.stdin, self._proc.stdout):
            try:
                stream.close()
            except OSError:
                pass
        self._terminated = True

    # ------------------------------------------------------------------
    # TOK: prefix-identity op (see module docstring)
    # ------------------------------------------------------------------
    def tokenize_prefix(self, text: str, *, timeout: float = 30.0) -> tuple[int, str]:
        """
        Send ``TOK <id> <b64(text)>`` and block (up to `timeout` seconds) for
        the matching reply, returning ``(token_count, token_ids_hash)``.

        This is a SEPARATE, synchronous one-shot op — NOT part of the
        submit()/poll() WorkerBackend decode path (see lane.WorkerBackend);
        it exists for whatever populates a PrefixRegistry
        (prefix_registry.py), not the hot decode dispatch loop.

        Raises RuntimeError if:
          - a decode request is currently in flight (this call would
            otherwise race the same stdin/stdout stream against
            submit()/poll()),
          - the worker is not alive (before sending, or dies while waiting),
          - the reply is malformed or carries a mismatched id, or
          - the worker replies with its generic per-request error convention
            for a malformed TOK line (native/tabm_decoder_worker.cpp's TOK
            handler — NOT a process FATAL; the decoded error is included),
          - no reply arrives within `timeout` — the worker is TERMINATED as
            part of raising this error (a wedged worker cannot be trusted
            not to reply late and poison a subsequent call's parsing); the
            backend instance must be recreated before issuing another
            request.
        """
        if self._inflight is not None:
            raise RuntimeError(
                "MNNAdapterWorkerBackend.tokenize_prefix: a decode request is "
                f"already in flight (request_id={self._inflight})"
            )
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"tabm_decoder_worker is not alive (exit {self._proc.returncode})"
            )
        self._begin_oneshot("tokenize_prefix")
        try:
            return self._tokenize_prefix_locked(text, timeout)
        finally:
            self._oneshot = None

    def _tokenize_prefix_locked(self, text: str, timeout: float) -> tuple[int, str]:
        tok_id = self._next_tok_id
        self._next_tok_id += 1
        b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        line = f"TOK {tok_id} {b64}\n"
        return self._exchange(line, timeout, lambda: self._scan_tok_res(tok_id), "TOK")

    def _exchange(self, line: str, timeout: float, scan: Callable, kind: str):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self._deadline = time.monotonic() + timeout
        self._output.begin()
        try:
            self._write_line(line, self._deadline)
            while True:
                self._run_health_check()
                result = scan()
                if result is not None:
                    return result
                remaining = self._deadline - time.monotonic()
                exited = self._proc.poll() is not None
                if remaining <= 0 or exited or self._output.eof.is_set():
                    if exited:
                        self._output.thread.join(timeout=0.2)
                    result = scan()
                    if result is not None:
                        return result
                    if remaining <= 0:
                        raise TimeoutError(f"no {kind} reply within timeout ({timeout}s)")
                    raise WorkerOutputError(
                        f"tabm_decoder_worker died or stdout EOF while waiting for {kind} "
                        f"reply (exit {self._proc.returncode})"
                    )
                self._output.wait(min(0.2, remaining))
        except TimeoutError as exc:
            self.terminate()
            raise RuntimeError(f"{exc}; worker terminated to prevent stale-reply poisoning") from exc
        except (OSError, WorkerOutputError):
            self.terminate()
            raise

    def _scan_tok_res(self, expected_id: int) -> Optional[tuple[int, str]]:
        """
        Consume complete lines from self._buf looking for the reply to TOK
        request `expected_id`. Non-"RES " lines are junk (prefix filter, same
        discipline as _scan_res()). A "RES " line is dispatched on its THIRD
        field: "TOK" -> the TOK-success shape (RES <id> TOK <count> <hash>);
        anything else -> native/tabm_decoder_worker.cpp's GENERIC per-request
        error RES shape, which is how a malformed TOK line is reported (NOT
        a process FATAL). Any remaining complete lines in the buffer are
        drained once this line is resolved (mirrors _scan_res()'s stale-line
        draining) so a later tokenize_prefix() call never inherits a stale
        reply.
        """
        while (line := self._next_line()) is not None:
            if not line.startswith(_RES_PREFIX):
                continue  # junk stdout line: ignore (prefix filter)

            fields = line.split()
            if len(fields) >= 3 and fields[2] == _TOK_TAG:
                if len(fields) != _TOK_RES_NUM_FIELDS:
                    self._drain_stale_res()
                    raise RuntimeError(f"malformed TOK RES line: {line!r}")
                _tag, s_id, _tok_tag, s_count, tok_hash = fields
                try:
                    rid = int(s_id)
                    token_count = int(s_count)
                    if token_count < 0:
                        raise ValueError("negative token count")
                except ValueError:
                    self._drain_stale_res()
                    raise RuntimeError(f"malformed TOK RES line: {line!r}")
                self._drain_stale_res()
                if rid != expected_id:
                    raise RuntimeError(
                        f"protocol error: TOK RES id {rid} does not match "
                        f"requested {expected_id}"
                    )
                return token_count, tok_hash

            # Not TOK-shaped: the worker's generic per-request error convention.
            err = _decode_generic_error(line)
            self._drain_stale_res()
            raise RuntimeError(f"tokenize_prefix failed for id={expected_id}: {err}")
        return None

    # ------------------------------------------------------------------
    # Prefix-cache request (one-shot, like tokenize_prefix)
    # ------------------------------------------------------------------

    def run_prefix_request(
        self,
        name: str,
        prefix_text: str,
        suffix_text: str,
        *,
        max_tokens: int = 16,
        timeout: float = 120.0,
    ) -> dict:
        """
        Send ``REQ <id> prefix ...`` (adapter_protocol.build_prefix_req) and
        block for the PFXINFO + RES pair. Returns::

            {"ok": bool, "output_ids": [int], "n_prompt": int,
             "prefill_us": int, "decode_us": int, "error": Optional[str],
             "pfx": {"hit": bool, "c1_prefill_us": int, "c2_prefill_us": int,
                     "n_prefix": int, "n_suffix": int} | None}

        ``pfx`` is None only if the worker's PFXINFO line was lost/garbled
        (the RES result still stands — callers must treat hit/miss as
        UNKNOWN then, not assume either).

        Same one-shot discipline as tokenize_prefix(): raises RuntimeError
        when a decode request is in flight, when the worker is/goes dead,
        on malformed/mismatched replies, and on timeout — the timeout path
        TERMINATES the worker (a wedged worker's late reply
        must never poison a later call).
        """
        self._reject_retained_owner()
        if self._inflight is not None:
            raise RuntimeError(
                "MNNAdapterWorkerBackend.run_prefix_request: a decode request "
                f"is already in flight (request_id={self._inflight})"
            )
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"tabm_decoder_worker is not alive (exit {self._proc.returncode})"
            )
        self._begin_oneshot("run_prefix_request")
        try:
            return self._run_prefix_request_locked(
                name, prefix_text, suffix_text, max_tokens, timeout
            )
        finally:
            self._oneshot = None

    # ------------------------------------------------------------------
    # Explicit serial retained root, separate from lane/router dispatch.
    # ------------------------------------------------------------------
    def _reject_retained_owner(self) -> None:
        states = self._retained_states.values() if self._retained_streams == 2 else (self._retained_state,)
        if any(state is not None and state["active"] for state in states):
            raise RequestValidationError("retained root owns worker; retained_release is required before ordinary requests")

    def retained_reset(self, stream_id: str, version: int, prefix_text: str, *, timeout=None) -> dict:
        """Bind/relabel the sole root. Raw prefix is ingested with first frame.

        Requires retained_context=True and ACTUAL native config precision=high,
        or an explicit CPU fixed-scale / OpenCL quant-GEMM low policy and its
        preloaded native v1 capability. Alternatively, retained_segmented_replay
        admits CPU/OpenCL low + greedy, chunk=0 and empty chunk_limits with a
        per_frame_v1 handshake. That reference uses the SAME backend and
        per-frame partition plus private query, not monolithic input equality.
        Full-logit error <=0.005 and token parity remain the hardware gate;
        a handshake/capability is not scientific acceptance.
        """
        return self._retained_request("reset", stream_id, version, timeout=timeout, prefix_text=prefix_text)

    def retained_append(self, stream_id: str, version: int, embeds_desc: object, *, timeout=None) -> dict:
        """Consume one projected fp32 TABM SHM frame; publish version+1.

        Caller owns the descriptor's producer lease until this synchronous
        call resolves (or worker termination is confirmed on transport failure).
        """
        return self._retained_request("append", stream_id, version, timeout=timeout, embeds_desc=embeds_desc)

    def retained_query(
        self, stream_id: str, version: int, suffix_text: str, *,
        max_tokens: int = 8, logits_name: Optional[str] = None, timeout=None,
    ) -> dict:
        """Raw private suffix + bounded decode, then verified root rollback.

        ttft_us is request-to-completed-first-token-step, not text emission.
        logits_name is a safe stem within retained_diagnostics_dir, never a
        path. Returned first_logits_file is an absolute fp32 file path or None.
        Physical removal is deferred by MNN; pending_remove exposes that fact.
        """
        return self._retained_request("query", stream_id, version, timeout=timeout,
                                      suffix_text=suffix_text, max_tokens=max_tokens, logits_name=logits_name)

    def retained_release(self, stream_id: str, version: int, *, timeout=None) -> dict:
        """Explicitly clear the owned root before returning to ordinary REQ."""
        return self._retained_request("release", stream_id, version, timeout=timeout)

    def retained_export(self, stream_id: str, version: int, snapshot_path, *, content_digest: str,
                        export_prefix=None, timeout=None) -> dict:
        """Export all or a completed prefix of the current OpenCL root."""
        return self._retained_request("export", stream_id, version, timeout=timeout,
                                      snapshot_path=snapshot_path, content_digest=content_digest,
                                      export_prefix=export_prefix)

    def retained_import(self, stream_id: str, version: int, snapshot_path, *, content_digest: str, timeout=None) -> dict:
        """Replace CPU root metadata; native layer packing is checked at query prefill."""
        return self._retained_request("import", stream_id, version, timeout=timeout,
                                      snapshot_path=snapshot_path, content_digest=content_digest)

    def retained_cache_import(self, stream_id, version, snapshot_path, *, content_digest, cached_source, timeout=None):
        return self._retained_request("cache_import", stream_id, version, timeout=timeout,
            snapshot_path=snapshot_path, content_digest=content_digest, cached_source=cached_source)

    def _retained_request(self, op, stream_id, version, *, timeout, **kwargs) -> dict:
        if not self._retained_enabled:
            raise RequestValidationError("retained_context=True is required")
        if op in ("export", "import", "cache_import") and not self._retained_kv_transfer:
            raise RequestValidationError("retained_kv_transfer=True is required")
        if op == "cache_import" and self._visual_prefix_contract != "visual_prefix_v1":
            raise RequestValidationError("cache_import requires CTXCAP visual_prefix_v1")
        if self._inflight is not None:
            raise RuntimeError("retained operation cannot overlap an in-flight decode request")
        if self._proc.poll() is not None:
            raise RuntimeError("tabm_decoder_worker is not alive")
        if self._retained_requests >= RETAINED_MAX_REQUESTS:
            raise RequestValidationError("retained request bound exhausted; recreate worker")
        line = build_retained_req(self._next_tok_id, op, stream_id, version, **kwargs)
        state = self._retained_states.get(stream_id) if self._retained_streams == 2 else self._retained_state
        if op not in ("reset", "import", "cache_import"):
            if (state is None or not state["active"] or stream_id != state["stream_id"]
                    or version != state["version"]):
                raise RequestValidationError("wrong retained stream/version (only current root exists)")
            if op != "release" and not state["valid"]:
                raise RequestValidationError("retained root is invalid; reset required")
        if op == "append":
            n_vis = _embeds_field(kwargs["embeds_desc"], "shape")[0]
            n_pre = state["prefix_tokens"] if state["root_end"] == 0 else 0
            if state["root_end"] + n_pre + n_vis > RETAINED_MAX_POSITIONS:
                raise RequestValidationError("retained append exceeds 8192 positions")
        if op == "query":
            if kwargs.get("logits_name") is not None and self._retained_diagnostics_dir is None:
                raise RequestValidationError("retained diagnostics directory is not configured")
            if state["root_end"] == 0 or state["root_end"] + kwargs["max_tokens"] + 1 > RETAINED_MAX_POSITIONS:
                raise RequestValidationError("no ingested frame or retained query exceeds 8192 positions")
        timeout = self.request_timeout if timeout is None else timeout
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        timeout = min(timeout, self.request_timeout)
        self._begin_oneshot("retained_" + op)
        req_id = self._next_tok_id
        self._next_tok_id += 1
        self._retained_requests += 1
        self._retained_visual_info = None
        self._retained_native_timing = None
        self._retained_kv_info = None
        try:
            result = self._exchange(line, timeout, lambda: self._scan_retained_res(
                req_id, op, stream_id, version, kwargs
            ), "CTX")
            self._retained_state = dict(result)
            if self._retained_streams == 2:
                self._retained_states[stream_id] = dict(result)
            return result
        except BaseException:
            # A post-write cancellation cannot return ownership of live SHM.
            self.terminate()
            raise
        finally:
            self._oneshot = None

    def _scan_retained_res(self, expected_id, op, stream_id, version, kwargs) -> Optional[dict]:
        while (line := self._next_line()) is not None:
            if line.startswith("CTXTIME "):
                try:
                    if self._retained_native_timing is not None:
                        raise ValueError("duplicate CTXTIME")
                    self._retained_native_timing = _parse_retained_timing(line, expected_id, op)
                except ValueError as exc:
                    self.terminate()
                    raise RuntimeError(f"retained worker protocol error: {exc}") from exc
                continue
            if line.startswith("CTXKV "):
                try:
                    evidence = _parse_retained_kv_transfer(line, expected_id)
                    if evidence is not None:
                        if self._retained_kv_info is not None:
                            raise ValueError("duplicate CTXKV")
                        self._retained_kv_info = evidence
                except ValueError as exc:
                    self.terminate()
                    raise RuntimeError(f"retained worker protocol error: {exc}") from exc
                continue
            visual = parse_vinfo(line)
            if visual is not None:
                if visual["id"] == expected_id:
                    self._retained_visual_info = visual
                continue
            if not line.startswith(_RES_PREFIX):
                continue
            try:
                result = parse_retained_res(line)
                if result["request_id"] != expected_id or result["op"] != op:
                    raise ValueError("mismatched CTX request id/operation")
                timing = self._retained_native_timing
                if timing is None and self._retained_segmented_replay and self._trace_enabled:
                    raise ValueError("missing CTXTIME for traced segmented replay")
                if timing is not None:
                    if result["ok"] and op in ("append", "query"):
                        required = ("prefill_start_s", "prefill_end_s")
                        if op == "query":
                            required += ("first_token_ready_s",)
                        if any(timing[key] == -1 for key in required):
                            raise ValueError("missing CTXTIME completed phase for successful operation")
                    result["native_timing"] = dict(timing)
                state = self._retained_states.get(stream_id) if self._retained_streams == 2 else self._retained_state
                if result["root_before"] != (state["root_end"] if state is not None else 0):
                    raise ValueError("mismatched CTX root_before")
                if not result["ok"]:
                    if state is None:
                        if result["active"] or result["stream_id"] is not None or result["root_end"]:
                            raise ValueError("failed CTX created an unverified root")
                    elif any(result[key] != state[key] for key in
                             ("active", "stream_id", "version", "root_end", "prefix_tokens")):
                        raise ValueError("failed CTX changed root ownership")
                if result["ok"]:
                    if op == "release":
                        if result["active"] or result["root_end"] or result["stream_id"] is not None:
                            raise ValueError("CTX release did not clear root")
                    elif (not result["active"] or result["stream_id"] != stream_id
                          or result["version"] != version + (op == "append")):
                        raise ValueError("mismatched CTX root ownership")
                    if op == "reset" and (result["root_end"] or not result["prefix_tokens"]):
                        raise ValueError("CTX reset did not prepare a fresh root")
                    if op == "export" and (
                        result["root_end"] != state["root_end"]
                        or result["prefix_tokens"] != state["prefix_tokens"]
                    ):
                        raise ValueError("CTX export changed root metadata")
                    if op == "append":
                        desc = kwargs["embeds_desc"]
                        n_vis, hidden = _embeds_field(desc, "shape")
                        n_pre = state["prefix_tokens"] if state["root_end"] == 0 else 0
                        n_input = n_pre + n_vis
                        copied_bytes = 0 if self._retained_borrow_input and not n_pre else n_input * hidden * 4
                        if (result["n_input"] != n_input
                                or result["root_end"] != state["root_end"] + n_input
                                or result["prefix_tokens"] != state["prefix_tokens"]
                                or result["copied_bytes"] != copied_bytes
                                or not result["first_logits_count"]):
                            raise ValueError("CTX append receipt differs from requested frame/prefix")
                        if (self._retained_visual_info is None
                                or self._retained_visual_info["nbytes"] != _embeds_field(desc, "nbytes")):
                            raise ValueError("CTX append lacks matching visual byte receipt")
                    if op == "query":
                        name = kwargs.get("logits_name")
                        if (result["root_end"] != result["root_before"]
                                or len(result["output_ids"]) > kwargs["max_tokens"]
                                or result["first_logits_file"] != (name + ".f32" if name else None)):
                            raise ValueError("CTX query changed root or diagnostics/output contract")
                if self._retained_visual_info is not None:
                    for key in ("nbytes", "crc32", "crc_us"):
                        result["visual_" + key] = self._retained_visual_info[key]
                if self._retained_kv_info is not None:
                    result["kv_transfer"] = dict(self._retained_kv_info)
                if result["first_logits_file"] is not None:
                    if self._retained_diagnostics_dir is None:
                        raise ValueError("unconfigured CTX logits output")
                    result["first_logits_file"] = os.path.join(
                        self._retained_diagnostics_dir, result["first_logits_file"]
                    )
            except ValueError as exc:
                # A malformed state receipt cannot make this root reusable.
                self.terminate()
                raise RuntimeError(f"retained worker protocol error: {exc}") from exc
            self._drain_stale_res()
            return result
        return None

    def _begin_oneshot(self, kind: str) -> None:
        """
        Single-scheduler REENTRANCY/MISUSE guard: a one-shot op owns the worker's
        stdin/stdout + _buf for its whole duration; this flag makes an
        interleaved call FAIL LOUD instead of corrupting both streams.
        It is NOT a thread-safe mutex (no lock; nothing else in this class
        is synchronized) — the backend follows TABM's single-scheduler-
        thread contract and must not be shared across threads.
        """
        if self._stopping:
            raise WorkerTerminationError("worker shutdown is unconfirmed; new work is refused")
        if self._oneshot is not None:
            raise RuntimeError(
                f"MNNAdapterWorkerBackend: a {self._oneshot} one-shot is "
                f"already in progress -- cannot start {kind}"
            )
        self._oneshot = kind

    def _run_prefix_request_locked(
        self, name: str, prefix_text: str, suffix_text: str,
        max_tokens: int, timeout: float,
    ) -> dict:
        req_id = self._next_tok_id
        self._next_tok_id += 1
        line = build_prefix_req(req_id, max_tokens, name, prefix_text, suffix_text)
        self._pfx_info = None
        return self._exchange(line, timeout, lambda: self._scan_pfx_res(req_id), "prefix")

    def _scan_pfx_res(self, expected_id: int) -> Optional[dict]:
        """
        Consume complete lines looking for request `expected_id`'s PFXINFO
        (stashed into self._pfx_info) and its standard 10-field RES (which
        completes the call). Junk lines are prefix-filtered as everywhere
        else; a PFXINFO for a DIFFERENT id is stale garbage and dropped
        (single-in-flight discipline makes that unreachable in practice).
        """
        while (line := self._next_line()) is not None:
            info = parse_pfxinfo(line)
            if info is not None:
                if info["id"] == expected_id:
                    self._pfx_info = info
                continue
            if not line.startswith(_RES_PREFIX):
                continue  # junk stdout line: ignore (prefix filter)
            fields = line.split()
            if len(fields) != _RES_NUM_FIELDS:
                self._drain_stale_res()
                raise RuntimeError(f"malformed RES line in prefix reply: {line!r}")
            s_id, s_ok, s_err = fields[1], fields[2], fields[-1]
            try:
                rid = int(s_id)
            except ValueError:
                self._drain_stale_res()
                raise RuntimeError(f"malformed RES id in prefix reply: {line!r}")
            self._drain_stale_res()
            if rid != expected_id:
                raise RuntimeError(
                    f"protocol error: prefix RES id {rid} does not match "
                    f"requested {expected_id}"
                )
            try:
                (n_prompt, _n_decode, prefill_us, decode_us,
                 _copied_bytes, output_ids) = parse_res_numbers(fields)
            except ValueError as exc:
                raise RuntimeError(f"{exc} in prefix reply: {line!r}") from exc
            if s_err == "-":
                error = None
            else:
                try:
                    error = base64.b64decode(s_err).decode("utf-8", "replace")
                except Exception:
                    error = s_err
            return {
                "ok": s_ok == "1" and bool(output_ids),
                "output_ids": output_ids,
                "n_prompt": n_prompt,
                "prefill_us": prefill_us,
                "decode_us": decode_us,
                "error": error,
                "pfx": self._pfx_info,
            }
        return None

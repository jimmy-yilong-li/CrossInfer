"""Bounded, synchronous Qwen image appends and latest-head responses.

One owner thread time-slices one native session; there is no pending ingestion
queue, response commit, eviction, continuous batching, or cross-backend KV.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from time import perf_counter
from typing import Self

import numpy as np

from edgeflow_split.contracts import ImageEmbedsBatch
from edgeflow_split.crossinfer_context import (
    LocalContextCoordinator,
    StreamVersionRegistry,
)
from edgeflow_split.crossinfer_session import (
    DecodeStep,
    ModelFamily,
    PositionContract,
    SessionConfig,
    StatefulLLMSession,
    TokenPieceStreamDecoder,
)
from edgeflow_split.ggml_transport_contracts import VisualPayloadView


@dataclass(frozen=True)
class VisualQueryResult:
    text: str
    token_ids: tuple[int, ...]
    query_token_count: int
    # Starts at response entry, excludes the initial image encode/prefix prefill.
    # Ends when the first non-EOG decode_step returns, not at a device timestamp.
    ttft_ms: float | None
    # Includes branch release and, for stepped responses, time between steps.
    total_ms: float


def _validate_event_time(event_time_ns: int, name: str = "event_time_ns") -> None:
    if type(event_time_ns) is not int or event_time_ns < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _visual_embeddings(batch: ImageEmbedsBatch) -> np.ndarray:
    try:
        grid = tuple(batch.image_grid_thw)
    except TypeError as exc:
        raise ValueError("image_grid_thw must contain exactly three integers") from exc
    if len(grid) != 3:
        raise ValueError("image_grid_thw must contain exactly three integers")
    for name, value in (
        ("num_visual_tokens", batch.num_visual_tokens),
        ("hidden_size", batch.hidden_size),
        *(("image_grid_thw", value) for value in grid),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not 0 < value <= 2**31 - 1
        ):
            raise ValueError(f"{name} must contain positive 32-bit integers")
    t, h, w = map(int, grid)
    if t != 1 or h % 2 or w % 2:
        raise ValueError(
            "image_grid_thw requires one image and even spatial dimensions"
        )
    if (h // 2) * (w // 2) != batch.num_visual_tokens:
        raise ValueError("num_visual_tokens does not match image_grid_thw")
    visual = VisualPayloadView.from_batch(batch).payload_array(dtype="float32")
    if visual.shape != (batch.num_visual_tokens, batch.hidden_size):
        raise ValueError("visual embedding shape does not match batch metadata")
    if not np.isfinite(visual).all():
        raise ValueError("visual embeddings must be finite")
    return visual


class QwenVisualContext:
    """Own a bounded image history with one private response at a time.

    Use as a context manager (or call close()) on the creating thread. Each
    response forks the latest published head from sequence 0 into sequence 1. Root
    appends require the previous response to be closed or its context exited;
    live-branch append is not admitted. The initial image counts toward max_frames.
    After append, queries selecting older boundaries are rejected, not redirected.
    """

    @staticmethod
    def validate_config(config: SessionConfig) -> None:
        """Check visual-position requirements before encoding or allocating KV."""
        if config.model_family != ModelFamily.QWEN25VL:
            raise ValueError("visual context requires the Qwen2.5-VL model family")
        if config.position_contract != PositionContract.QWEN25VL_MROPE:
            raise ValueError("visual context requires Qwen M-RoPE positions")
        if config.n_seq_max < 2:
            raise ValueError("visual context requires at least two sequences")

    def __init__(
        self,
        batch: ImageEmbedsBatch,
        config: SessionConfig,
        *,
        max_frames: int = 1,
        event_time_ns: int = 0,
    ) -> None:
        self.validate_config(config)
        if type(max_frames) is not int or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")
        _validate_event_time(event_time_ns)
        visual = _visual_embeddings(batch)
        self._owner_thread = threading.get_ident()
        self._session = StatefulLLMSession(config)
        self._closed = False
        self._active_response: VisualResponse | None = None
        self._max_frames = max_frames
        self._frame_count = 1
        self._initial_event_time_ns = event_time_ns
        self._event_time_ns = event_time_ns
        self._hidden_size = batch.hidden_size
        try:
            self._session.ingest_qwen25vl_visual_prefix(
                0,
                visual_embeddings=visual,
                image_grid_thw=batch.image_grid_thw,
            )
            self._stream_id = self._session.session_incarnation_id
            self._coordinator = LocalContextCoordinator(
                StreamVersionRegistry(),
                lane_id="local",
                session=self._session,
            )
            self._coordinator.publish_from_sequence(
                self._stream_id,
                source_sequence_id=0,
                event_time_ns=event_time_ns,
            )
        except BaseException as exc:
            try:
                self.close()
            except BaseException as cleanup_error:
                exc.add_note(
                    f"visual context initialization cleanup failed: {cleanup_error}"
                )
                raise exc from cleanup_error
            raise

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("visual context must be used by its creating thread")

    def _require_open(self) -> None:
        self._require_owner()
        if self._closed:
            raise RuntimeError("visual context is closed")

    def _require_response_slot(self) -> None:
        self._require_open()
        if self._active_response is not None:
            raise RuntimeError("visual context already has an active response")

    def append(self, batch: ImageEmbedsBatch, *, event_time_ns: int) -> None:
        """Append/publish synchronously with no active response; never queue it."""
        self._require_response_slot()
        _validate_event_time(event_time_ns)
        if event_time_ns <= self._event_time_ns:
            raise ValueError("event_time_ns must be strictly increasing")
        if self._frame_count >= self._max_frames:
            raise ValueError("max_frames bound exceeded")
        visual = _visual_embeddings(batch)
        if batch.hidden_size != self._hidden_size:
            raise ValueError(
                "visual embedding hidden_size must match the initial image"
            )
        self._session.append_qwen25vl_visual(
            0,
            visual_embeddings=visual,
            image_grid_thw=batch.image_grid_thw,
        )
        try:
            self._coordinator.publish_from_sequence(
                self._stream_id,
                source_sequence_id=0,
                event_time_ns=event_time_ns,
            )
        except BaseException as exc:
            # A changed root without a published version cannot admit more work.
            try:
                self.close()
            except BaseException as cleanup_error:
                exc.add_note(
                    f"visual context publication cleanup failed: {cleanup_error}"
                )
                raise exc from cleanup_error
            raise
        self._frame_count += 1
        self._event_time_ns = event_time_ns

    def response(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        query_event_time_ns: int | None = None,
    ) -> VisualResponse:
        """Create a one-use response context; fork/query occurs on entry.

        Entry rejects pre-initial query times with KeyError and historical query
        times after append with RuntimeError. None selects the latest head.
        """
        self._require_response_slot()
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        if "\0" in prompt:
            raise ValueError("prompt must not contain NUL characters")
        prompt.encode("utf-8")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if query_event_time_ns is not None:
            _validate_event_time(query_event_time_ns, "query_event_time_ns")
        return VisualResponse(self, prompt, max_new_tokens, query_event_time_ns)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        query_event_time_ns: int | None = None,
    ) -> VisualQueryResult:
        with self.response(
            prompt,
            max_new_tokens=max_new_tokens,
            query_event_time_ns=query_event_time_ns,
        ) as response:
            while not response.done:
                response.step()
            return response.result

    def close(self) -> None:
        self._require_owner()
        if not self._closed:
            try:
                if self._active_response is not None:
                    self._active_response.close()
            finally:
                self._session.close()
                self._closed = True

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._require_owner()
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"visual context cleanup failed: {cleanup_error}")


class VisualResponse:
    """Step a latest-head response inside ``with context.response(...)``.

    step() returns the native DecodeStep, including EOG. done becomes true at
    EOG or the token limit, when the branch is released and result is available.
    close() or an early context exit cancels an unfinished response. No step,
    done, or result access is valid before entry or after close/context exit.
    """

    def __init__(
        self,
        context: QwenVisualContext,
        prompt: str,
        max_new_tokens: int,
        query_event_time_ns: int | None,
    ) -> None:
        self._context = context
        self._prompt = prompt
        self._max_new_tokens = max_new_tokens
        self._query_event_time_ns = query_event_time_ns
        self._entered = False
        self._closed = False
        self._branch_context = None
        self._result: VisualQueryResult | None = None
        self._tokens: list[int] = []
        self._pieces: list[str] = []
        self._decoder = TokenPieceStreamDecoder()
        self._ttft_ms: float | None = None

    def __enter__(self) -> Self:
        if self._closed or self._entered:
            raise RuntimeError("visual response is closed or already entered")
        context = self._context
        context._require_response_slot()
        # Recheck at entry: a prepared response may predate a serialized append.
        if self._query_event_time_ns is not None:
            if self._query_event_time_ns < context._initial_event_time_ns:
                raise KeyError(
                    f"stream {context._stream_id} has no version at event time "
                    f"{self._query_event_time_ns}"
                )
            if self._query_event_time_ns < context._event_time_ns:
                raise RuntimeError(
                    "historical visual queries are not supported after append; "
                    "only the latest head is available"
                )
        context._active_response = self
        self._entered = True
        self._start = perf_counter()
        try:
            branch_context = context._coordinator.response_branch(
                context._stream_id,
                destination_sequence_id=1,
                query_event_time_ns=self._query_event_time_ns,
            )
            branch = branch_context.__enter__()
            self._branch_context = branch_context
            self._sequence_id = branch.sequence_id
            self._query_token_count = context._session.ingest_qwen25vl_query(
                branch.sequence_id, self._prompt
            )
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return self

    def _require_active(self) -> None:
        if self._closed or not self._entered:
            raise RuntimeError("visual response is closed or not entered")
        self._context._require_open()

    @property
    def done(self) -> bool:
        self._require_active()
        return self._result is not None

    @property
    def result(self) -> VisualQueryResult:
        self._require_active()
        if self._result is None:
            raise RuntimeError("visual response is not finished")
        return self._result

    def step(self, *, capture_logits: bool = False) -> DecodeStep:
        """Decode one token, optionally capturing diagnostic logits."""
        self._require_active()
        if self._result is not None:
            raise RuntimeError("visual response is already finished")
        if type(capture_logits) is not bool:
            raise ValueError("capture_logits must be a bool")
        try:
            session = self._context._session
            if capture_logits:
                step = session.decode_step(self._sequence_id, capture_logits=True)
            else:
                step = session.decode_step(self._sequence_id)
            if not step.is_eog:
                if self._ttft_ms is None:
                    self._ttft_ms = (perf_counter() - self._start) * 1000
                self._tokens.append(step.token_id)
                self._pieces.append(
                    self._decoder.feed(session.token_piece_bytes(step.token_id))
                )
            if step.is_eog or len(self._tokens) == self._max_new_tokens:
                self._pieces.append(self._decoder.finish())
                self._release_branch()
                self._result = VisualQueryResult(
                    text="".join(self._pieces),
                    token_ids=tuple(self._tokens),
                    query_token_count=self._query_token_count,
                    ttft_ms=self._ttft_ms,
                    total_ms=(perf_counter() - self._start) * 1000,
                )
            return step
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def _release_branch(self, exc_type=None, exc=None, traceback=None) -> None:
        branch_context = self._branch_context
        if branch_context is not None:
            self._branch_context = None
            branch_context.__exit__(exc_type, exc, traceback)

    def close(self) -> None:
        self.__exit__(None, None, None)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._context._require_owner()
        if self._closed:
            return
        try:
            self._release_branch(exc_type, exc, traceback)
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"visual response cleanup failed: {cleanup_error}")
        finally:
            self._closed = True
            if self._context._active_response is self:
                self._context._active_response = None

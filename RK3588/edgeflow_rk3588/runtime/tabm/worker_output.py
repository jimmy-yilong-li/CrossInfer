"""Continuously drain worker stdout, retaining only bounded protocol output."""
from __future__ import annotations

import os
import select
import threading
import time
from collections import deque


class WorkerOutputError(RuntimeError):
    """The stdout transport can no longer deliver a trustworthy response."""


class WorkerOutput:
    MAX_LINE_BYTES = 1024 * 1024
    MAX_PENDING_BYTES = 1024 * 1024
    MAX_LINES = 4096
    _PROTOCOL = (b"READY ", b"FATAL ", b"RES ", b"PFXINFO ", b"VINFO ", b"TINFO ")

    def __init__(self, stream, *, profile: bool = False) -> None:
        self._fd = stream.fileno()
        os.set_blocking(self._fd, False)
        self._prefixes = self._PROTOCOL + ((b"total kernel time =",) if profile else ())
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self.eof = threading.Event()
        self._lines: deque[tuple[str, float]] = deque()
        self._pending_bytes = 0
        self._line_count = 0
        self._partial = bytearray()
        self._discard = False
        self._accepting = True
        self._error: str | None = None
        self.thread = threading.Thread(target=self._read, name="tabm-stdout", daemon=True)
        self.thread.start()

    def begin(self) -> None:
        with self._condition:
            self._lines.clear()
            self._pending_bytes = self._line_count = 0
            # Never attach a fragment from an earlier operation to this one.
            self._discard = self._discard or bool(self._partial)
            self._partial.clear()
            self._accepting = True

    def finish(self) -> None:
        with self._condition:
            self._accepting = False
            self._lines.clear()
            self._pending_bytes = 0

    def pop(self) -> tuple[str, float] | None:
        with self._condition:
            if self._error is not None:
                raise WorkerOutputError(self._error)
            if not self._lines:
                return None
            item = self._lines.popleft()
            self._pending_bytes -= len(item[0].encode("utf-8"))
            return item

    def wait(self, timeout: float) -> None:
        with self._condition:
            if not self._lines and self._error is None and not self.eof.is_set():
                self._condition.wait(timeout)

    def close(self) -> None:
        self._stop.set()
        self.thread.join(timeout=0.5)
        if self.thread.is_alive():
            raise WorkerOutputError("stdout reader did not stop within 0.5s")

    def _fail(self, error: str) -> None:
        self._error = error
        self._accepting = False
        self._condition.notify_all()

    def _feed(self, chunk: bytes) -> None:
        # Only this reader frames bytes. The coordinator only consumes complete
        # lines, so split UTF-8, partial writes and EOF cannot race its parser.
        with self._condition:
            pieces = chunk.split(b"\n")
            for index, piece in enumerate(pieces):
                complete = index < len(pieces) - 1
                if not piece and not complete:
                    continue
                if not self._accepting:
                    self._discard = True
                if not self._discard:
                    self._partial.extend(piece)
                    useful = any(
                        self._partial.startswith(prefix) or prefix.startswith(self._partial)
                        for prefix in self._prefixes
                    )
                    if not useful:
                        self._discard = True
                    elif len(self._partial) > self.MAX_LINE_BYTES:
                        self._fail("worker protocol line exceeds bounded stdout limit")
                        self._discard = True
                    if self._discard:
                        self._partial.clear()
                if not complete:
                    continue
                if not self._discard and self._partial:
                    line = self._partial.decode("utf-8", "replace")
                    size = len(line.encode("utf-8"))
                    if (self._line_count >= self.MAX_LINES
                            or self._pending_bytes + size > self.MAX_PENDING_BYTES):
                        self._fail("worker protocol/profile summaries exceed bounded stdout limit")
                    else:
                        self._lines.append((line, time.monotonic()))
                        self._line_count += 1
                        self._pending_bytes += size
                        # Single request ownership: the first terminal reply
                        # wins; later logs/replies cannot poison a future op.
                        if self._partial.startswith((b"RES ", b"READY ", b"FATAL ")):
                            self._accepting = False
                        self._condition.notify_all()
                self._partial.clear()
                self._discard = False

    def _read(self) -> None:
        try:
            while not self._stop.is_set():
                ready, _, _ = select.select([self._fd], [], [], 0.05)
                if not ready:
                    continue
                try:
                    chunk = os.read(self._fd, 16384)
                except BlockingIOError:
                    continue
                if not chunk:
                    break
                self._feed(chunk)
        except Exception as exc:
            with self._condition:
                self._fail(f"worker stdout read failed: {exc}")
        finally:
            # An unterminated final line is not a valid protocol response.
            self.eof.set()
            with self._condition:
                self._condition.notify_all()

"""Qwen3.5 surface ingestion through the existing versioned native session.

Only the current completed root is forked. The recurrent and attention state
travel together; a response is discarded by dropping its entire sequence, never
by applying attention-only suffix truncation to a recurrent model.
"""

from __future__ import annotations

import time

from edgeflow_split.crossinfer_context import LocalContextCoordinator, StreamVersionRegistry


class Qwen35SurfaceContext:
    def __init__(self, session, *, stream_id: str):
        self.session = session
        self.stream_id = stream_id
        self.registry = StreamVersionRegistry()
        self.coordinator = LocalContextCoordinator(
            self.registry, lane_id="metal", session=session,
        )

    def answer(self, descriptor, prompt: str, *, budget: int, arrival_s: float):
        if budget < 1:
            raise ValueError("output budget must be positive")
        started = time.monotonic()
        frame_id = descriptor.frame_id
        producer = dict(descriptor.handoff_receipt)
        receipt = self.session.append_qwen35_visual(0, descriptor=descriptor)
        appended = time.monotonic()
        if producer["producer_pointer"] != receipt.surface_pointer:
            raise RuntimeError("producer and session mapped different visual output buffers")
        event_time = int(descriptor.timestamp_ms * 1e6) if descriptor.timestamp_ms else frame_id + 1
        version = self.coordinator.publish_from_sequence(
            self.stream_id, source_sequence_id=0, event_time_ns=event_time,
        )
        output, first, last = [], None, None
        with self.coordinator.response_branch(
            self.stream_id, destination_sequence_id=1, query_event_time_ns=event_time,
        ) as branch:
            query_started = time.monotonic()
            query_tokens = self.session.ingest_qwen35_query(branch.sequence_id, prompt)
            for index in range(budget):
                token, _is_eog = self.session.peek_token(branch.sequence_id)
                stamp = time.monotonic()
                if first is None:
                    first = stamp
                last = stamp
                output.append(token)
                # Fixed work, including EOS; do not compute a discarded final step.
                if index + 1 < budget:
                    self.session.ingest_tokens(branch.sequence_id, [token])
        text = b"".join(self.session.token_piece_bytes(token) for token in output).decode(
            "utf-8", errors="replace",
        )
        finished = time.monotonic()
        return {
            "index": frame_id, "arrival_s": arrival_s, "started_s": started,
            "context_prefill_finished_s": appended, "query_started_s": query_started,
            "first_token_s": first, "decode_finished_s": last, "finished_s": finished,
            "output_ids": output, "text": text, "query_tokens": query_tokens,
            "context_version": version.version_id,
            "context_event_time_ns": event_time,
            "context_end_position": version.end_position,
            "context_prefill_ms": (appended - started) * 1000,
            "query_prefill_ms": (first - query_started) * 1000,
            "tpot_ms": (last - first) * 1000 / (budget - 1) if budget > 1 else None,
            "surface_handoff": {
                **producer, "consumer_pointer": receipt.decoder_pointer,
                "pointer_equal": receipt.surface_pointer == receipt.decoder_pointer,
                "payload_bytes": receipt.payload_bytes,
            },
        }

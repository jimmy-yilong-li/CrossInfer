"""Bounded TABM orchestration for leased visual producers and native sessions.

The caller owns the language session and consumes on its owner thread. Only
visual production moves to a background thread. Payloads never materialize in
this layer; the original descriptor owns its lease until consumption completes.
"""

from __future__ import annotations

import threading

from edgeflow_split.shared_tensor import HandleRingBuffer


class SurfacePipeline:
    def __init__(self, capacity: int = 2):
        self._ring = HandleRingBuffer(capacity)

    @property
    def metrics(self) -> dict[str, int | str]:
        return {
            "implementation": "HandleRingBuffer",
            "capacity": self._ring.capacity(),
            "occupancy": self._ring.occupancy(),
            "max_occupancy": self._ring.max_occupancy(),
            "tokens_pushed": self._ring.total_tokens_pushed(),
        }

    def _drain(self):
        while (descriptor := self._ring.pop()) is not None:
            descriptor.release()

    def run(self, requests, encode, consume, *, overlap: bool, prepare=None):
        requests = list(requests)
        results = []
        condition = threading.Condition()
        stop = threading.Event()
        done = threading.Event()
        errors = []

        def produce():
            try:
                for index, request in enumerate(requests):
                    if stop.is_set():
                        break
                    descriptor = encode(index, request)
                    with condition:
                        condition.wait_for(lambda: stop.is_set() or not self._ring.is_full())
                        if stop.is_set():
                            descriptor.release()
                            break
                        self._ring.push(descriptor)
                        condition.notify_all()
            except BaseException as exc:
                errors.append(exc)
            finally:
                with condition:
                    done.set()
                    condition.notify_all()

        producer = threading.Thread(target=produce, name="crossinfer-visual") if overlap else None
        if producer:
            producer.start()
        try:
            for index, request in enumerate(requests):
                if prepare is not None:
                    prepare(index, request)
                if producer:
                    with condition:
                        condition.wait_for(lambda: done.is_set() or not self._ring.is_empty())
                        descriptor = self._ring.pop()
                        condition.notify_all()
                    if descriptor is None:
                        if errors:
                            raise errors[0]
                        raise RuntimeError("visual producer ended before all requests")
                else:
                    descriptor = encode(index, request)
                    self._ring.push(descriptor)
                    descriptor = self._ring.pop()
                try:
                    results.append(consume(descriptor, request))
                finally:
                    descriptor.release()
            if errors:
                raise errors[0]
            return results
        finally:
            with condition:
                stop.set()
                self._drain()
                condition.notify_all()
            if producer:
                producer.join()
            self._drain()

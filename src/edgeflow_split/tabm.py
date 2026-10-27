from threading import Lock


class TabmRingBuffer:
    """Bounded single-producer/single-consumer ring buffer for in-process payloads."""

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int):
            raise TypeError("capacity must be an int")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._buffer: list[bytes | None] = [None] * capacity
        self._read_index = 0
        self._write_index = 0
        self._occupancy = 0
        self._max_occupancy = 0
        self._blocked_push_count = 0
        self._blocked_pop_count = 0
        self._lock = Lock()

    def capacity(self) -> int:
        return self._capacity

    def occupancy(self) -> int:
        with self._lock:
            return self._occupancy

    def max_occupancy(self) -> int:
        with self._lock:
            return self._max_occupancy

    def blocked_push_count(self) -> int:
        with self._lock:
            return self._blocked_push_count

    def blocked_pop_count(self) -> int:
        with self._lock:
            return self._blocked_pop_count

    def is_full(self) -> bool:
        with self._lock:
            return self._occupancy == self._capacity

    def is_empty(self) -> bool:
        with self._lock:
            return self._occupancy == 0

    def push(self, payload: bytes) -> bool:
        with self._lock:
            if self._occupancy == self._capacity:
                self._blocked_push_count += 1
                return False
            self._buffer[self._write_index] = bytes(payload)
            self._write_index = (self._write_index + 1) % self._capacity
            self._occupancy += 1
            if self._occupancy > self._max_occupancy:
                self._max_occupancy = self._occupancy
            return True

    def pop(self) -> bytes | None:
        with self._lock:
            if self._occupancy == 0:
                self._blocked_pop_count += 1
                return None
            payload = self._buffer[self._read_index]
            self._buffer[self._read_index] = None
            self._read_index = (self._read_index + 1) % self._capacity
            self._occupancy -= 1
            return payload

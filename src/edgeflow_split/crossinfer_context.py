from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Protocol, Self

from edgeflow_split.crossinfer_session import SequenceInfo, StateCompatibilitySignature


@dataclass(frozen=True, order=True)
class LaneResidency:
    lane_id: str
    session_incarnation_id: str
    sequence_id: int
    compatibility: StateCompatibilitySignature


@dataclass(frozen=True)
class StreamVersion:
    stream_id: str
    version_id: int
    source_version_id: int | None
    end_position: int
    event_time_ns: int
    lane_residencies: tuple[LaneResidency, ...]
    reference_count: int


@dataclass(frozen=True)
class LocalResponseBranch:
    stream_id: str
    source_version_id: int
    source_end_position: int
    lane_id: str
    sequence_id: int


class LocalForkSession(Protocol):
    @property
    def signature(self) -> StateCompatibilitySignature: ...

    @property
    def session_incarnation_id(self) -> str: ...

    def bind_stream(self, stream_id: str) -> None: ...

    def sequence_info(self, sequence_id: int) -> SequenceInfo: ...

    def fork_prefix(
        self, source_sequence_id: int, destination_sequence_id: int, end_position: int
    ) -> None: ...

    def remove_sequence(self, sequence_id: int) -> None: ...

    def close(self) -> None: ...


class VersionLease:
    def __init__(
        self, registry: StreamVersionRegistry, stream_id: str, version_id: int
    ) -> None:
        self._registry = registry
        self.stream_id = stream_id
        self.version_id = version_id
        self._released = False

    @property
    def version(self) -> StreamVersion:
        if self._released:
            raise RuntimeError("stream-version lease is released")
        return self._registry.get(self.stream_id, self.version_id)

    def release(self) -> None:
        if self._released:
            return
        self._registry.release(self.stream_id, self.version_id)
        self._released = True

    def __enter__(self) -> Self:
        if self._released:
            raise RuntimeError("stream-version lease is released")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.release()


class StreamVersionRegistry:
    """Single-owner immutable version table for snapshot-bound responses."""

    def __init__(self) -> None:
        self._owner_thread = threading.get_ident()
        self._versions: dict[str, dict[int, StreamVersion]] = {}
        self._latest: dict[str, int] = {}

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError(
                "stream-version registry must be used by its creating thread"
            )

    def _build_next_version(
        self, stream_id: str, *, end_position: int, event_time_ns: int
    ) -> StreamVersion:
        if not stream_id:
            raise ValueError("stream_id must not be empty")
        if end_position <= 0:
            raise ValueError("end_position must be positive")
        if event_time_ns < 0:
            raise ValueError("event_time_ns must be non-negative")

        source_version_id = self._latest.get(stream_id)
        previous = (
            self._versions[stream_id][source_version_id]
            if source_version_id is not None
            else None
        )
        if previous is not None:
            if end_position <= previous.end_position:
                raise ValueError("published end_position must advance the stream")
            if event_time_ns < previous.event_time_ns:
                raise ValueError("published event_time_ns must be monotonic")
            version_id = previous.version_id + 1
        else:
            version_id = 0

        return StreamVersion(
            stream_id=stream_id,
            version_id=version_id,
            source_version_id=source_version_id,
            end_position=end_position,
            event_time_ns=event_time_ns,
            lane_residencies=(),
            reference_count=0,
        )

    def _commit_version(self, version: StreamVersion) -> StreamVersion:
        self._versions.setdefault(version.stream_id, {})[version.version_id] = version
        self._latest[version.stream_id] = version.version_id
        return version

    def publish(
        self, stream_id: str, *, end_position: int, event_time_ns: int
    ) -> StreamVersion:
        self._require_owner()
        return self._commit_version(
            self._build_next_version(
                stream_id, end_position=end_position, event_time_ns=event_time_ns
            )
        )

    def publish_resident(
        self,
        stream_id: str,
        *,
        end_position: int,
        event_time_ns: int,
        lane_id: str,
        session_incarnation_id: str,
        sequence_id: int,
        compatibility: StateCompatibilitySignature,
    ) -> StreamVersion:
        """Publish only after the lane materialization has been validated."""
        self._require_owner()
        candidate = self._build_next_version(
            stream_id, end_position=end_position, event_time_ns=event_time_ns
        )
        resident = self._with_lane_residency(
            candidate,
            lane_id=lane_id,
            session_incarnation_id=session_incarnation_id,
            sequence_id=sequence_id,
            compatibility=compatibility,
        )
        return self._commit_version(resident)

    def latest(self, stream_id: str) -> StreamVersion:
        self._require_owner()
        try:
            return self._versions[stream_id][self._latest[stream_id]]
        except KeyError as exc:
            raise KeyError(f"unknown stream: {stream_id}") from exc

    def get(self, stream_id: str, version_id: int) -> StreamVersion:
        self._require_owner()
        try:
            return self._versions[stream_id][version_id]
        except KeyError as exc:
            raise KeyError(f"unknown stream version: {stream_id}@{version_id}") from exc

    def acquire(
        self, stream_id: str, *, event_time_ns: int | None = None
    ) -> VersionLease:
        self._require_owner()
        versions = self._versions.get(stream_id)
        if not versions:
            raise KeyError(f"unknown stream: {stream_id}")
        if event_time_ns is None:
            version_id = self._latest[stream_id]
        else:
            eligible = [
                version.version_id
                for version in versions.values()
                if version.event_time_ns <= event_time_ns
            ]
            if not eligible:
                raise LookupError(
                    f"stream {stream_id} has no version at event time {event_time_ns}"
                )
            version_id = max(eligible)
        version = versions[version_id]
        versions[version_id] = replace(
            version, reference_count=version.reference_count + 1
        )
        return VersionLease(self, stream_id, version_id)

    def release(self, stream_id: str, version_id: int) -> StreamVersion:
        self._require_owner()
        version = self.get(stream_id, version_id)
        if version.reference_count <= 0:
            raise RuntimeError("stream version released without a matching acquire")
        updated = replace(version, reference_count=version.reference_count - 1)
        self._versions[stream_id][version_id] = updated
        return updated

    def record_lane_residency(
        self,
        stream_id: str,
        version_id: int,
        *,
        lane_id: str,
        session_incarnation_id: str,
        sequence_id: int,
        compatibility: StateCompatibilitySignature,
    ) -> StreamVersion:
        self._require_owner()
        version = self.get(stream_id, version_id)
        updated = self._with_lane_residency(
            version,
            lane_id=lane_id,
            session_incarnation_id=session_incarnation_id,
            sequence_id=sequence_id,
            compatibility=compatibility,
        )
        self._versions[stream_id][version_id] = updated
        return updated

    def _with_lane_residency(
        self,
        version: StreamVersion,
        *,
        lane_id: str,
        session_incarnation_id: str,
        sequence_id: int,
        compatibility: StateCompatibilitySignature,
    ) -> StreamVersion:
        if not lane_id:
            raise ValueError("lane_id must not be empty")
        if not session_incarnation_id:
            raise ValueError("session_incarnation_id must not be empty")
        if sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        for other_stream_id, versions in self._versions.items():
            for other_version in versions.values():
                for other in other_version.lane_residencies:
                    if (
                        other.lane_id != lane_id
                        or other.session_incarnation_id != session_incarnation_id
                        or other.sequence_id != sequence_id
                    ):
                        continue
                    if other_stream_id != version.stream_id:
                        raise RuntimeError(
                            f"lane {lane_id} sequence {sequence_id} already materializes "
                            "another stream"
                        )
                    if other.compatibility != compatibility:
                        raise RuntimeError(
                            f"lane {lane_id} sequence {sequence_id} has incompatible state"
                        )
        by_lane = {
            residency.lane_id: residency for residency in version.lane_residencies
        }
        existing = by_lane.get(lane_id)
        residency = LaneResidency(
            lane_id=lane_id,
            session_incarnation_id=session_incarnation_id,
            sequence_id=sequence_id,
            compatibility=compatibility,
        )
        if existing is not None and existing != residency:
            raise RuntimeError(
                f"lane {lane_id} already holds a different materialization"
            )
        by_lane[lane_id] = residency
        return replace(version, lane_residencies=tuple(sorted(by_lane.values())))

    def remove_lane_residency(
        self,
        stream_id: str,
        version_id: int,
        lane_id: str,
        *,
        session_incarnation_id: str,
    ) -> StreamVersion:
        self._require_owner()
        version = self.get(stream_id, version_id)
        if version.reference_count != 0:
            raise RuntimeError(
                "cannot remove residency while the stream version is referenced"
            )
        residency = next(
            (item for item in version.lane_residencies if item.lane_id == lane_id),
            None,
        )
        if residency is None:
            raise KeyError(
                f"lane {lane_id} does not hold stream version {stream_id}@{version_id}"
            )
        if residency.session_incarnation_id != session_incarnation_id:
            raise RuntimeError(
                f"session incarnation {session_incarnation_id} does not own lane "
                f"{lane_id} residency for {stream_id}@{version_id}"
            )
        remaining = tuple(
            item for item in version.lane_residencies if item.lane_id != lane_id
        )
        updated = replace(version, lane_residencies=remaining)
        self._versions[stream_id][version_id] = updated
        return updated


class LocalContextCoordinator:
    """Bind immutable stream versions to one local unified-KV session.

    This class implements publish, acquire, fork, and release. It deliberately
    does not merge response KV, move state across backends, or batch decode.
    """

    def __init__(
        self,
        registry: StreamVersionRegistry,
        *,
        lane_id: str,
        session: LocalForkSession,
    ) -> None:
        if not lane_id:
            raise ValueError("lane_id must not be empty")
        self._registry = registry
        self._lane_id = lane_id
        self._session = session

    def publish_from_sequence(
        self,
        stream_id: str,
        *,
        source_sequence_id: int,
        event_time_ns: int,
    ) -> StreamVersion:
        if source_sequence_id != 0:
            raise ValueError("Only canonical root sequence zero can be published")
        info = self._session.sequence_info(source_sequence_id)
        if not info.active or info.next_position <= 0:
            raise RuntimeError("cannot publish an inactive or empty source sequence")
        self._session.bind_stream(stream_id)
        return self._registry.publish_resident(
            stream_id,
            end_position=info.next_position,
            event_time_ns=event_time_ns,
            lane_id=self._lane_id,
            session_incarnation_id=self._session.session_incarnation_id,
            sequence_id=source_sequence_id,
            compatibility=self._session.signature,
        )

    @contextmanager
    def response_branch(
        self,
        stream_id: str,
        *,
        destination_sequence_id: int,
        query_event_time_ns: int | None = None,
    ) -> Iterator[LocalResponseBranch]:
        lease = self._registry.acquire(stream_id, event_time_ns=query_event_time_ns)
        try:
            version = lease.version
            residency = next(
                (
                    item
                    for item in version.lane_residencies
                    if item.lane_id == self._lane_id
                    and item.session_incarnation_id
                    == self._session.session_incarnation_id
                ),
                None,
            )
            if residency is None:
                raise RuntimeError(
                    f"stream version {stream_id}@{version.version_id} is not resident "
                    f"in local session {self._session.session_incarnation_id} "
                    f"on lane {self._lane_id}"
                )
            if residency.compatibility != self._session.signature:
                raise RuntimeError(
                    "lane materialization compatibility does not match the local session"
                )
            self._session.bind_stream(stream_id)
            self._session.fork_prefix(
                residency.sequence_id,
                destination_sequence_id,
                version.end_position,
            )
        except BaseException:
            lease.release()
            raise

        branch = LocalResponseBranch(
            stream_id=stream_id,
            source_version_id=version.version_id,
            source_end_position=version.end_position,
            lane_id=self._lane_id,
            sequence_id=destination_sequence_id,
        )
        body_error: BaseException | None = None
        try:
            yield branch
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            try:
                info = self._session.sequence_info(destination_sequence_id)
                if info.active:
                    self._session.remove_sequence(destination_sequence_id)
            except BaseException as cleanup_error:
                primary_error = body_error if body_error is not None else cleanup_error
                if body_error is not None:
                    body_error.add_note(
                        f"local response branch cleanup also failed: {cleanup_error}"
                    )
                try:
                    self._session.close()
                except BaseException as close_error:
                    primary_error.add_note(
                        f"local response session close also failed: {close_error}"
                    )
                if body_error is None:
                    raise
            finally:
                lease.release()

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OccupancySnapshot:
    label: str
    occupancy: int
    max_occupancy: int
    blocked_push_count: int
    blocked_pop_count: int


@dataclass(frozen=True)
class TimelineEvent:
    stage: str
    frame_index: int
    worker: str
    start_ms: float
    end_ms: float


@dataclass
class RuntimeTrace:
    """Lightweight local runtime trace for stage timings and occupancy snapshots."""

    stage_times_ms: dict[str, float] = field(default_factory=dict)
    occupancy_snapshots: list[OccupancySnapshot] = field(default_factory=list)
    timeline_events: list[TimelineEvent] = field(default_factory=list)

    def record(self, name: str, value: float) -> None:
        self.stage_times_ms[name] = value

    def add_time(self, name: str, value: float) -> float:
        self.stage_times_ms[name] = self.stage_times_ms.get(name, 0.0) + value
        return self.stage_times_ms[name]

    def add_occupancy_snapshot(
        self,
        *,
        label: str,
        occupancy: int,
        max_occupancy: int,
        blocked_push_count: int,
        blocked_pop_count: int,
    ) -> OccupancySnapshot:
        snapshot = OccupancySnapshot(
            label=label,
            occupancy=occupancy,
            max_occupancy=max_occupancy,
            blocked_push_count=blocked_push_count,
            blocked_pop_count=blocked_pop_count,
        )
        self.occupancy_snapshots.append(snapshot)
        return snapshot

    def add_timeline_event(
        self,
        *,
        stage: str,
        frame_index: int,
        worker: str,
        start_ms: float,
        end_ms: float,
    ) -> TimelineEvent:
        event = TimelineEvent(
            stage=stage,
            frame_index=frame_index,
            worker=worker,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        self.timeline_events.append(event)
        return event

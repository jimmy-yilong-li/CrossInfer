"""Exercise the production descriptor queue without accelerator hardware."""

import threading

import pytest

from edgeflow_split.surface_backed_descriptor import SurfaceBackedVisualDescriptor


class Lease:
    def __init__(self):
        self.releases = 0

    def release(self):
        self.releases += 1


def descriptor(index):
    return SurfaceBackedVisualDescriptor(
        descriptor_id=index, surface_lease=Lease(), token_count=144,
        hidden_size=1024, dtype="float32", image_grid_thw=(1, 24, 24),
        frame_id=index, timestamp_ms=0,
    )


@pytest.mark.parametrize("overlap", [False, True])
def test_tabm_transports_original_descriptor_and_releases_after_consumption(overlap):
    from edgeflow_split.overlap_pipeline.surface_session import SurfacePipeline

    original = [descriptor(index) for index in range(4)]
    leases = [item.surface_lease for item in original]
    owner = threading.get_ident()

    def consume(item, request):
        assert threading.get_ident() == owner
        assert item is original[request]
        assert not item.released
        return item.frame_id

    pipeline = SurfacePipeline(capacity=2)
    assert pipeline.run(range(4), lambda index, _: original[index], consume,
                        overlap=overlap) == list(range(4))
    assert [lease.releases for lease in leases] == [1] * 4
    assert pipeline.metrics["tokens_pushed"] == 576
    assert pipeline.metrics["max_occupancy"] <= 2
    assert pipeline.metrics["occupancy"] == 0


def test_producer_continues_while_consumer_uses_previous_descriptor():
    from edgeflow_split.overlap_pipeline.surface_session import SurfacePipeline

    first_consuming, second_produced = threading.Event(), threading.Event()

    def encode(index, request):
        if index == 1:
            assert first_consuming.wait(2)
            second_produced.set()
        return descriptor(index)

    def consume(item, request):
        if request == 0:
            first_consuming.set()
            assert second_produced.wait(2)
        return request

    assert SurfacePipeline(2).run(range(2), encode, consume, overlap=True) == [0, 1]


def test_owner_can_prepare_current_input_before_visual_descriptor_is_ready():
    from edgeflow_split.overlap_pipeline.surface_session import SurfacePipeline

    owner = threading.get_ident()
    prepared = [threading.Event() for _ in range(2)]
    published = []

    def encode(index, request):
        assert prepared[index].wait(2)
        return descriptor(index)

    def prepare(index, request):
        assert threading.get_ident() == owner
        assert published == list(range(index))
        prepared[index].set()

    def consume(item, request):
        published.append(request)
        return request

    assert SurfacePipeline(2).run(range(2), encode, consume,
                                 overlap=True, prepare=prepare) == [0, 1]


@pytest.mark.parametrize("which", ["producer", "consumer"])
def test_failure_drains_all_owned_descriptors(which):
    from edgeflow_split.overlap_pipeline.surface_session import SurfacePipeline

    created, leases = [], []

    def encode(index, request):
        if which == "producer" and index == 1:
            raise ValueError("producer failed")
        item = descriptor(index)
        created.append(item)
        leases.append(item.surface_lease)
        return item

    def consume(item, request):
        if which == "consumer":
            raise ValueError("consumer failed")
        return request

    pipeline = SurfacePipeline(1)
    with pytest.raises(ValueError, match=f"{which} failed"):
        pipeline.run(range(4), encode, consume, overlap=True)
    assert all(item.released for item in created)
    assert all(lease.releases == 1 for lease in leases)
    assert pipeline.metrics["occupancy"] == 0

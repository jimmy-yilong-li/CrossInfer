from types import SimpleNamespace

from edgeflow_split.surface_backed_descriptor import SurfaceBackedVisualDescriptor


def test_context_uses_published_version_whole_branch_and_fixed_decode_budget():
    from edgeflow_split.qwen35_surface_context import Qwen35SurfaceContext

    class Session:
        signature = "test-compatible-state"
        session_incarnation_id = "test-incarnation"

        def __init__(self):
            self.calls = []
            self.position = 0
            self.branch = False

        def bind_stream(self, stream_id):
            self.calls.append(("bind", stream_id))

        def sequence_info(self, sequence):
            return SimpleNamespace(active=sequence == 0 or self.branch,
                                   next_position=self.position)

        def append_qwen35_visual(self, sequence, *, descriptor):
            assert sequence == 0
            self.position += 20
            descriptor.release()
            return SimpleNamespace(surface_pointer=4096, decoder_pointer=4096,
                                   payload_bytes=576, next_position=self.position)

        def fork_prefix(self, source, destination, end):
            assert end == self.position
            self.calls.append(("fork", source, destination, end))
            self.branch = True

        def ingest_qwen35_query(self, sequence, prompt):
            self.calls.append(("query", sequence, prompt))
            return 4

        def peek_token(self, sequence):
            return 7, True

        def ingest_tokens(self, sequence, tokens):
            self.calls.append(("advance", sequence, tokens))

        def token_piece_bytes(self, token):
            return b"x"

        def remove_sequence(self, sequence):
            self.calls.append(("drop", sequence))
            self.branch = False

    session = Session()
    context = Qwen35SurfaceContext(session, stream_id="stream")
    rows = []
    for index in range(2):
        item = SurfaceBackedVisualDescriptor(index, None, 144, 1, "float32",
                                            (1, 24, 24), index, 1000 + index,
                                            handoff_receipt={"producer_pointer": 4096})
        rows.append(context.answer(item, "Describe.", budget=3, arrival_s=0))
    assert [row["context_version"] for row in rows] == [0, 1]
    assert [row["context_event_time_ns"] for row in rows] == [1000000000, 1001000000]
    assert [row["output_ids"] for row in rows] == [[7, 7, 7]] * 2
    assert len([call for call in session.calls if call[0] == "advance"]) == 4
    assert [call for call in session.calls if call[0] == "drop"] == [("drop", 1)] * 2
    assert session.position == 40
    assert all(row["surface_handoff"]["pointer_equal"] for row in rows)

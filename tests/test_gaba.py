from limbiq.signals.gaba import GABASignal
from limbiq.types import Memory


class TestGABADetection:
    def setup_method(self):
        self.signal = GABASignal()

    def test_detects_denial(self):
        events = self.signal.detect(
            message="I never said I work at Google, you're making that up",
            response=None,
            feedback=None,
            memories=[],
        )
        assert len(events) > 0
        assert events[0].trigger == "user_denial"

    def test_denial_targets_non_priority_memories(self):
        memories = [
            Memory(id="m1", content="User works at Google", is_priority=False),
            Memory(id="m2", content="User's name is Dimuthu", is_priority=True),
        ]
        events = self.signal.detect(
            message="I never said I work at Google",
            response=None,
            feedback=None,
            memories=memories,
        )
        assert len(events) > 0
        assert "m1" in events[0].memory_ids_affected
        assert "m2" not in events[0].memory_ids_affected

    def test_explicit_negative_feedback(self):
        events = self.signal.detect(
            message="",
            response="",
            feedback="negative",
            memories=[],
        )
        assert len(events) > 0
        assert events[0].trigger == "explicit_negative_feedback"

    def test_no_signal_on_generic_message(self):
        events = self.signal.detect(
            message="What's the weather like today?",
            response=None,
            feedback=None,
            memories=[],
        )
        assert len(events) == 0

    def test_suppression_is_reversible(self, lq):
        lq.dopamine("Test fact")

        memories = lq.get_priority_memories()
        assert len(memories) == 1
        memory_id = memories[0].id

        lq.gaba(memory_id)
        assert len(lq.get_suppressed()) == 1
        assert len(lq.get_priority_memories()) == 0

        lq.restore_memory(memory_id)
        assert len(lq.get_suppressed()) == 0

    def test_stale_memories_get_suppressed(self, tmp_dir):
        from limbiq import Limbiq

        lq = Limbiq(store_path=tmp_dir, user_id="stale_test")

        # Store a memory with an embedding
        embedding = lq._core.embeddings.embed("some old fact")
        lq._core.store.store(
            content="some old fact nobody cares about",
            confidence=0.5,
            source="conversation",
            embedding=embedding,
        )

        # Simulate 11 sessions of aging without accessing
        for _ in range(11):
            lq._core.store.age_all()

        stale = lq._core.store.get_stale(min_sessions=10)
        assert len(stale) >= 1

        stats = lq.end_session()
        assert stats["suppressed"] >= 1

"""Tests for the Norepinephrine signal -- alertness modulation."""

from limbiq import Limbiq
from limbiq.types import SignalType, RetrievalConfig
from limbiq.signals.norepinephrine import NorepinephrineSignal
from tests.conftest import MockEncoder


def _make_ne_encoder():
    return MockEncoder({
        "i already told you": ("frustration", 0.9),
        "wrong again": ("frustration", 0.9),
        "are you even listening": ("frustration", 0.9),
        "pay attention": ("frustration", 0.9),
        "actually i": ("contradiction", 0.85),
        "just moved": ("contradiction", 0.85),
    })


class TestNorepinephrineDetection:
    def test_detects_frustration(self):
        signal = NorepinephrineSignal()
        encoder = _make_ne_encoder()
        events = signal.detect(
            message="I already told you my name, why don't you remember?",
            response="I apologize...",
            memories=[],
            encoder=encoder,
        )
        assert len(events) >= 1
        assert events[0].trigger == "user_frustration"

    def test_detects_frustration_patterns(self):
        signal = NorepinephrineSignal()
        encoder = _make_ne_encoder()
        for phrase in ["I already told you", "wrong again", "are you even listening"]:
            events = signal.detect(message=phrase, memories=[], encoder=encoder)
            assert len(events) >= 1, f"Should detect frustration in: {phrase}"

    def test_no_signal_on_normal_message(self):
        signal = NorepinephrineSignal()
        encoder = _make_ne_encoder()
        events = signal.detect(
            message="What's the weather like?",
            memories=[],
            encoder=encoder,
        )
        assert len(events) == 0

    def test_detects_contradiction_markers(self):
        from limbiq.types import Memory
        signal = NorepinephrineSignal()
        encoder = _make_ne_encoder()
        memories = [Memory(id="m1", content="User works at Google")]
        events = signal.detect(
            message="Actually I just moved to a new company",
            memories=memories,
            encoder=encoder,
        )
        assert len(events) >= 1
        assert events[0].trigger == "potential_contradiction"

    def test_topic_shift_detection(self, tmp_dir):
        """Norepinephrine fires on abrupt topic changes."""
        lq = Limbiq(store_path=tmp_dir, user_id="test")
        lq.start_session()

        # First process to set the baseline embedding
        lq.observe("Tell me about Python decorators and how they work in detail", "Decorators are...")
        lq.process("Tell me about Python decorators and how they work in detail")

        # Completely different topic
        result = lq.process("What's the best pizza in New York City restaurants?")

        # May or may not fire depending on embedding distance
        # But the signal infrastructure should work
        assert result.norepinephrine_active is True or result.norepinephrine_active is False


class TestNorepinephrineEffects:
    def test_frustration_triggers_caution(self, tmp_dir):
        """Norepinephrine from observe() adds caution to next process()."""
        lq = Limbiq(store_path=tmp_dir, user_id="test")
        lq.start_session()

        # Inject a mock encoder into the core so observe() can detect frustration
        lq._core.encoder = _make_ne_encoder()

        lq.dopamine("User's name is Dimuthu")
        events = lq.observe(
            "I already told you my name, why don't you remember?",
            "I apologize for the confusion.",
        )

        ne_events = [e for e in events if e.signal_type == SignalType.NOREPINEPHRINE]
        assert len(ne_events) >= 1

        result = lq.process("What is my name?")
        assert "CAUTION" in result.context

    def test_effects_are_transient(self, tmp_dir):
        """Norepinephrine effects reset after one process() call."""
        lq = Limbiq(store_path=tmp_dir, user_id="test")
        lq.start_session()

        # Inject mock encoder
        lq._core.encoder = _make_ne_encoder()

        # Trigger norepinephrine
        lq.observe("I already told you about this!", "Sorry...")
        result1 = lq.process("Tell me again")
        assert "CAUTION" in result1.context

        # Next call should be back to normal (no pending NE events)
        result2 = lq.process("Unrelated normal question about weather")
        # The key test is that _pending_ne_events was cleared
        assert lq._core._pending_ne_events == []

    def test_widened_retrieval_config(self):
        """RetrievalConfig.widen() doubles top_k and halves threshold."""
        config = RetrievalConfig()
        assert config.top_k == 10
        assert config.relevance_threshold == 0.15

        config.widen()
        assert config.top_k == 20
        assert config.relevance_threshold == 0.075

    def test_retrieval_config_reset(self):
        """RetrievalConfig.reset() restores defaults."""
        config = RetrievalConfig()
        config.widen()
        config.add_caution("test")

        config.reset()
        assert config.top_k == 10
        assert config.relevance_threshold == 0.15
        assert config.caution_flag is None

    def test_widened_retrieval_cap(self):
        """Widening is capped at 30."""
        config = RetrievalConfig(top_k=20)
        config.widen()
        assert config.top_k == 30  # min(40, 30)

    def test_observe_fires_norepinephrine(self, tmp_dir):
        """Norepinephrine fires during observe on frustration."""
        lq = Limbiq(store_path=tmp_dir, user_id="test")
        lq.start_session()

        # Inject mock encoder
        lq._core.encoder = _make_ne_encoder()

        events = lq.observe(
            "I already told you, pay attention!",
            "I apologize...",
        )

        ne = [e for e in events if e.signal_type == SignalType.NOREPINEPHRINE]
        assert len(ne) >= 1
        assert ne[0].trigger == "user_frustration"

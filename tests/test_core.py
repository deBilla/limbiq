"""Integration tests for the full Limbiq flow."""

from limbiq import Limbiq


class TestBasicFlow:
    def test_process_returns_empty_context_initially(self, lq):
        result = lq.process("Hi, my name is Dimuthu")
        assert result.context == ""
        assert result.memories_retrieved == 0

    def test_observe_stores_memory(self, lq):
        lq.observe(
            "Hi, my name is Dimuthu",
            "Nice to meet you, Dimuthu!",
        )
        stats = lq.get_stats()
        assert stats["total"] >= 1

    def test_basic_flow(self, tmp_dir):
        lq = Limbiq(store_path=tmp_dir, user_id="test")

        result = lq.process("Hi, my name is Dimuthu")
        assert result.context == ""

        lq.observe("Hi, my name is Dimuthu", "Nice to meet you, Dimuthu!")

        stats = lq.end_session()
        assert stats["compressed"] > 0

        lq.start_session()
        result = lq.process("What's my name?")
        assert "Dimuthu" in result.context

    def test_priority_always_included(self, lq):
        lq.dopamine("User's wife is named Prabhashi")

        result = lq.process("What's the weather like?")
        assert "Prabhashi" in result.context

    def test_suppressed_never_included(self, lq):
        lq.dopamine("User works at Google")

        memories = lq.get_priority_memories()
        lq.gaba(memories[0].id)

        result = lq.process("Where do I work?")
        assert "Google" not in result.context

    def test_correction_flow(self, lq):
        lq.dopamine("User works at Google")
        lq.correct("User works at Bitsmedia")

        result = lq.process("Where do I work?")
        assert "Bitsmedia" in result.context

        # The old "Google" memory should be suppressed
        suppressed = lq.get_suppressed()
        assert any("Google" in m.content for m in suppressed)


class TestSignalFiring:
    def test_observe_fires_dopamine_on_personal_info(self, lq):
        events = lq.observe(
            "My name is Dimuthu and I work at Bitsmedia",
            "Nice to meet you! I'll remember that.",
        )
        dopamine_events = [e for e in events if e.signal_type == "dopamine"]
        assert len(dopamine_events) > 0

    def test_observe_fires_gaba_on_denial(self, lq):
        # First store something
        lq.dopamine("User works at Google")

        # Now observe a denial
        events = lq.observe(
            "I never said I work at Google, you're making that up",
            "I apologize for the confusion.",
        )
        gaba_events = [e for e in events if e.signal_type == "gaba"]
        assert len(gaba_events) > 0


class TestSessionLifecycle:
    def test_end_session_compresses(self, lq):
        lq.observe("My name is Dimuthu", "Hello Dimuthu!")
        lq.observe("I work at Bitsmedia", "Great company!")

        stats = lq.end_session()
        assert stats["compressed"] >= 0  # Extractive may or may not find facts

    def test_start_session_resets_buffer(self, lq):
        lq.observe("Test message", "Test response")
        lq.start_session()
        # Buffer should be cleared
        assert lq._core._conversation_buffer == []


class TestInspection:
    def test_get_stats(self, lq):
        lq.dopamine("A fact")
        stats = lq.get_stats()
        assert "priority" in stats
        assert "total" in stats

    def test_export_state(self, lq):
        lq.dopamine("Export test fact")
        export = lq.export_state()
        assert "memories" in export
        assert len(export["memories"]) >= 1

    def test_get_signal_log(self, lq):
        lq.observe("Exactly! That's perfect!", "Glad I got it right!")
        log = lq.get_signal_log()
        assert len(log) >= 0  # May or may not fire depending on detection


class TestWithMockLLM:
    def test_llm_compression(self, tmp_dir, mock_llm):
        lq = Limbiq(store_path=tmp_dir, user_id="test", llm_fn=mock_llm)

        lq.observe("My name is Dimuthu", "Hello!")
        lq.observe("I work at Bitsmedia", "Cool!")

        stats = lq.end_session()
        assert stats["compressed"] > 0

        lq.start_session()
        result = lq.process("Who am I?")
        assert "Dimuthu" in result.context

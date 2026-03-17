from limbiq.context.builder import ContextBuilder
from limbiq.types import Memory, MemoryTier


class TestContextBuilder:
    def setup_method(self):
        self.builder = ContextBuilder()

    def test_empty_context(self):
        result = self.builder.build([], [], set())
        assert result == ""

    def test_priority_only(self):
        priority = [
            Memory(id="1", content="User's wife is Prabhashi", is_priority=True),
        ]
        result = self.builder.build(priority, [], set())
        assert "Prabhashi" in result
        assert "IMPORTANT" in result
        assert "<memory_context>" in result

    def test_relevant_only(self):
        relevant = [
            Memory(id="1", content="Discussed Python yesterday", confidence=0.8),
        ]
        result = self.builder.build([], relevant, set())
        assert "Python" in result
        assert "80% confidence" in result

    def test_priority_comes_first(self):
        priority = [
            Memory(id="1", content="PRIORITY FACT", is_priority=True),
        ]
        relevant = [
            Memory(id="2", content="RELEVANT FACT", confidence=0.7),
        ]
        result = self.builder.build(priority, relevant, set())
        priority_pos = result.index("PRIORITY FACT")
        relevant_pos = result.index("RELEVANT FACT")
        assert priority_pos < relevant_pos

    def test_suppressed_excluded(self):
        relevant = [
            Memory(id="1", content="Should appear", confidence=0.8),
            Memory(id="2", content="Should NOT appear", confidence=0.6),
        ]
        result = self.builder.build([], relevant, {"2"})
        assert "Should appear" in result
        assert "Should NOT appear" not in result

    def test_no_duplicate_priority_in_relevant(self):
        priority = [
            Memory(id="1", content="Same fact", is_priority=True),
        ]
        relevant = [
            Memory(id="1", content="Same fact", confidence=0.9),
            Memory(id="2", content="Other fact", confidence=0.7),
        ]
        result = self.builder.build(priority, relevant, set())
        assert result.count("Same fact") == 1

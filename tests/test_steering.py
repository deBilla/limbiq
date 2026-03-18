"""Tests for steered inference.

Integration tests requiring MLX are skipped if MLX is not available.
Mock-based tests verify the steering logic without a real model.
"""

import pytest

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class TestRetrievalConfigIntegration:
    """Verify steering metadata flows through process()."""

    def test_process_result_has_steering_fields(self, lq):
        """ProcessResult includes v0.2 steering-relevant fields."""
        result = lq.process("Hello")
        assert hasattr(result, "active_rules")
        assert hasattr(result, "clusters_loaded")
        assert hasattr(result, "norepinephrine_active")

    def test_signals_fired_in_process_result(self, lq):
        """Signals fired during process are included in result."""
        result = lq.process("Hello")
        assert isinstance(result.signals_fired, list)


class TestSteeredModelMock:
    """Test SteeredModel logic with mock vectors."""

    def test_add_and_remove_steer(self):
        from tests.test_bridge import MockSteeredModel

        model = MockSteeredModel()
        model.add_steer("test", {14: [0.0] * 16}, alpha=1.0)
        assert "test" in model.get_active_steers()

        model.remove_steer("test")
        assert "test" not in model.get_active_steers()

    def test_clear_steers(self):
        from tests.test_bridge import MockSteeredModel

        model = MockSteeredModel()
        model.add_steer("a", {}, 1.0)
        model.add_steer("b", {}, 1.0)
        assert len(model.get_active_steers()) == 2

        model.clear_steers()
        assert len(model.get_active_steers()) == 0

    def test_duplicate_name_replaces(self):
        from tests.test_bridge import MockSteeredModel

        model = MockSteeredModel()
        model.add_steer("test", {}, alpha=1.0)
        model.add_steer("test", {}, alpha=2.0)

        assert len(model.get_active_steers()) == 1
        assert model.active_steers[0]["alpha"] == 2.0


class TestEvaluatorStructure:
    """Test evaluator data structures (no model needed)."""

    def test_steering_result_word_counts(self):
        from limbiq.steering.evaluator import SteeringResult

        result = SteeringResult(
            prompt="test",
            baseline="one two three four five",
            positive="one two",
            negative="one two three four five six seven eight nine ten",
        )
        assert result.baseline_words == 5
        assert result.positive_words == 2
        assert result.negative_words == 10

    def test_dimension_evaluation_summary(self):
        from limbiq.steering.evaluator import DimensionEvaluation, SteeringResult

        results = [
            SteeringResult(
                prompt="test",
                baseline="word " * 100,
                positive="word " * 50,
                negative="word " * 200,
            )
        ]
        evaluation = DimensionEvaluation(dimension="conciseness", results=results)
        summary = evaluation.summary()

        assert summary["dimension"] == "conciseness"
        assert summary["avg_baseline_words"] == 100
        assert summary["avg_positive_words"] == 50
        assert summary["avg_negative_words"] == 200


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not installed")
class TestSteeredInferenceMLX:
    """Integration tests requiring MLX. Skipped in CI."""
    pass

"""Tests for the signal-to-steering bridge mapping.

These tests use mock objects and don't require MLX or a real model.
"""

import numpy as np
import pytest

from limbiq.types import SignalEvent, SignalType, BehavioralRule


class MockSteeredModel:
    """Mock SteeredModel that tracks steer operations."""

    def __init__(self):
        self.active_steers: list[dict] = []

    def add_steer(self, name, vectors, alpha=1.0):
        self.active_steers = [s for s in self.active_steers if s["name"] != name]
        self.active_steers.append({"name": name, "vectors": vectors, "alpha": alpha})

    def remove_steer(self, name):
        self.active_steers = [s for s in self.active_steers if s["name"] != name]

    def clear_steers(self):
        self.active_steers = []

    def get_active_steers(self):
        return [s["name"] for s in self.active_steers]


def _make_mock_vectors():
    """Create mock steering vectors for testing."""
    mock_vec = {14: np.zeros(16), 15: np.zeros(16), 16: np.zeros(16)}
    return {
        "memory_attention": {"vectors": mock_vec},
        "confidence": {"vectors": mock_vec},
        "honesty": {"vectors": mock_vec},
        "conciseness": {"vectors": mock_vec},
        "formality": {"vectors": mock_vec},
        "technical_depth": {"vectors": mock_vec},
        "creativity": {"vectors": mock_vec},
        "helpfulness": {"vectors": mock_vec},
    }


class TestBridgeSignalMapping:
    def setup_method(self):
        from limbiq.steering.bridge import LimbiqSteeringBridge

        self.model = MockSteeredModel()
        self.bridge = LimbiqSteeringBridge.__new__(LimbiqSteeringBridge)
        self.bridge.model = self.model
        self.bridge.vector_dir = "./vectors"
        self.bridge.vectors = _make_mock_vectors()

    def test_dopamine_activates_memory_attention(self):
        events = [SignalEvent(signal_type=SignalType.DOPAMINE, trigger="test")]
        self.bridge.apply_signals(signals_fired=events)

        names = self.model.get_active_steers()
        assert "dopamine_attention" in names
        assert "dopamine_confidence" in names

    def test_gaba_activates_honesty(self):
        events = [SignalEvent(signal_type=SignalType.GABA, trigger="test")]
        self.bridge.apply_signals(signals_fired=events)

        names = self.model.get_active_steers()
        assert "gaba_honesty" in names
        assert "gaba_uncertainty" in names

        # Confidence should be NEGATIVE alpha
        uncertainty = [s for s in self.model.active_steers if s["name"] == "gaba_uncertainty"]
        assert uncertainty[0]["alpha"] < 0

    def test_norepinephrine_activates_caution(self):
        events = [SignalEvent(signal_type=SignalType.NOREPINEPHRINE, trigger="test")]
        self.bridge.apply_signals(signals_fired=events)

        names = self.model.get_active_steers()
        assert "norepi_honesty" in names
        assert "norepi_attention" in names

    def test_norepinephrine_via_flag(self):
        """Norepinephrine active flag applies steering even without explicit events."""
        self.bridge.apply_signals(norepinephrine_active=True)

        names = self.model.get_active_steers()
        assert "norepi_honesty" in names

    def test_acetylcholine_activates_depth(self):
        events = [SignalEvent(signal_type=SignalType.ACETYLCHOLINE, trigger="test")]
        self.bridge.apply_signals(signals_fired=events)

        names = self.model.get_active_steers()
        assert "ach_depth" in names
        assert "ach_helpful" in names

    def test_acetylcholine_via_clusters(self):
        """Cluster loading triggers acetylcholine steering."""
        self.bridge.apply_signals(clusters_loaded=["python"])

        names = self.model.get_active_steers()
        assert "ach_depth" in names

    def test_serotonin_maps_concise_rule(self):
        rules = [BehavioralRule(
            pattern_key="prefers_concise",
            rule_text="Keep responses brief and concise.",
            confidence=0.9,
        )]
        self.bridge.apply_signals(active_rules=rules)

        names = self.model.get_active_steers()
        assert "serotonin_conciseness" in names

        steer = [s for s in self.model.active_steers if s["name"] == "serotonin_conciseness"]
        assert steer[0]["alpha"] > 0  # Positive for conciseness

    def test_serotonin_maps_casual_rule(self):
        rules = [BehavioralRule(
            pattern_key="casual_tone",
            rule_text="Use a casual, conversational tone.",
            confidence=1.0,
        )]
        self.bridge.apply_signals(active_rules=rules)

        names = self.model.get_active_steers()
        assert "serotonin_formality" in names

        steer = [s for s in self.model.active_steers if s["name"] == "serotonin_formality"]
        assert steer[0]["alpha"] < 0  # Negative = away from formal

    def test_multiple_signals_compose(self):
        """Multiple signals can be active simultaneously."""
        events = [
            SignalEvent(signal_type=SignalType.DOPAMINE, trigger="test"),
            SignalEvent(signal_type=SignalType.NOREPINEPHRINE, trigger="test"),
        ]
        rules = [BehavioralRule(
            pattern_key="concise",
            rule_text="Be concise and brief.",
            confidence=1.0,
        )]
        self.bridge.apply_signals(
            signals_fired=events,
            active_rules=rules,
        )

        names = self.model.get_active_steers()
        # Should have dopamine + norepinephrine + serotonin steers
        assert len(names) >= 4

    def test_clear_on_each_apply(self):
        """Steers are cleared before each apply_signals call."""
        self.bridge.apply_signals(
            signals_fired=[SignalEvent(signal_type=SignalType.DOPAMINE, trigger="test")]
        )
        assert len(self.model.get_active_steers()) >= 1

        # Second call with no signals should clear everything
        self.bridge.apply_signals()
        assert len(self.model.get_active_steers()) == 0

    def test_no_vectors_graceful(self):
        """Bridge works gracefully when no vectors are loaded."""
        self.bridge.vectors = {}
        self.bridge.apply_signals(
            signals_fired=[SignalEvent(signal_type=SignalType.DOPAMINE, trigger="test")]
        )
        assert len(self.model.get_active_steers()) == 0

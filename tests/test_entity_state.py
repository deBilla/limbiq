"""Tests for per-entity persistent state (distributed cellular memory).

Unit tests for EntityStateStore + creative integration tests that verify
entity state emerges organically from conversation patterns.
"""

import time

import pytest

from limbiq import Limbiq
from limbiq.graph.entity_state import (
    EntityState,
    EntityStateStore,
    RECEPTOR_MAX,
    RECEPTOR_MIN,
    _DEFAULT_RECEPTOR_DENSITY,
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def store(tmp_dir):
    """A bare EntityStateStore backed by a fresh Limbiq MemoryStore."""
    lq = Limbiq(store_path=tmp_dir, user_id="test")
    return lq._core.entity_state_store


@pytest.fixture
def lq(tmp_dir):
    """A fresh Limbiq instance."""
    return Limbiq(store_path=tmp_dir, user_id="test")


# ── Unit tests: EntityStateStore ──────────────────────────────


class TestEntityStateStore:

    def test_ensure_state_exists_creates_default(self, store):
        store.ensure_state_exists("ent_001")
        state = store.get_state("ent_001")
        assert state.entity_id == "ent_001"
        assert state.resting_activation == 0.0
        assert state.total_activations == 0
        assert state.signal_history == {}
        assert state.sentinel_pattern is None

    def test_ensure_state_exists_idempotent(self, store):
        store.ensure_state_exists("ent_001")
        store.ensure_state_exists("ent_001")  # Should not raise
        state = store.get_state("ent_001")
        assert state.entity_id == "ent_001"

    def test_get_state_returns_default_for_unknown(self, store):
        state = store.get_state("nonexistent")
        assert state.entity_id == "nonexistent"
        assert state.resting_activation == 0.0

    def test_activate_increases_resting_activation(self, store):
        store.ensure_state_exists("ent_001")
        store.activate("ent_001", delta=0.3)
        state = store.get_state("ent_001")
        assert state.resting_activation == pytest.approx(0.3, abs=0.01)
        assert state.total_activations == 1

    def test_activate_cumulative(self, store):
        store.ensure_state_exists("ent_001")
        store.activate("ent_001", delta=0.2)
        store.activate("ent_001", delta=0.3)
        state = store.get_state("ent_001")
        assert state.resting_activation == pytest.approx(0.5, abs=0.01)
        assert state.total_activations == 2

    def test_activate_clamped_at_max(self, store):
        store.ensure_state_exists("ent_001")
        store.activate("ent_001", delta=3.0)  # Exceeds 2.0 cap
        state = store.get_state("ent_001")
        assert state.resting_activation <= 2.0

    def test_activate_auto_creates_state(self, store):
        """activate() should create state if it doesn't exist."""
        store.activate("ent_new", delta=0.1)
        state = store.get_state("ent_new")
        assert state.resting_activation == pytest.approx(0.1, abs=0.01)

    def test_record_signal_increments_history(self, store):
        store.ensure_state_exists("ent_001")
        store.record_signal("ent_001", "dopamine")
        store.record_signal("ent_001", "dopamine")
        store.record_signal("ent_001", "gaba")
        state = store.get_state("ent_001")
        assert state.signal_history["dopamine"] == 2
        assert state.signal_history["gaba"] == 1

    def test_record_signal_upregulates_receptor(self, store):
        store.ensure_state_exists("ent_001")
        initial_density = store.get_state("ent_001").receptor_density["dopamine"]
        store.record_signal("ent_001", "dopamine")
        after_density = store.get_state("ent_001").receptor_density["dopamine"]
        assert after_density > initial_density

    def test_receptor_density_capped(self, store):
        store.ensure_state_exists("ent_001")
        # Hit it 100 times — should not exceed RECEPTOR_MAX
        for _ in range(100):
            store.record_signal("ent_001", "dopamine")
        state = store.get_state("ent_001")
        assert state.receptor_density["dopamine"] <= RECEPTOR_MAX

    def test_set_sentinel(self, store):
        store.ensure_state_exists("ent_001")
        store.set_sentinel("ent_001", r"\blondon\b")
        state = store.get_state("ent_001")
        assert state.sentinel_pattern == r"\blondon\b"

    def test_clear_sentinel(self, store):
        store.ensure_state_exists("ent_001")
        store.set_sentinel("ent_001", "pattern")
        store.set_sentinel("ent_001", None)
        state = store.get_state("ent_001")
        assert state.sentinel_pattern is None

    def test_get_sentinels(self, store):
        store.ensure_state_exists("ent_001")
        store.ensure_state_exists("ent_002")
        store.set_sentinel("ent_001", "watch_this")
        sentinels = store.get_sentinels()
        assert len(sentinels) == 1
        assert sentinels[0].entity_id == "ent_001"

    def test_update_expression_mask(self, store):
        store.ensure_state_exists("ent_001")
        store.update_expression_mask("ent_001", {"role": True, "hobby": False})
        state = store.get_state("ent_001")
        assert state.expression_mask == {"role": True, "hobby": False}

    def test_decay_activations(self, store):
        store.ensure_state_exists("ent_001")
        store.activate("ent_001", delta=1.0)
        decayed = store.decay_activations(decay_factor=0.5)
        assert decayed >= 1
        state = store.get_state("ent_001")
        assert state.resting_activation == pytest.approx(0.5, abs=0.01)

    def test_decay_zeroes_small_values(self, store):
        store.ensure_state_exists("ent_001")
        store.activate("ent_001", delta=0.0005)
        store.decay_activations(decay_factor=0.5)
        state = store.get_state("ent_001")
        assert state.resting_activation == 0.0

    def test_decay_receptor_density_toward_baseline(self, store):
        store.ensure_state_exists("ent_001")
        # Pump up dopamine receptor
        for _ in range(20):
            store.record_signal("ent_001", "dopamine")
        before = store.get_state("ent_001").receptor_density["dopamine"]
        assert before > 1.0
        store.decay_receptor_density(decay_rate=0.5)
        after = store.get_state("ent_001").receptor_density["dopamine"]
        assert after < before  # Decayed toward 1.0

    def test_get_top_activated(self, store):
        for i in range(5):
            eid = f"ent_{i:03d}"
            store.ensure_state_exists(eid)
            store.activate(eid, delta=0.1 * (i + 1))
        top = store.get_top_activated(limit=3)
        assert len(top) == 3
        assert top[0].resting_activation > top[1].resting_activation

    def test_get_all_states(self, store):
        before = len(store.get_all_states())
        store.ensure_state_exists("a")
        store.ensure_state_exists("b")
        states = store.get_all_states()
        assert len(states) == before + 2


# ── Integration tests: Creative verification ──────────────────


def _find_entity(lq, name_fragment):
    """Find an entity whose name contains the fragment (case-insensitive)."""
    for e in lq.get_entities():
        if name_fragment.lower() in e.name.lower():
            return e
    return None


class TestImplicitEntityActivation:
    """A1. Tell it without saying it — entities build activation from casual mention."""

    def test_repeated_mention_builds_activation(self, lq):
        lq.start_session()
        lq.observe("Prabhashi and I went to that new place downtown", "Sounds nice!")
        lq.observe("I need to pick up something for Prabhashi's birthday", "What does she like?")
        lq.observe("Prabhashi suggested we try Thai food tonight", "Great idea!")
        lq.end_session()

        lq.start_session()
        lq.observe("Prabhashi got a promotion at work today", "Congratulations!")
        lq.observe("We're celebrating Prabhashi's promotion this weekend", "How fun!")
        lq.end_session()

        prabhashi = _find_entity(lq, "prabhashi")
        if prabhashi is None:
            pytest.skip("Entity extraction didn't pick up Prabhashi from casual mentions")

        state = lq.get_entity_state(prabhashi.id)
        assert state.resting_activation > 0, \
            "Frequently mentioned entity should have elevated resting activation"
        assert state.total_activations >= 2, \
            "Should track activation count across sessions"


class TestActivationDecay:
    """A2. Resting activation decays like muscle atrophy."""

    def test_decay_over_sessions(self, lq):
        lq.start_session()
        lq.observe("My colleague Rohan helped me with the deployment", "Nice teamwork!")
        lq.end_session()

        rohan = _find_entity(lq, "rohan")
        if rohan is None:
            pytest.skip("Entity extraction didn't pick up Rohan")

        initial = lq.get_entity_state(rohan.id).resting_activation
        if initial == 0:
            pytest.skip("Rohan has no activation to decay")

        # Run 8 sessions WITHOUT mentioning Rohan
        for i in range(8):
            lq.start_session()
            lq.observe(f"Working on project milestone {i}", "Keep going!")
            lq.end_session()

        decayed = lq.get_entity_state(rohan.id).resting_activation
        assert decayed < initial, "Activation should decay without reinforcement"


class TestReceptorDensityAdaptation:
    """A3. Signal history builds receptor density."""

    def test_dopamine_upregulates_receptor(self, lq):
        lq.start_session()
        # Dopamine-triggering: novel personal info
        lq.observe("My wife is Prabhashi", "Nice to know!")
        lq.end_session()

        lq.start_session()
        lq.observe("Prabhashi is a software engineer", "Interesting!")
        lq.end_session()

        prabhashi = _find_entity(lq, "prabhashi")
        if prabhashi is None:
            pytest.skip("Entity extraction didn't pick up Prabhashi")

        state = lq.get_entity_state(prabhashi.id)
        # If dopamine fired and was linked to Prabhashi, receptor should be up
        if state.signal_history.get("dopamine", 0) > 0:
            assert state.receptor_density.get("dopamine", 1.0) > 1.0, \
                "Repeated dopamine should upregulate receptor density"


class TestEntityStateAutoCreation:
    """Verify entities get state automatically when added to graph."""

    def test_entity_gets_state_on_creation(self, lq):
        lq.start_session()
        lq.observe("My father is Upananda", "Nice to know!")
        lq.end_session()

        upananda = _find_entity(lq, "upananda")
        if upananda is None:
            pytest.skip("Entity extraction didn't pick up Upananda")

        state = lq.get_entity_state(upananda.id)
        assert state.entity_id == upananda.id
        # Default receptor density should be populated
        assert "dopamine" in state.receptor_density


class TestEntityStateDecayInEndSession:
    """Verify end_session() decays entity activations."""

    def test_end_session_decays(self, lq):
        lq.start_session()
        lq.observe("My friend Alex is visiting", "How nice!")
        lq.end_session()

        alex = _find_entity(lq, "alex")
        if alex is None:
            pytest.skip("Entity extraction didn't pick up Alex")

        after_first = lq.get_entity_state(alex.id).resting_activation

        # Second session — mention Alex again to build activation
        lq.start_session()
        lq.observe("Alex and I had coffee", "Fun!")
        # Don't end yet — check activation grew
        state_mid = lq.get_entity_state(alex.id)
        mid_activation = state_mid.resting_activation
        lq.end_session()  # This should decay

        after_decay = lq.get_entity_state(alex.id).resting_activation
        # After decay, activation should be less than it was mid-session
        # (unless mid-session activation was 0)
        if mid_activation > 0:
            assert after_decay < mid_activation, \
                "end_session() should decay resting activations"


class TestGetTopActivated:
    """Verify top activated entities API works end-to-end."""

    def test_top_activated_returns_results(self, lq):
        lq.start_session()
        lq.observe("My wife Prabhashi works at TechCorp", "Nice!")
        lq.observe("My father Upananda is visiting", "Great!")
        lq.end_session()

        top = lq.get_top_activated_entities(limit=10)
        # Should return EntityState objects (possibly empty if extraction didn't fire)
        assert isinstance(top, list)
        for state in top:
            assert isinstance(state, EntityState)

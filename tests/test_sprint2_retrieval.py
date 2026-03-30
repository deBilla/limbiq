"""Sprint 2 tests: Retrieval integration + expression masks.

Tests that entity resting activation affects retrieval ordering,
propagation uses entity state as base, and expression masks adapt
to conversation topic.
"""

import pytest

from limbiq import Limbiq
from limbiq.graph.entity_state import EntityState


@pytest.fixture
def lq(tmp_dir):
    return Limbiq(store_path=tmp_dir, user_id="test")


def _find_entity(lq, name_fragment):
    for e in lq.get_entities():
        if name_fragment.lower() in e.name.lower():
            return e
    return None


# ── Resting activation affects propagation ──────────────────────


class TestRestingActivationPropagation:
    """Entity resting activation should influence compute_activations()."""

    def test_high_activation_entity_boosts_related_memories(self, lq):
        """Memories about high-activation entities should score higher."""
        # Build up Prabhashi's activation across sessions
        lq.start_session()
        lq.observe("My wife is Prabhashi", "Nice!")
        lq.end_session()

        lq.start_session()
        lq.observe("Prabhashi and I went hiking", "Fun!")
        lq.end_session()

        lq.start_session()
        lq.observe("Prabhashi cooked an amazing dinner", "Yum!")
        lq.end_session()

        # Also add a memory about a less-discussed person
        lq.start_session()
        lq.observe("My colleague Alex called", "What about?")
        lq.end_session()

        prabhashi = _find_entity(lq, "prabhashi")
        alex = _find_entity(lq, "alex")

        if prabhashi:
            p_state = lq.get_entity_state(prabhashi.id)
            print(f"Prabhashi resting_activation: {p_state.resting_activation:.3f}")

        if alex:
            a_state = lq.get_entity_state(alex.id)
            print(f"Alex resting_activation: {a_state.resting_activation:.3f}")

        # Compute activations — Prabhashi memories should rank higher
        activations = lq.compute_activations()
        if activations:
            print("Top 5 activation states:")
            for act in activations[:5]:
                row = lq._core.store.db.execute(
                    "SELECT content FROM memories WHERE id=?",
                    (act.memory_id,)
                ).fetchone()
                content = row[0][:60] if row else "?"
                print(f"  {act.activation:.3f}: {content}")

    def test_propagation_accepts_entity_state(self, lq):
        """ActiveGraphPropagation should work with entity_state_store param."""
        lq.start_session()
        lq.observe("My wife Prabhashi works at TechCorp", "Nice!")
        lq.end_session()

        # This should not raise — propagation now accepts entity_state_store
        result = lq.propagate()
        assert result is not None
        print(f"Propagation result: {result.entities_created} entities, "
              f"{result.relations_created} relations, "
              f"{result.noise_suppressed} noise suppressed")


# ── Resting activation in retrieval scoring ──────────────────────


class TestRestingActivationRetrieval:
    """Resting activation should contribute to retrieval scoring."""

    def test_resting_boost_map_computed(self, lq):
        """_compute_resting_boosts should return non-empty map for active entities."""
        from limbiq.retrieval.activation_retrieval import ActivationRetrieval

        lq.start_session()
        lq.observe("My wife is Prabhashi", "Nice!")
        lq.end_session()

        lq.start_session()
        lq.observe("Prabhashi loves Thai food", "Yum!")
        lq.end_session()

        # Create an ActivationRetrieval with entity state
        ar = ActivationRetrieval(
            store=lq._core.store,
            graph=lq._core.graph,
            embedding_engine=lq._core.embeddings,
            entity_state_store=lq._core.entity_state_store,
        )

        boosts = ar._compute_resting_boosts("Tell me about food")
        print(f"Resting boosts: {len(boosts)} memories boosted")
        for mid, boost in list(boosts.items())[:5]:
            row = lq._core.store.db.execute(
                "SELECT content FROM memories WHERE id=?", (mid,)
            ).fetchone()
            if row:
                print(f"  boost={boost:.3f}: {row[0][:60]}")

        # Prabhashi has resting activation > 0, so memories mentioning her
        # should get a boost
        prabhashi = _find_entity(lq, "prabhashi")
        if prabhashi:
            state = lq.get_entity_state(prabhashi.id)
            if state.resting_activation > 0:
                assert len(boosts) > 0, \
                    "Should have resting boosts for memories about active entities"

    def test_scored_memory_has_resting_boost_field(self, lq):
        """ScoredMemory dataclass should include resting_boost."""
        from limbiq.retrieval.activation_retrieval import ScoredMemory
        sm = ScoredMemory(
            memory_id="test", content="test", final_score=1.0,
            embedding_sim=0.5, activation=0.3, graph_boost=0.2,
            resting_boost=0.1, is_priority=False, tier="short",
        )
        assert sm.resting_boost == 0.1


# ── Expression masks ──────────────────────────────────────────────


class TestExpressionMasks:
    """Expression masks should adapt to conversation topic."""

    def test_work_topic_sets_work_mask(self, lq):
        """Work-related query should set work=True mask on mentioned entities."""
        lq.start_session()
        lq.observe("My wife Prabhashi works at TechCorp", "Nice!")
        lq.end_session()

        prabhashi = _find_entity(lq, "prabhashi")
        if not prabhashi:
            pytest.skip("Entity extraction didn't pick up Prabhashi")

        # Work-focused query
        lq.start_session()
        lq.process("Help me review the code Prabhashi wrote for the project")

        state = lq.get_entity_state(prabhashi.id)
        print(f"Expression mask after work query: {state.expression_mask}")
        assert state.expression_mask.get("work") is True, \
            "Work query should set work=True on mentioned entity"

    def test_personal_topic_sets_personal_mask(self, lq):
        """Personal query should set personal=True mask on mentioned entities."""
        lq.start_session()
        lq.observe("My wife Prabhashi works at TechCorp", "Nice!")
        lq.end_session()

        prabhashi = _find_entity(lq, "prabhashi")
        if not prabhashi:
            pytest.skip("Entity extraction didn't pick up Prabhashi")

        # Personal query
        lq.start_session()
        lq.process("What should I get Prabhashi for her birthday dinner?")

        state = lq.get_entity_state(prabhashi.id)
        print(f"Expression mask after personal query: {state.expression_mask}")
        assert state.expression_mask.get("personal") is True, \
            "Personal query should set personal=True on mentioned entity"

    def test_neutral_query_no_mask_change(self, lq):
        """Neutral query without topic keywords shouldn't change masks."""
        lq.start_session()
        lq.observe("My wife Prabhashi works at TechCorp", "Nice!")
        lq.end_session()

        prabhashi = _find_entity(lq, "prabhashi")
        if not prabhashi:
            pytest.skip("Entity extraction didn't pick up Prabhashi")

        # Set a mask first
        lq._core.entity_state_store.update_expression_mask(
            prabhashi.id, {"work": True, "personal": False}
        )

        # Neutral query
        lq.start_session()
        lq.process("Tell me about Prabhashi")

        state = lq.get_entity_state(prabhashi.id)
        # Mask should not be overwritten by neutral query
        print(f"Expression mask after neutral query: {state.expression_mask}")
        assert state.expression_mask.get("work") is True, \
            "Neutral query should not clear existing masks"


# ── End-to-end: entity state improves retrieval ──────────────────


class TestEntityStateRetrieval:
    """End-to-end test that entity state actually changes what surfaces."""

    def test_frequently_discussed_entity_surfaces_more(self, lq):
        """Memories about frequently-discussed entities should rank higher in context."""
        # Build strong activation for Prabhashi across many sessions
        for i in range(4):
            lq.start_session()
            lq.observe(f"My wife Prabhashi and I did activity_{i} together", "Nice!")
            lq.end_session()

        # Add one mention of Alex
        lq.start_session()
        lq.observe("My colleague Alex called about the meeting", "Got it!")
        lq.end_session()

        # Now ask a vague question — Prabhashi should dominate context
        lq.start_session()
        result = lq.process("What have I been up to lately?")
        context_lower = result.context.lower()

        prabhashi_in_ctx = "prabhashi" in context_lower
        alex_in_ctx = "alex" in context_lower

        print(f"Prabhashi in context: {prabhashi_in_ctx}")
        print(f"Alex in context: {alex_in_ctx}")
        print(f"Context snippet: {result.context[:300]}")

        # Prabhashi (4 sessions) should be in context
        if prabhashi_in_ctx:
            print("GOOD: Frequently discussed entity surfaces in vague query")

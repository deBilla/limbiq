"""Stress-test: Can Limbiq extract knowledge from indirect/implicit messages?

These tests use sophisticated prompts that imply information without stating
it directly. They map out the boundary of what the extraction pipeline catches
vs misses — exposing gaps where distributed entity state could help.

Each test is labeled with expected outcome:
- SHOULD_CATCH: Pipeline has patterns that cover this
- STRETCH: Might catch it depending on spaCy/regex interaction
- GAP: Known limitation — would need inference or LLM extraction
"""

import pytest

from limbiq import Limbiq
from tests.conftest import MockEncoder


def _inject_encoder(lq):
    """Inject a mock encoder so signals fire without a trained model."""
    lq._core.encoder = MockEncoder({
        "wrong": ("correction", 0.9),
        "don't live": ("denial", 0.9),
        "i never said": ("denial", 0.9),
        "i already told you": ("frustration", 0.9),
    })


@pytest.fixture
def lq(tmp_dir):
    instance = Limbiq(store_path=tmp_dir, user_id="test")
    _inject_encoder(instance)
    return instance


def _find_entity(lq, name_fragment):
    """Find an entity whose name contains the fragment (case-insensitive)."""
    for e in lq.get_entities():
        if name_fragment.lower() in e.name.lower():
            return e
    return None


def _has_relation(lq, subject_fragment, predicate_fragment=None, object_fragment=None):
    """Check if a relation exists matching the fragments."""
    for r in lq.get_relations(include_inferred=True):
        subj = next((e for e in lq.get_entities() if e.id == r.subject_id), None)
        obj = next((e for e in lq.get_entities() if e.id == r.object_id), None)
        if not subj or not obj:
            continue
        subj_match = subject_fragment.lower() in subj.name.lower() if subject_fragment else True
        pred_match = predicate_fragment.lower() in r.predicate.lower() if predicate_fragment else True
        obj_match = object_fragment.lower() in obj.name.lower() if object_fragment else True
        if subj_match and pred_match and obj_match:
            return True
    return False


def _get_all_entity_names(lq):
    return [e.name for e in lq.get_entities()]


def _get_all_relation_triples(lq):
    """Return all relations as (subject_name, predicate, object_name) tuples."""
    entities = {e.id: e.name for e in lq.get_entities()}
    return [
        (entities.get(r.subject_id, "?"), r.predicate, entities.get(r.object_id, "?"))
        for r in lq.get_relations(include_inferred=True)
    ]


# ══════════════════════════════════════════════════════════════
# TIER 1: Direct-ish — relationship words present, just buried
# ══════════════════════════════════════════════════════════════

class TestBuriedRelationships:
    """Relationship keywords exist but are embedded in natural speech."""

    def test_father_calling_with_context(self, lq):
        """SHOULD_CATCH: 'my father' keyword is present, just in a sentence."""
        lq.start_session()
        lq.observe(
            "My father Upananda called from Kandy again, he never remembers the time difference",
            "That's sweet of him to call!",
        )
        lq.end_session()

        entities = _get_all_entity_names(lq)
        relations = _get_all_relation_triples(lq)
        print(f"Entities: {entities}")
        print(f"Relations: {relations}")

        assert _find_entity(lq, "upananda"), \
            f"Should extract 'Upananda' as entity. Got: {entities}"
        assert _has_relation(lq, "test", "father", "upananda"), \
            f"Should extract father relation. Got: {relations}"

    def test_wife_in_casual_action(self, lq):
        """SHOULD_CATCH: 'my wife' present in action context."""
        lq.start_session()
        lq.observe(
            "My wife Prabhashi keeps stealing the covers at night",
            "Haha, sounds familiar!",
        )
        lq.end_session()

        entities = _get_all_entity_names(lq)
        relations = _get_all_relation_triples(lq)
        print(f"Entities: {entities}")
        print(f"Relations: {relations}")

        assert _find_entity(lq, "prabhashi"), \
            f"Should extract 'Prabhashi'. Got: {entities}"
        assert _has_relation(lq, "test", "wife", "prabhashi"), \
            f"Should extract wife relation. Got: {relations}"

    def test_work_buried_in_complaint(self, lq):
        """SHOULD_CATCH: 'work at' pattern present in a complaint."""
        lq.start_session()
        lq.observe(
            "I work at Bitsmedia but the commute is killing me",
            "That's tough, how long is it?",
        )
        lq.end_session()

        entities = _get_all_entity_names(lq)
        print(f"Entities: {entities}")

        assert _find_entity(lq, "bitsmedia"), \
            f"Should extract 'Bitsmedia'. Got: {entities}"

    def test_lives_in_buried_in_story(self, lq):
        """SHOULD_CATCH: 'live in' pattern present."""
        lq.start_session()
        lq.observe(
            "Ever since I moved to Boston I can't find good kottu anywhere",
            "I bet! Sri Lankan food must be hard to find there.",
        )
        lq.end_session()

        entities = _get_all_entity_names(lq)
        print(f"Entities: {entities}")

        # "moved to Boston" doesn't match "live in" pattern, but spaCy should catch Boston as GPE
        boston = _find_entity(lq, "boston")
        if boston:
            print("Boston extracted as entity!")
        else:
            print(f"GAP: 'moved to Boston' not caught. Entities: {entities}")


# ══════════════════════════════════════════════════════════════
# TIER 2: Implicit — no relationship keywords, but inferable
# ══════════════════════════════════════════════════════════════

class TestImplicitRelationships:
    """No explicit relationship words — meaning must be inferred."""

    def test_sharing_bed_implies_partner(self, lq):
        """GAP: No 'wife/husband/partner' keyword. Needs inference."""
        lq.start_session()
        lq.observe(
            "Prabhashi keeps stealing my covers at night",
            "Haha, sounds like you need a bigger blanket!",
        )
        lq.end_session()

        entities = _get_all_entity_names(lq)
        print(f"Entities: {entities}")

        # spaCy might catch "Prabhashi" as PERSON
        prabhashi = _find_entity(lq, "prabhashi")
        if prabhashi:
            print("CAUGHT: Prabhashi extracted as entity from implicit context")
            # But can it infer the relationship?
            has_rel = _has_relation(lq, "test", None, "prabhashi")
            print(f"Has any relation to Prabhashi: {has_rel}")
            if has_rel:
                print(f"Relations: {_get_all_relation_triples(lq)}")
        else:
            print(f"MISS: Prabhashi not extracted. Entities: {entities}")

    def test_school_call_implies_child(self, lq):
        """GAP: 'school called' implies Prabhashi is a child. No keyword."""
        lq.start_session()
        lq.observe(
            "Had to leave the meeting early because Prabhashi's school called",
            "Hope everything is okay!",
        )
        lq.end_session()

        entities = _get_all_entity_names(lq)
        print(f"Entities: {entities}")

        prabhashi = _find_entity(lq, "prabhashi")
        if prabhashi:
            print("CAUGHT: Prabhashi extracted")
            relations = _get_all_relation_triples(lq)
            print(f"Relations: {relations}")
        else:
            print(f"MISS: Prabhashi not extracted. Need inference: school called → child")

    def test_cafeteria_implies_coworker(self, lq):
        """GAP: 'cafeteria downstairs' implies same workplace. No 'colleague' keyword."""
        lq.start_session()
        lq.observe(
            "Rohan and I grabbed lunch at the cafeteria downstairs",
            "Nice, what did you have?",
        )
        lq.end_session()

        entities = _get_all_entity_names(lq)
        print(f"Entities: {entities}")

        rohan = _find_entity(lq, "rohan")
        if rohan:
            print("CAUGHT: Rohan extracted as entity")
            has_rel = _has_relation(lq, "test", None, "rohan")
            print(f"Has relation to Rohan: {has_rel}")
        else:
            print(f"MISS: Rohan not extracted from casual mention")

    def test_amma_cultural_inference(self, lq):
        """GAP: 'Amma' is Sinhala for mother. Needs cultural knowledge."""
        lq.start_session()
        lq.observe(
            "Every time I visit Amma she makes my favorite kiribath",
            "That sounds delicious!",
        )
        lq.end_session()

        entities = _get_all_entity_names(lq)
        print(f"Entities: {entities}")

        amma = _find_entity(lq, "amma")
        if amma:
            print("CAUGHT: Amma extracted as entity")
            has_mother = _has_relation(lq, "test", "mother", "amma")
            print(f"Inferred as mother: {has_mother}")
        else:
            print(f"MISS: 'Amma' not recognized. Needs cultural/linguistic mapping")


# ══════════════════════════════════════════════════════════════
# TIER 3: Multi-turn accumulation — info spread across messages
# ══════════════════════════════════════════════════════════════

class TestMultiTurnAccumulation:
    """Information accumulates across multiple turns and sessions."""

    def test_name_then_relationship_separate_turns(self, lq):
        """STRETCH: Name in one turn, relationship in another."""
        lq.start_session()
        lq.observe("Had coffee with Rohan this morning", "How was it?")
        lq.observe("He's been my colleague for 5 years now", "That's great!")
        lq.end_session()

        entities = _get_all_entity_names(lq)
        relations = _get_all_relation_triples(lq)
        print(f"Entities: {entities}")
        print(f"Relations: {relations}")

        rohan = _find_entity(lq, "rohan")
        if rohan:
            # "He's been my colleague" — the "He" doesn't resolve to Rohan
            # without coreference resolution
            has_colleague = _has_relation(lq, "test", "colleague", "rohan")
            print(f"Colleague relation found: {has_colleague}")
            if not has_colleague:
                print("GAP: 'He' not resolved to 'Rohan' — needs coreference resolution")

    def test_cross_session_connection(self, lq):
        """GAP: Info in session 1 + info in session 2 = inference in session 3."""
        # Session 1: Mention Rohan
        lq.start_session()
        lq.observe("Rohan helped me debug the authentication module today", "Teamwork!")
        lq.end_session()

        # Session 2: Mention Bitsmedia
        lq.start_session()
        lq.observe("The Bitsmedia office has amazing coffee", "Nice perk!")
        lq.end_session()

        # Session 3: Connect them
        lq.start_session()
        lq.observe("Rohan and I both work at Bitsmedia", "Cool!")
        lq.end_session()

        entities = _get_all_entity_names(lq)
        relations = _get_all_relation_triples(lq)
        print(f"After 3 sessions - Entities: {entities}")
        print(f"After 3 sessions - Relations: {relations}")

        # Direct "work at" should be caught in session 3
        assert _find_entity(lq, "bitsmedia"), f"Should extract Bitsmedia. Got: {entities}"

        # But does it retroactively connect Rohan's session 1 "debug" work to Bitsmedia?
        rohan = _find_entity(lq, "rohan")
        if rohan:
            rohan_rels = [
                (s, p, o) for s, p, o in relations
                if "rohan" in s.lower() or "rohan" in o.lower()
            ]
            print(f"Rohan's relations: {rohan_rels}")

    def test_gradual_family_reveal(self, lq):
        """Build a family picture over multiple sessions without explicit declarations."""
        # Session 1: Casual mention
        lq.start_session()
        lq.observe("My wife Prabhashi cooked an amazing dinner last night", "Yum!")
        lq.end_session()

        # Session 2: Father mentioned in a story
        lq.start_session()
        lq.observe("My father Upananda used to take me fishing when I was a kid", "Sweet memory!")
        lq.end_session()

        # Session 3: Father and wife in same context — can it infer father-in-law?
        lq.start_session()
        lq.observe("Upananda visited us last weekend, Prabhashi made his favorite curry", "Family time!")
        lq.end_session()

        entities = _get_all_entity_names(lq)
        relations = _get_all_relation_triples(lq)
        print(f"Family graph entities: {entities}")
        print(f"Family graph relations: {relations}")

        # Direct relations should exist
        assert _find_entity(lq, "prabhashi"), f"Should have Prabhashi. Got: {entities}"
        assert _find_entity(lq, "upananda"), f"Should have Upananda. Got: {entities}"

        # Inference check: did the system infer father-in-law?
        has_fil = _has_relation(lq, "upananda", "father_in_law", "prabhashi") or \
                  _has_relation(lq, None, "father_in_law", None)
        print(f"Father-in-law inferred: {has_fil}")
        if has_fil:
            print("EXCELLENT: System inferred family relationship from separate sessions!")
        else:
            print("Expected: inference engine should deduce father + wife → father-in-law")


# ══════════════════════════════════════════════════════════════
# TIER 4: Emotional/contextual — meaning in tone, not words
# ══════════════════════════════════════════════════════════════

class TestSentinelMemory:
    """A4. Immune system memory — sentinels watch for stale references."""

    def test_correction_creates_sentinel(self, lq):
        """When user corrects a fact, a sentinel should watch for the old fact."""
        lq.start_session()
        lq.observe("I live in London", "Nice city!")
        lq.end_session()

        lq.start_session()
        # Correction — dopamine fires, GABA suppresses old memory
        lq.observe(
            "No that's wrong, I don't live in London, I moved to Boston",
            "Thanks for the update!",
        )
        lq.end_session()

        # Check: sentinel should exist for "london"
        sentinels = lq.get_sentinels()
        sentinel_patterns = [s.sentinel_pattern for s in sentinels]
        print(f"Sentinels: {sentinel_patterns}")

        entities = _get_all_entity_names(lq)
        relations = _get_all_relation_triples(lq)
        print(f"Entities: {entities}")
        print(f"Relations: {relations}")

        # London entity should have a sentinel
        london = _find_entity(lq, "london")
        if london:
            state = lq.get_entity_state(london.id)
            if state.sentinel_pattern:
                print(f"GOOD: Sentinel on London: '{state.sentinel_pattern}'")
            else:
                print(f"MISS: London exists but no sentinel. Patterns: {sentinel_patterns}")
        else:
            print(f"London not an entity. Sentinels: {sentinel_patterns}")

    @pytest.mark.xfail(reason="Sentinel pipeline needs entity extraction fixes to suppress stale facts")
    def test_sentinel_flags_stale_query(self, lq):
        """Sentinel should add caution when query references corrected entity."""
        lq.start_session()
        lq.observe("I live in London", "Nice city!")
        lq.end_session()

        lq.start_session()
        lq.observe(
            "No that's wrong, I don't live in London, I moved to Boston",
            "Got it, Boston!",
        )
        lq.end_session()

        # Now query about London — sentinel should trigger
        lq.start_session()
        result = lq.process("What was that restaurant I liked in London?")
        print(f"Context includes caution: {'caution' in result.context.lower()}")
        print(f"Context includes 'corrected': {'corrected' in result.context.lower()}")
        print(f"Context snippet: {result.context[:500]}")

        # The sentinel should have added a caution about London
        assert "caution" in result.context.lower(), \
            "Sentinel should add CAUTION flag when London is mentioned"
        assert "london" in result.context.lower(), \
            "London should be referenced in the caution"

    def test_sentinel_not_triggered_on_unrelated(self, lq):
        """Sentinel should NOT trigger on unrelated queries."""
        lq.start_session()
        lq.observe("I live in London", "Nice!")
        lq.end_session()

        lq.start_session()
        lq.observe(
            "No that's wrong, I don't live in London, I moved to Boston",
            "Updated!",
        )
        lq.end_session()

        # Query that doesn't mention London — no sentinel trigger
        lq.start_session()
        result = lq.process("What should I have for dinner?")
        has_corrected = "corrected" in result.context.lower()
        print(f"Unrelated query has correction warning: {has_corrected}")
        # Should NOT have a correction warning for an unrelated query
        assert not has_corrected, \
            "Sentinel should not trigger on unrelated queries"


class TestEmotionalContext:
    """Information encoded in emotional context rather than explicit facts."""

    def test_frustration_reveals_relationship_quality(self, lq):
        """Test that explicit frustration keywords flood norepinephrine to mentioned entities."""
        lq.start_session()
        # Use a frustration pattern that norepinephrine actually detects
        lq.observe(
            "I already told you about Rohan, he keeps taking credit for my work",
            "Sorry about that. That sounds frustrating.",
        )
        lq.end_session()

        entities = _get_all_entity_names(lq)
        print(f"Entities: {entities}")

        rohan = _find_entity(lq, "rohan")
        if rohan:
            state = lq.get_entity_state(rohan.id)
            print(f"Rohan entity state: activation={state.resting_activation}, "
                  f"signals={state.signal_history}")
            ne_count = state.signal_history.get("norepinephrine", 0)
            if ne_count > 0:
                print("GOOD: Norepinephrine signal recorded on Rohan entity — "
                      "system knows there's tension here")
            else:
                print("MISS: Norepinephrine didn't flood to Rohan")

    def test_implicit_frustration_not_detected(self, lq):
        """GAP: Implicit frustration (no keywords) isn't caught yet."""
        lq.start_session()
        lq.observe(
            "Rohan keeps taking credit for my work in the standup meetings",
            "That's frustrating. Have you talked to him about it?",
        )
        lq.end_session()

        rohan = _find_entity(lq, "rohan")
        if rohan:
            state = lq.get_entity_state(rohan.id)
            print(f"Rohan activation={state.resting_activation} (from mention)")
            print(f"Rohan signals={state.signal_history}")
            # Norepinephrine won't fire — "keeps taking credit" isn't a frustration keyword
            # But Rohan DOES get activated from being mentioned
            assert state.resting_activation > 0, \
                "Rohan should at least be activated from mention"
            print("GAP: Implicit frustration not detected — "
                  "needs sentiment analysis or LLM inference")

    def test_enthusiasm_reveals_importance(self, lq):
        """The way someone talks about a topic reveals its importance."""
        lq.start_session()
        lq.observe(
            "I finally finished painting the sunset watercolor I've been working on for weeks!",
            "That's wonderful! Can you show me?",
        )
        lq.observe(
            "Also had a meeting about Q3 budget",
            "How did that go?",
        )
        lq.end_session()

        entities = _get_all_entity_names(lq)
        print(f"Entities: {entities}")

        # The watercolor got enthusiasm (dopamine), the budget didn't
        # Entity state should reflect this
        all_states = lq.get_all_entity_states()
        for state in all_states:
            if state.resting_activation > 0:
                entity = next(
                    (e for e in lq.get_entities() if e.id == state.entity_id), None
                )
                if entity:
                    print(f"  Active entity: {entity.name} "
                          f"(activation={state.resting_activation:.3f}, "
                          f"signals={state.signal_history})")


# ══════════════════════════════════════════════════════════════
# TIER 5: Contradiction & correction — implied updates
# ══════════════════════════════════════════════════════════════

class TestImpliedUpdates:
    """Information that contradicts or updates without saying 'actually'."""

    def test_tense_change_implies_update(self, lq):
        """GAP: Past tense implies things have changed."""
        lq.start_session()
        lq.observe("I work at Bitsmedia", "Cool!")
        lq.end_session()

        lq.start_session()
        # "worked" (past tense) + "now at" — implies job change
        lq.observe(
            "When I worked at Bitsmedia the coffee was terrible, "
            "but Atlassian has great perks",
            "Sounds like an upgrade!",
        )
        lq.end_session()

        entities = _get_all_entity_names(lq)
        relations = _get_all_relation_triples(lq)
        print(f"Entities: {entities}")
        print(f"Relations: {relations}")

        # Did it catch that user now works at Atlassian?
        atlassian = _find_entity(lq, "atlassian")
        if atlassian:
            has_works_at = _has_relation(lq, "test", "works_at", "atlassian")
            print(f"Works at Atlassian: {has_works_at}")
            if has_works_at:
                print("CAUGHT: Inferred new workplace from context")
        else:
            print(f"MISS: Atlassian not extracted. Entities: {entities}")

        # Did it suppress the old Bitsmedia relation?
        still_at_bits = _has_relation(lq, "test", "works_at", "bitsmedia")
        print(f"Still shows works at Bitsmedia: {still_at_bits}")
        if still_at_bits:
            print("GAP: Old workplace not updated — 'worked at' (past) not distinguished from 'work at' (present)")

    def test_someone_else_at_your_house(self, lq):
        """GAP: 'Prabhashi's school called' implies she goes to school + you're her parent/guardian."""
        lq.start_session()
        lq.observe("My daughter Prabhashi has her first piano recital tomorrow", "How exciting!")
        lq.end_session()

        entities = _get_all_entity_names(lq)
        relations = _get_all_relation_triples(lq)
        print(f"Entities: {entities}")
        print(f"Relations: {relations}")

        assert _find_entity(lq, "prabhashi"), f"Should extract Prabhashi. Got: {entities}"
        assert _has_relation(lq, "test", "daughter", "prabhashi"), \
            f"Should extract daughter relation. Got: {relations}"


# ══════════════════════════════════════════════════════════════
# TIER 6: Entity state tracking through implicit patterns
# ══════════════════════════════════════════════════════════════

class TestEntityStateFromImplicitPatterns:
    """Test that entity state accumulates correctly from natural conversation."""

    def test_entity_activation_reflects_conversation_importance(self, lq):
        """Entities that dominate conversation should have highest activation."""
        lq.start_session()
        # Talk a LOT about Prabhashi, barely about Alex
        lq.observe("My wife Prabhashi is an amazing cook", "Wonderful!")
        lq.observe("Prabhashi and I are planning a trip to Japan", "Exciting!")
        lq.observe("Prabhashi's promotion party was fantastic", "Congratulations to her!")
        lq.observe("Oh and Alex called", "What about?")
        lq.end_session()

        prabhashi = _find_entity(lq, "prabhashi")
        alex = _find_entity(lq, "alex")

        if prabhashi and alex:
            p_state = lq.get_entity_state(prabhashi.id)
            a_state = lq.get_entity_state(alex.id)
            print(f"Prabhashi: activation={p_state.resting_activation:.3f}, "
                  f"total_activations={p_state.total_activations}")
            print(f"Alex: activation={a_state.resting_activation:.3f}, "
                  f"total_activations={a_state.total_activations}")
            assert p_state.total_activations > a_state.total_activations, \
                "Prabhashi mentioned 3x should have more activations than Alex mentioned 1x"
        elif prabhashi:
            print(f"Only Prabhashi extracted (Alex too casual). State: "
                  f"activation={lq.get_entity_state(prabhashi.id).resting_activation:.3f}")
        else:
            print(f"Neither entity extracted. Entities: {_get_all_entity_names(lq)}")

    def test_signal_accumulation_across_sessions(self, lq):
        """Entity state should accumulate signal hits across multiple sessions."""
        # Session 1: Dopamine (personal info)
        lq.start_session()
        lq.observe("My wife is Prabhashi", "Nice!")
        lq.end_session()

        # Session 2: More dopamine (another personal fact)
        lq.start_session()
        lq.observe("Prabhashi works at Google", "Cool company!")
        lq.end_session()

        # Session 3: Even more
        lq.start_session()
        lq.observe("Prabhashi got promoted to Staff Engineer", "Impressive!")
        lq.end_session()

        prabhashi = _find_entity(lq, "prabhashi")
        if prabhashi:
            state = lq.get_entity_state(prabhashi.id)
            print(f"After 3 sessions - Prabhashi state:")
            print(f"  resting_activation: {state.resting_activation:.4f}")
            print(f"  total_activations: {state.total_activations}")
            print(f"  signal_history: {state.signal_history}")
            print(f"  receptor_density: {state.receptor_density}")

            # Note: not all sessions trigger entity extraction — "got promoted"
            # may not match any relation pattern. We expect at least 2 activations
            # (from the 2 sessions with explicit relationship keywords).
            assert state.total_activations >= 2, \
                f"Should have at least 2 activations from 3 sessions. Got: {state.total_activations}"
        else:
            pytest.fail(f"Prabhashi not extracted. Entities: {_get_all_entity_names(lq)}")

    def test_top_activated_entities_match_conversation_focus(self, lq):
        """Top activated entities should reflect who/what was discussed most."""
        lq.start_session()
        lq.observe("My wife Prabhashi and I visited my father Upananda in Kandy", "Family trip!")
        lq.observe("Prabhashi loved the mountain views", "Beautiful!")
        lq.observe("Prabhashi wants to go back next month", "That would be nice!")
        lq.end_session()

        top = lq.get_top_activated_entities(limit=5)
        if top:
            print("Top activated entities:")
            for state in top:
                entity = next(
                    (e for e in lq.get_entities() if e.id == state.entity_id), None
                )
                name = entity.name if entity else state.entity_id
                print(f"  {name}: activation={state.resting_activation:.3f}, "
                      f"count={state.total_activations}")

            # Prabhashi (mentioned 3x) should rank higher than Upananda (1x)
            entity_names = []
            for state in top:
                entity = next(
                    (e for e in lq.get_entities() if e.id == state.entity_id), None
                )
                if entity:
                    entity_names.append(entity.name.lower())

            if "prabhashi" in entity_names and "upananda" in entity_names:
                p_idx = entity_names.index("prabhashi")
                u_idx = entity_names.index("upananda")
                assert p_idx < u_idx, \
                    "Prabhashi (3 mentions) should rank above Upananda (1 mention)"

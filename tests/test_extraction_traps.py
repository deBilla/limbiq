"""Stress-test: Prompts designed to trick the extraction pipeline into
creating WRONG relations between entities.

These simulate real conversation patterns that caused the
Renuka/Dilini/Yuenshe misattribution bug — multiple people mentioned
together, ambiguous possessives, pronouns that could refer to anyone.

Each test feeds prompts through the full observe() pipeline and then
checks the graph for incorrect relations that should NOT exist.
"""

import pytest

from limbiq import Limbiq


@pytest.fixture
def lq(tmp_dir):
    return Limbiq(store_path=tmp_dir, user_id="test")


def _entities(lq):
    return {e.name.lower(): e for e in lq.get_entities()}


def _relations(lq):
    """Return all relations as (subject_name, predicate, object_name) tuples."""
    eid_to_name = {e.id: e.name for e in lq.get_entities()}
    return [
        (eid_to_name.get(r.subject_id, "?").lower(), r.predicate,
         eid_to_name.get(r.object_id, "?").lower(), r.is_inferred)
        for r in lq.get_relations(include_inferred=True)
    ]


def _has_relation(rels, subj=None, pred=None, obj=None):
    for s, p, o, inf in rels:
        if (subj is None or subj in s) and \
           (pred is None or pred in p) and \
           (obj is None or obj in o):
            return True
    return False


def _print_graph(lq, label=""):
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"{'='*60}")
    rels = _relations(lq)
    for s, p, o, inf in rels:
        tag = " [inferred]" if inf else ""
        print(f"  {s} --{p}--> {o}{tag}")
    if not rels:
        print("  (no relations)")
    print()


# ══════════════════════════════════════════════════════════════
# TRAP 1: Multiple people in one sentence — who belongs to whom?
# ══════════════════════════════════════════════════════════════

class TestMultiPersonAmbiguity:
    """Sentences with 2-3 people where the system might mix up relations."""

    def test_friend_mentioned_with_family(self, lq):
        """TRAP: 'My sister Dilini and my friend Yuenshe went shopping'
        Should NOT create Dilini --sister--> Yuenshe."""
        lq.start_session()
        lq.observe(
            "My sister Dilini and my friend Yuenshe went shopping together",
            "Sounds fun!",
        )
        lq.end_session()

        _print_graph(lq, "sister + friend in same sentence")
        rels = _relations(lq)

        # Correct: test --sister--> dilini, test --friend--> yuenshe
        # WRONG: dilini --sister--> yuenshe (the actual bug we saw)
        assert not _has_relation(rels, "dilini", "sister", "yuenshe"), \
            "WRONG: Should NOT create Dilini--sister-->Yuenshe"
        assert not _has_relation(rels, "yuenshe", "sister", "dilini"), \
            "WRONG: Should NOT create Yuenshe--sister-->Dilini"

    def test_two_family_members_same_sentence(self, lq):
        """TRAP: 'My mother Renuka and my sister Dilini came to visit'
        Should NOT create Renuka --sister--> Dilini or Renuka --mother--> Dilini."""
        lq.start_session()
        lq.observe(
            "My mother Renuka and my sister Dilini came to visit me last weekend",
            "That's lovely!",
        )
        lq.end_session()

        _print_graph(lq, "mother + sister in same sentence")
        rels = _relations(lq)

        # Correct relations to user
        assert _has_relation(rels, "test", "mother", "renuka"), \
            f"Should have test--mother-->Renuka. Got: {rels}"
        assert _has_relation(rels, "test", "sister", "dilini"), \
            f"Should have test--sister-->Dilini. Got: {rels}"

        # WRONG cross-relations
        assert not _has_relation(rels, "renuka", "sister", "dilini"), \
            "WRONG: Renuka is NOT Dilini's sister"
        assert not _has_relation(rels, "renuka", "mother", "dilini"), \
            "WRONG: Should not create Renuka--mother-->Dilini (she's MY mother)"
        assert not _has_relation(rels, "dilini", "sister", "renuka"), \
            "WRONG: Dilini is NOT Renuka's sister"

    def test_wife_and_friend_dinner(self, lq):
        """TRAP: 'Prabhashi and Rohan came for dinner'
        Should NOT create Prabhashi --wife--> Rohan or any relation between them."""
        lq.start_session()
        lq.observe("My wife Prabhashi is amazing", "She sounds great!")
        lq.observe("My friend Rohan is visiting from Sydney", "Cool!")
        lq.observe(
            "Prabhashi and Rohan really hit it off at dinner last night",
            "Great to hear!",
        )
        lq.end_session()

        _print_graph(lq, "wife + friend at dinner")
        rels = _relations(lq)

        # Should NOT create any relation between Prabhashi and Rohan
        assert not _has_relation(rels, "prabhashi", "wife", "rohan"), \
            "WRONG: Prabhashi is NOT Rohan's wife"
        assert not _has_relation(rels, "rohan", "husband", "prabhashi"), \
            "WRONG: Rohan is NOT Prabhashi's husband"
        assert not _has_relation(rels, "prabhashi", "friend", "rohan"), \
            "WRONG: Should not infer friendship between them"

    def test_three_people_activity(self, lq):
        """TRAP: 'My brother Amal, my wife Prabhashi, and my colleague Rohan went hiking'
        Should create 3 relations to user. Inferred family connections (brother-in-law)
        are acceptable, but WRONG explicit cross-relations are not."""
        lq.start_session()
        lq.observe(
            "My brother Amal, my wife Prabhashi, and my colleague Rohan went hiking together",
            "What a group!",
        )
        lq.end_session()

        _print_graph(lq, "three people in one activity")
        rels = _relations(lq)

        # Core relations to user should exist
        assert _has_relation(rels, "test", "brother", "amal"), \
            f"Should have test--brother-->Amal. Got: {rels}"
        assert _has_relation(rels, "test", "wife", "prabhashi"), \
            f"Should have test--wife-->Prabhashi. Got: {rels}"

        # Inferred: amal --brother_in_law_of--> prabhashi is CORRECT (brother + wife → BIL)
        # But no WRONG explicit relations between the three
        for a in ["amal", "prabhashi", "rohan"]:
            for b in ["amal", "prabhashi", "rohan"]:
                if a == b:
                    continue
                for pred in ["brother", "wife", "colleague", "sister", "husband"]:
                    wrong = [(s, p, o, inf) for s, p, o, inf in rels
                             if a in s and pred in p and b in o and not inf]
                    assert not wrong, \
                        f"WRONG explicit: {a} --{pred}--> {b} should not exist"


# ══════════════════════════════════════════════════════════════
# TRAP 2: Possessive ambiguity — whose X is it?
# ══════════════════════════════════════════════════════════════

class TestPossessiveAmbiguity:
    """Possessives that could be misattributed."""

    def test_friends_pet_not_mine(self, lq):
        """TRAP: 'Yuenshe's dog Murphy is adorable'
        Should create Yuenshe --pet--> Murphy, NOT test --pet--> Murphy."""
        lq.start_session()
        lq.observe(
            "My friend Yuenshe has a dog called Murphy, he's so adorable",
            "Cute!",
        )
        lq.end_session()

        _print_graph(lq, "friend's pet")
        rels = _relations(lq)

        # Murphy belongs to Yuenshe, not to the user
        if _has_relation(rels, "test", "pet", "murphy"):
            print("WRONG: Murphy assigned as user's pet instead of Yuenshe's")
        if _has_relation(rels, "yuenshe", "pet", "murphy"):
            print("CORRECT: Murphy is Yuenshe's pet")

    def test_wifes_father_not_my_father(self, lq):
        """TRAP: 'Prabhashi's father Chandrasiri called'
        Should create Prabhashi --father--> Chandrasiri, NOT test --father--> Chandrasiri."""
        lq.start_session()
        lq.observe("My wife is Prabhashi", "Nice!")
        lq.observe(
            "Prabhashi's father Chandrasiri called us this morning",
            "What did he want?",
        )
        lq.end_session()

        _print_graph(lq, "wife's father")
        rels = _relations(lq)

        # Chandrasiri is Prabhashi's father, should NOT be user's father
        assert not _has_relation(rels, "test", "father", "chandrasiri"), \
            "WRONG: Chandrasiri is Prabhashi's father, not user's"

    def test_sisters_husband_not_mine(self, lq):
        """TRAP: 'My sister Dilini's husband Kamal works at Google'
        Should NOT create test --husband--> Kamal."""
        lq.start_session()
        lq.observe(
            "My sister Dilini's husband Kamal works at Google",
            "Cool company!",
        )
        lq.end_session()

        _print_graph(lq, "sister's husband")
        rels = _relations(lq)

        assert not _has_relation(rels, "test", "husband", "kamal"), \
            "WRONG: Kamal is Dilini's husband, not user's"


# ══════════════════════════════════════════════════════════════
# TRAP 3: Sequential mentions that could merge relationships
# ══════════════════════════════════════════════════════════════

class TestSequentialConfusion:
    """Separate turns about different people that could bleed into each other."""

    def test_separate_turn_relationship_bleeding(self, lq):
        """TRAP: Turn 1 about wife, Turn 2 about colleague — don't mix them."""
        lq.start_session()
        lq.observe("My wife Prabhashi cooked dinner", "Yum!")
        lq.observe("My colleague Rohan presented at the standup", "How did it go?")
        lq.end_session()

        _print_graph(lq, "wife turn then colleague turn")
        rels = _relations(lq)

        assert not _has_relation(rels, "prabhashi", None, "rohan"), \
            f"WRONG: No relation should exist between Prabhashi and Rohan"
        assert not _has_relation(rels, "rohan", None, "prabhashi"), \
            f"WRONG: No relation should exist between Rohan and Prabhashi"

    def test_cross_session_entity_type_stability(self, lq):
        """TRAP: Mention someone as friend in session 1, then in different context in session 2.
        Entity type should not flip."""
        lq.start_session()
        lq.observe("My friend Yuenshe is visiting from China", "Nice!")
        lq.end_session()

        lq.start_session()
        lq.observe("Yuenshe and my sister Dilini went to the mall", "Fun!")
        lq.end_session()

        _print_graph(lq, "friend + sister in separate sessions")
        rels = _relations(lq)

        # Yuenshe should remain a friend, NOT become a sister
        assert not _has_relation(rels, "test", "sister", "yuenshe"), \
            "WRONG: Yuenshe is a friend, not a sister"
        assert not _has_relation(rels, "dilini", "sister", "yuenshe"), \
            "WRONG: Dilini and Yuenshe are NOT sisters"
        assert not _has_relation(rels, "yuenshe", "sister", "dilini"), \
            "WRONG: Yuenshe and Dilini are NOT sisters"


# ══════════════════════════════════════════════════════════════
# TRAP 4: LLM response content bleeding into graph
# ══════════════════════════════════════════════════════════════

class TestResponseBleeding:
    """LLM response content should NOT create graph relations."""

    def test_response_relationship_not_extracted(self, lq):
        """TRAP: LLM says 'your father-in-law' — don't extract as new relation."""
        lq.start_session()
        lq.observe("My wife is Prabhashi", "Nice to know about your wife Prabhashi!")
        lq.observe(
            "Prabhashi's father is Chandrasiri",
            "So Chandrasiri is your father-in-law! That's wonderful.",
        )
        lq.end_session()

        _print_graph(lq, "response mentions father-in-law")
        rels = _relations(lq)

        # The LLM response says "father-in-law" but the extraction should
        # come from inference, not from parsing the response text
        # Key check: no duplicate or conflicting relations
        fil_rels = [(s, p, o) for s, p, o, _ in rels if "father" in p]
        print(f"Father-related relations: {fil_rels}")

    def test_response_suggestion_not_stored(self, lq):
        """TRAP: LLM suggests 'you should visit your sister' — don't create sister entity."""
        lq.start_session()
        lq.observe(
            "I'm feeling lonely this weekend",
            "You should visit your sister or call a friend! "
            "Maybe your colleague Sarah could join you for coffee.",
        )
        lq.end_session()

        _print_graph(lq, "response suggests sister/Sarah")
        ents = _entities(lq)

        # "Sarah" from the LLM response should NOT become an entity
        assert "sarah" not in ents, \
            "WRONG: Sarah from LLM response should not be extracted as entity"


# ══════════════════════════════════════════════════════════════
# TRAP 5: Ambiguous "is" statements
# ══════════════════════════════════════════════════════════════

class TestAmbiguousStatements:
    """Statements where 'is' could mean different things."""

    def test_is_not_always_relationship(self, lq):
        """TRAP: 'Prabhashi is amazing at cooking' — 'is' doesn't mean relation."""
        lq.start_session()
        lq.observe("My wife Prabhashi is amazing at cooking Thai food", "Sounds delicious!")
        lq.end_session()

        _print_graph(lq, "'is amazing' != relation")
        rels = _relations(lq)

        # Should have wife relation, but NOT "is_a amazing" or similar junk
        assert _has_relation(rels, "test", "wife", "prabhashi"), \
            "Should have wife relation"
        # Check for junk
        for s, p, o, _ in rels:
            if "prabhashi" in s and p == "is_a":
                print(f"JUNK: {s} --{p}--> {o}")

    def test_location_is_not_person(self, lq):
        """TRAP: 'Kandy is beautiful' — Kandy is a place, not a person."""
        lq.start_session()
        lq.observe(
            "My father Upananda lives in Kandy, it's such a beautiful city",
            "I've heard Kandy is gorgeous!",
        )
        lq.end_session()

        _print_graph(lq, "place mentioned")
        ents = _entities(lq)

        # Kandy should be a place, not a person
        if "kandy" in ents:
            assert ents["kandy"].entity_type == "place", \
                f"Kandy should be type 'place', got '{ents['kandy'].entity_type}'"

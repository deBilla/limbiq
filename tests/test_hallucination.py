#!/usr/bin/env python3
"""
Test the hallucination detection system against known hallucination-prone queries.

Uses Dimuthu's actual knowledge graph (from data/limbiq/) to verify
that the grounding analyzer and fact verifier work correctly.
"""

import sys
sys.path.insert(0, "/sessions/brave-friendly-mayer/mnt/limbiq")
sys.path.insert(0, "/sessions/brave-friendly-mayer/mnt/living-llm")

from limbiq import Limbiq, GroundingLevel, ClaimStatus


def setup():
    """Load limbiq with existing data."""
    lq = Limbiq(
        store_path="/sessions/brave-friendly-mayer/test_limbiq_data",
        user_id="default",
        embedding_model="all-MiniLM-L6-v2",
    )
    detector = lq.get_hallucination_detector()
    user_name = lq._core._graph_user_name
    return lq, detector, user_name


def test_grounding():
    """Test the pre-generation grounding analyzer."""
    lq, detector, user_name = setup()

    print("=" * 70)
    print("GROUNDING ANALYZER TESTS")
    print("=" * 70)

    test_cases = [
        # (query, expected_type, expected_level, description)
        ("who is my father", "personal", GroundingLevel.GROUNDED,
         "Should be GROUNDED — father is in graph"),

        ("who is my wife", "personal", GroundingLevel.GROUNDED,
         "Should be GROUNDED — wife is in graph"),

        ("where do i work", "personal", GroundingLevel.GROUNDED,
         "Should be GROUNDED — works_at is in graph"),

        ("what is my favorite color", "personal", GroundingLevel.UNGROUNDED,
         "Should be UNGROUNDED — no color info in graph"),

        ("what is my cat's name", "personal", GroundingLevel.PARTIAL,
         "Should be PARTIAL — has user context but no pet info"),

        ("how old am i", "personal", GroundingLevel.UNGROUNDED,
         "Should be UNGROUNDED — no age info"),

        ("what is machine learning", "general", None,
         "Should be GENERAL type — not personal"),

        ("hello", "meta", None,
         "Should be META type — greeting"),

        ("who is my wife's father", "personal", None,
         "Should be PERSONAL — possessive query"),

        ("tell me about Prabhashi", "personal", None,
         "Should detect known entity"),
    ]

    passed = 0
    for query, expected_type, expected_level, desc in test_cases:
        # Run process to get context
        result = lq.process(query)
        graph_result = lq.query_graph(query)
        world = lq.get_world_summary()

        grounding = detector.pre_generate(
            query=query,
            graph_result=graph_result if graph_result.get("answered") else None,
            memories_retrieved=result.memories_retrieved,
            memory_context=result.context,
            world_summary=world,
        )

        type_ok = grounding.query_type == expected_type
        level_ok = expected_level is None or grounding.level == expected_level
        ok = type_ok and level_ok

        status = "✓" if ok else "✗"
        print(f"\n  {status} {desc}")
        print(f"    Query: \"{query}\"")
        print(f"    Type: {grounding.query_type} (expected: {expected_type}) {'✓' if type_ok else '✗'}")
        if expected_level:
            print(f"    Level: {grounding.level.value} (expected: {expected_level.value}) {'✓' if level_ok else '✗'}")
        else:
            print(f"    Level: {grounding.level.value}")
        print(f"    Entities: {grounding.known_entities_mentioned}")
        print(f"    Relevance: {grounding.memory_relevance_score:.2f}")
        if grounding.constraint_prompt:
            print(f"    Constraint: {grounding.constraint_prompt[:80]}...")

        if ok:
            passed += 1

    print(f"\n  GROUNDING: {passed}/{len(test_cases)} passed")
    return passed, len(test_cases)


def test_verification():
    """Test the post-generation fact verifier."""
    lq, detector, user_name = setup()

    print("\n" + "=" * 70)
    print("FACT VERIFIER TESTS")
    print("=" * 70)

    test_cases = [
        # (response, expected_verified, expected_contradicted, description)
        (
            "Your father is Upananda.",
            1, 0,
            "Correct fact — should be VERIFIED"
        ),
        (
            "Your wife is Prabhashi.",
            1, 0,
            "Correct fact — should be VERIFIED"
        ),
        (
            "Your father is John.",
            0, 1,
            "Wrong fact — should be CONTRADICTED"
        ),
        (
            "Your wife is Sarah.",
            0, 1,
            "Wrong fact — should be CONTRADICTED"
        ),
        (
            "You work at Bitsmedia as a software engineer.",
            1, 0,
            "Correct work info — should be VERIFIED"
        ),
        (
            "You work at Google.",
            0, 1,
            "Wrong employer — should be CONTRADICTED"
        ),
        (
            "Your father is Upananda and your mother is Renuka.",
            2, 0,
            "Two correct facts — both should be VERIFIED"
        ),
        (
            "Your father is John and your wife is Prabhashi.",
            1, 1,
            "Mixed — one correct, one wrong"
        ),
        (
            "The weather is nice today.",
            0, 0,
            "No personal claims — nothing to verify"
        ),
        (
            "Your favorite color is blue.",
            0, 0,
            "Unknown claim — can't verify or contradict (UNVERIFIED)"
        ),
    ]

    passed = 0
    for response, expected_verified, expected_contradicted, desc in test_cases:
        verification = detector.post_generate(
            response=response,
            user_entity_name=user_name,
            query="test",
        )

        v_ok = verification.verified_count == expected_verified
        c_ok = verification.contradicted_count == expected_contradicted
        ok = v_ok and c_ok

        status = "✓" if ok else "✗"
        print(f"\n  {status} {desc}")
        print(f"    Response: \"{response}\"")
        print(f"    Verified: {verification.verified_count} (expected: {expected_verified}) {'✓' if v_ok else '✗'}")
        print(f"    Contradicted: {verification.contradicted_count} (expected: {expected_contradicted}) {'✓' if c_ok else '✗'}")
        print(f"    Unverified: {verification.unverified_count}")
        print(f"    Score: {verification.hallucination_score:.2f}")

        for claim in verification.claims:
            print(f"    → {claim.status.value}: \"{claim.text}\" | {claim.evidence}")

        if ok:
            passed += 1

    print(f"\n  VERIFICATION: {passed}/{len(test_cases)} passed")
    return passed, len(test_cases)


def test_should_regenerate():
    """Test the regeneration decision logic."""
    lq, detector, user_name = setup()

    print("\n" + "=" * 70)
    print("REGENERATION DECISION TESTS")
    print("=" * 70)

    test_cases = [
        ("Your father is John.", True, "Wrong fact → should regenerate"),
        ("Your father is Upananda.", False, "Correct fact → accept"),
        ("I don't have that information.", False, "Abstention → accept"),
        ("The weather is nice.", False, "No claims → accept"),
    ]

    passed = 0
    for response, expected_regen, desc in test_cases:
        verification = detector.post_generate(
            response=response, user_entity_name=user_name, query="who is my father",
        )
        grounding = detector.pre_generate(
            query="who is my father",
            graph_result=lq.query_graph("who is my father"),
            memories_retrieved=5,
            memory_context="[KNOWN FACT] Your father is Upananda",
            world_summary="",
        )

        should_regen = detector.should_regenerate(verification, grounding)
        ok = should_regen == expected_regen

        status = "✓" if ok else "✗"
        print(f"\n  {status} {desc}")
        print(f"    Response: \"{response}\"")
        print(f"    Regenerate: {should_regen} (expected: {expected_regen})")

        if ok:
            passed += 1

    print(f"\n  REGENERATION: {passed}/{len(test_cases)} passed")
    return passed, len(test_cases)


def test_correction_prompt():
    """Test correction prompt generation."""
    lq, detector, user_name = setup()

    print("\n" + "=" * 70)
    print("CORRECTION PROMPT TEST")
    print("=" * 70)

    # Simulate a hallucinated response
    verification = detector.post_generate(
        response="Your father is John and your wife is Sarah.",
        user_entity_name=user_name,
        query="who is my father",
    )

    correction = detector.correction_prompt(
        verification, "who is my father",
        memory_context="[KNOWN FACT] Your father is Upananda. Your wife is Prabhashi."
    )

    print(f"\n  Correction prompt:\n{correction}\n")

    has_wrong = "WRONG" in correction
    has_correct = "CORRECT" in correction
    ok = has_wrong and has_correct
    print(f"  Contains WRONG markers: {has_wrong}")
    print(f"  Contains CORRECT markers: {has_correct}")
    print(f"  {'✓' if ok else '✗'} Correction prompt is well-formed")

    return 1 if ok else 0, 1


if __name__ == "__main__":
    total_passed = 0
    total_tests = 0

    for test_fn in [test_grounding, test_verification, test_should_regenerate, test_correction_prompt]:
        p, t = test_fn()
        total_passed += p
        total_tests += t

    print("\n" + "=" * 70)
    print(f"TOTAL: {total_passed}/{total_tests} passed")
    if total_passed == total_tests:
        print("ALL TESTS PASSED ✓")
    else:
        print(f"FAILURES: {total_tests - total_passed}")
    print("=" * 70)

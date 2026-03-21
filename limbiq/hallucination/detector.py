"""
Hallucination Detector — orchestrates grounding + verification + correction.

Neurotransmitter analogy: Glutamate (excitatory inhibition).
Real glutamate triggers neural excitation. Here, the hallucination signal
triggers a "stop and re-examine" response — forcing the system to regenerate
with tighter constraints when confabulation is detected.

Pipeline:
1. PRE-GENERATION:  GroundingAnalyzer classifies query → builds constraints
2. POST-GENERATION: FactVerifier extracts claims → cross-checks graph/memory
3. DECISION:        If contradictions found → flag for regeneration
4. SIGNAL:          Fire Glutamate signal for audit trail

The detector does NOT call the LLM itself. It provides:
  - pre_generate()  → GroundingReport (inject into prompt)
  - post_generate() → VerificationReport (check response)
  - should_regenerate() → bool (based on verification)
  - correction_prompt() → str (constraints for regeneration)
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from limbiq.hallucination.grounding import (
    GroundingAnalyzer,
    GroundingReport,
    GroundingLevel,
)
from limbiq.hallucination.verifier import (
    FactVerifier,
    VerificationReport,
    ClaimStatus,
)
from limbiq.types import SignalEvent, SignalType

logger = logging.getLogger(__name__)


@dataclass
class HallucinationEvent:
    """Record of a detected hallucination for audit trail."""
    query: str
    response: str
    grounding_level: str
    claims_total: int = 0
    claims_verified: int = 0
    claims_contradicted: int = 0
    claims_unverified: int = 0
    hallucination_score: float = 0.0
    action_taken: str = ""          # "accepted", "regenerated", "flagged"
    timestamp: float = field(default_factory=time.time)
    contradictions: list = field(default_factory=list)


class HallucinationDetector:
    """
    Main orchestrator for hallucination detection and prevention.

    Usage in the engine:

        # 1. Before LLM generation
        grounding = detector.pre_generate(query, graph_result, process_result)
        # → inject grounding.constraint_prompt into system prompt
        # → use grounding.suggested_temperature

        # 2. After LLM generation
        verification = detector.post_generate(response, user_name, query)

        # 3. Check if regeneration needed
        if detector.should_regenerate(verification):
            correction = detector.correction_prompt(verification, query)
            # → regenerate with correction injected
            # → verify again (max 1 retry to avoid loops)

        # 4. Record for audit
        detector.record(query, response, grounding, verification, action)
    """

    def __init__(self, graph_store, memory_store, embedding_engine, signal_log=None):
        self.grounding = GroundingAnalyzer(graph_store, memory_store, embedding_engine)
        self.verifier = FactVerifier(graph_store, memory_store)
        self.signal_log = signal_log

        # Audit trail
        self._events: list[HallucinationEvent] = []
        self._stats = {
            "total_checks": 0,
            "hallucinations_caught": 0,
            "regenerations_triggered": 0,
            "contradictions_found": 0,
        }

    def pre_generate(
        self,
        query: str,
        graph_result: dict = None,
        memories_retrieved: int = 0,
        memory_context: str = "",
        world_summary: str = "",
    ) -> GroundingReport:
        """
        Pre-generation analysis. Call BEFORE the LLM generates.

        Returns a GroundingReport with:
        - constraint_prompt: inject this into the system/user prompt
        - suggested_temperature: use this for generation
        - level: GROUNDED / PARTIAL / UNGROUNDED
        """
        return self.grounding.analyze(
            query=query,
            graph_result=graph_result,
            memories_retrieved=memories_retrieved,
            memory_context=memory_context,
            world_summary=world_summary,
        )

    def post_generate(
        self,
        response: str,
        user_entity_name: str = "user",
        query: str = "",
        conversation_history: list[dict] = None,
    ) -> VerificationReport:
        """
        Post-generation verification. Call AFTER the LLM generates.

        Returns a VerificationReport with per-claim verification status.
        """
        return self.verifier.verify(
            response=response,
            user_entity_name=user_entity_name,
            query=query,
            conversation_history=conversation_history,
        )

    def should_regenerate(
        self,
        verification: VerificationReport,
        grounding: GroundingReport = None,
    ) -> bool:
        """
        Determine if the response should be regenerated.

        Triggers regeneration when:
        - Any claim is CONTRADICTED (wrong fact stated)
        - Hallucination score > 0.5 AND query was personal
        - Narrative fabrication detected (fabricated memories/events/experiences)
        """
        if verification.contradicted_count > 0:
            return True

        if grounding and grounding.query_type == "personal":
            if verification.hallucination_score > 0.5:
                return True

        # Narrative fabrication: unverified claims about shared history
        # are always hallucinations regardless of query type
        narrative_types = {
            "fabricated_memory", "fabricated_recall", "fabricated_event",
            "fabricated_interaction", "fabricated_state", "fabricated_preference",
            "fabricated_llm_persona", "fabricated_health", "fabricated_presupposition",
            "fabricated_attribution",
        }
        for claim in verification.claims:
            if claim.predicate in narrative_types and claim.status == ClaimStatus.UNVERIFIED:
                return True

        return False

    def correction_prompt(
        self,
        verification: VerificationReport,
        query: str,
        memory_context: str = "",
    ) -> str:
        """
        Build a correction prompt for regeneration.

        This is injected when the first response had contradictions.
        It explicitly tells the model what's WRONG and what's RIGHT.
        """
        parts = []
        parts.append(
            "CRITICAL CORRECTION: Your previous response contained incorrect information. "
            "You MUST fix the following errors:"
        )

        narrative_types = {
            "fabricated_memory", "fabricated_recall", "fabricated_event",
            "fabricated_interaction", "fabricated_state", "fabricated_preference",
            "fabricated_llm_persona", "fabricated_health", "fabricated_presupposition",
            "fabricated_attribution",
        }

        has_contradictions = False
        has_fabrications = False

        for claim in verification.claims:
            if claim.status == ClaimStatus.CONTRADICTED:
                has_contradictions = True
                parts.append(
                    f"\n  ✗ WRONG: \"{claim.text}\""
                    f"\n  ✓ CORRECT: {claim.evidence}"
                )
            elif claim.predicate in narrative_types and claim.status == ClaimStatus.UNVERIFIED:
                has_fabrications = True
                parts.append(
                    f"\n  ✗ FABRICATED: \"{claim.text}\""
                    f"\n    This never happened. You have no record of this."
                )

        if has_fabrications:
            parts.append(
                "\nDo NOT invent shared experiences, memories, visits, or events. "
                "You have NO memory of past interactions unless explicitly provided in context. "
                "Do NOT pretend to remember things that never happened."
            )

        parts.append(
            "\nAnswer the question again using ONLY the facts from the context. "
            "Do NOT repeat the errors above."
        )

        if memory_context:
            parts.append(f"\nAvailable facts:\n{memory_context}")

        return "\n".join(parts)

    def record(
        self,
        query: str,
        response: str,
        grounding: GroundingReport,
        verification: VerificationReport,
        action: str = "accepted",
    ) -> Optional[HallucinationEvent]:
        """
        Record a hallucination check result for audit trail.
        Fires a Glutamate signal if hallucination was detected.
        """
        self._stats["total_checks"] += 1

        event = HallucinationEvent(
            query=query,
            response=response[:500],
            grounding_level=grounding.level.value,
            claims_total=len(verification.claims),
            claims_verified=verification.verified_count,
            claims_contradicted=verification.contradicted_count,
            claims_unverified=verification.unverified_count,
            hallucination_score=verification.hallucination_score,
            action_taken=action,
            contradictions=[
                {"claim": c.text, "evidence": c.evidence}
                for c in verification.claims
                if c.status == ClaimStatus.CONTRADICTED
            ],
        )
        self._events.append(event)

        # Fire hallucination signal if issues detected
        if verification.contradicted_count > 0 or verification.hallucination_score > 0.5:
            self._stats["hallucinations_caught"] += 1
            self._stats["contradictions_found"] += verification.contradicted_count

            if action == "regenerated":
                self._stats["regenerations_triggered"] += 1

            if self.signal_log:
                signal_event = SignalEvent(
                    signal_type=SignalType.NOREPINEPHRINE,  # Reuse NE for now
                    trigger="hallucination_detected",
                    details={
                        "hallucination_score": verification.hallucination_score,
                        "contradicted": verification.contradicted_count,
                        "unverified": verification.unverified_count,
                        "action": action,
                        "contradictions": event.contradictions,
                    },
                )
                self.signal_log.log(signal_event)

            logger.warning(
                f"Hallucination detected: score={verification.hallucination_score:.2f}, "
                f"contradictions={verification.contradicted_count}, "
                f"action={action}"
            )

        return event

    def get_stats(self) -> dict:
        """Return hallucination detection statistics."""
        return dict(self._stats)

    def get_recent_events(self, limit: int = 20) -> list[dict]:
        """Return recent hallucination events for dashboard."""
        events = self._events[-limit:]
        return [
            {
                "query": e.query[:100],
                "grounding_level": e.grounding_level,
                "hallucination_score": e.hallucination_score,
                "claims_total": e.claims_total,
                "claims_verified": e.claims_verified,
                "claims_contradicted": e.claims_contradicted,
                "action": e.action_taken,
                "contradictions": e.contradictions,
                "timestamp": e.timestamp,
            }
            for e in events
        ]

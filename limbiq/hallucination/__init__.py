"""
Hallucination Detection & Prevention for Limbiq.

Three-layer defense against LLM confabulation:

1. GroundingAnalyzer (pre-generation)
   - Classifies queries as grounded/ungrounded before LLM runs
   - Injects explicit abstention instructions when no relevant context exists
   - Prevents the #1 hallucination vector: answering without evidence

2. FactVerifier (post-generation)
   - Extracts factual claims from LLM responses
   - Cross-checks each claim against the knowledge graph and memory store
   - Returns verified/unverified/contradicted status per claim

3. HallucinationDetector (orchestrator)
   - Coordinates grounding + verification into a single pipeline
   - Drives self-correction: if ungrounded claims detected, triggers regeneration
   - Fires a hallucination signal (neurotransmitter analogy: "Glutamate")
"""

from limbiq.hallucination.grounding import GroundingAnalyzer
from limbiq.hallucination.verifier import FactVerifier
from limbiq.hallucination.detector import HallucinationDetector

__all__ = ["GroundingAnalyzer", "FactVerifier", "HallucinationDetector"]

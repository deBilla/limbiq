"""
Fact Verifier — post-generation defense against hallucination.

After the LLM generates a response, this module:
1. Extracts factual claims (entity-relation-entity triples)
2. Cross-checks each claim against the knowledge graph
3. Cross-checks against stored memories
4. Returns a VerificationReport with per-claim status

Claim statuses:
  VERIFIED     — Claim matches a graph relation or stored memory
  UNVERIFIED   — Claim can't be confirmed (no matching data)
  CONTRADICTED — Claim conflicts with stored knowledge

The detector uses this to decide whether to accept, flag, or regenerate.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ClaimStatus(Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"


@dataclass
class Claim:
    """A single factual claim extracted from a response."""
    text: str                          # Original claim text
    subject: str = ""                  # Entity that the claim is about
    predicate: str = ""                # Relationship type
    object_value: str = ""             # The claimed value
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    evidence: str = ""                 # What confirmed/contradicted it
    confidence: float = 0.0            # Confidence in the verification


@dataclass
class VerificationReport:
    """Result of verifying all claims in a response."""
    claims: list[Claim] = field(default_factory=list)
    verified_count: int = 0
    unverified_count: int = 0
    contradicted_count: int = 0
    hallucination_score: float = 0.0   # 0 = clean, 1 = all hallucinated
    flagged_text: str = ""             # Response with flags inserted


# Patterns to extract personal claims from LLM responses.
# These capture "Your X is Y", "You work at Y", "You live in Y" etc.
CLAIM_PATTERNS = [
    # "Your father is Upananda" / "Your wife's name is Prabhashi"
    # Use [A-Z]\w+ to capture ONLY the proper noun (no trailing words like "and")
    (r"[Yy]our\s+(\w+(?:'s\s+\w+)?)\s+(?:is|was|are)\s+(?:named?\s+)?([A-Z]\w+)\b",
     lambda m: ("user", m.group(1).lower().strip(), m.group(2).strip())),

    # Appositive: "your dad, John" / "your father, Upananda" / "your wife Prabhashi"
    (r"[Yy]our\s+(dad|father|mother|mom|wife|husband|brother|sister|son|daughter|partner|boss)[,\s]+([A-Z]\w+)\b",
     lambda m: ("user", _NORMALIZE_RELATION.get(m.group(1).lower(), m.group(1).lower()), m.group(2).strip())),

    # Compound relation appositive: "your father-in-law, George" / "your mother-in-law, Susan"
    (r"[Yy]our\s+(father-in-law|mother-in-law|brother-in-law|sister-in-law|son-in-law|daughter-in-law|step-father|step-mother|step-dad|step-mom|ex-wife|ex-husband)[,\s]+([A-Z]\w+)\b",
     lambda m: ("user", _NORMALIZE_RELATION.get(m.group(1).lower(), m.group(1).lower()), m.group(2).strip())),

    # "You have a sister named Dilani"
    (r"[Yy]ou\s+have\s+(?:a|an)\s+(\w+)\s+(?:named?|called)\s+([A-Z]\w+)\b",
     lambda m: ("user", m.group(1).lower(), m.group(2).strip())),

    # "You work at Bitsmedia" / "You live in Singapore"
    # Capture the place/company name but stop at punctuation, "as", "and", etc.
    (r"[Yy]ou\s+(work|live|study|teach|reside)\s+(?:at|in|for)\s+([A-Z]\w+)\b",
     lambda m: ("user", f"{m.group(1).lower()}_at", m.group(2).strip())),

    # "Dimuthu works at ..." / "Prabhashi is ..."
    (r"([A-Z]\w+)\s+(?:is|was|are)\s+(?:your\s+)?(\w+)\b",
     lambda m: (m.group(1).strip(), "is", m.group(2).strip())),

    # "X's father is Y" / "X's wife is Y"
    (r"([A-Z]\w+)(?:'s|s)\s+(father|mother|wife|husband|brother|sister|son|daughter|boss)\s+(?:is|was)\s+([A-Z]\w+)\b",
     lambda m: (m.group(1).strip(), m.group(2).lower(), m.group(3).strip())),

    # Numbered or listed facts: "1. Your name is X" or "- Your job is X"
    (r"(?:^|\n)\s*(?:\d+\.|-)\s*[Yy]our\s+(\w+)\s+(?:is|are)\s+(.+?)(?:\.|$)",
     lambda m: ("user", m.group(1).lower(), m.group(2).strip())),
]

# Normalize informal relation words to graph predicates
_NORMALIZE_RELATION = {
    "dad": "father",
    "mom": "mother",
    "partner": "wife",  # Could be husband, but we check both
    "father-in-law": "father_in_law_of",
    "mother-in-law": "mother_in_law_of",
    "brother-in-law": "brother_in_law",
    "sister-in-law": "sister_in_law",
    "step-father": "step_father",
    "step-mother": "step_mother",
    "step-dad": "step_father",
    "step-mom": "step_mother",
    "ex-wife": "ex_wife",
    "ex-husband": "ex_husband",
}

# Relations in the graph that we can verify against
VERIFIABLE_PREDICATES = {
    "father", "mother", "wife", "husband", "brother", "sister",
    "son", "daughter", "boss", "friend", "colleague",
    "works_at", "lives_in", "role", "married_to",
    "father_in_law_of", "mother_in_law_of",
    "sibling", "parent",
}


class FactVerifier:
    """
    Verifies factual claims in LLM responses against the knowledge graph.

    Usage:
        verifier = FactVerifier(graph_store, memory_store)
        report = verifier.verify(response_text, user_entity_name="Dimuthu")

        if report.contradicted_count > 0:
            # Hallucination detected — trigger correction
            ...
    """

    def __init__(self, graph_store, memory_store):
        self.graph = graph_store
        self.store = memory_store

    def verify(
        self,
        response: str,
        user_entity_name: str = "user",
        query: str = "",
        conversation_history: list[dict] = None,
    ) -> VerificationReport:
        """
        Extract and verify all factual claims in a response.

        Args:
            response: The LLM's generated text
            user_entity_name: The user's entity name in the graph
            query: Original query (for context)
            conversation_history: Recent conversation messages (used to avoid
                                  flagging things mentioned in conversation as fabricated)
        """
        # Build a text blob from conversation history for reference checking
        self._conversation_text = ""
        if conversation_history:
            self._conversation_text = " ".join(
                m.get("content", "") for m in conversation_history
            ).lower()

        claims = self._extract_claims(response, user_entity_name)

        # Also check for unknown entity references (proper nouns not in graph)
        phantom_claims = self._detect_phantom_entities(response, user_entity_name, claims)
        claims.extend(phantom_claims)

        # Check for narrative fabrication (invented experiences/events/shared history)
        narrative_claims = self._detect_narrative_fabrication(response, user_entity_name)
        claims.extend(narrative_claims)

        if not claims:
            return VerificationReport(hallucination_score=0.0)

        verified = 0
        unverified = 0
        contradicted = 0

        for claim in claims:
            self._verify_claim(claim, user_entity_name)

            if claim.status == ClaimStatus.VERIFIED:
                verified += 1
            elif claim.status == ClaimStatus.CONTRADICTED:
                contradicted += 1
            else:
                unverified += 1

        total = len(claims)
        # Hallucination score: contradictions are worst, unverified is concerning
        hallucination_score = (
            (contradicted * 1.0 + unverified * 0.3) / total
            if total > 0 else 0.0
        )

        return VerificationReport(
            claims=claims,
            verified_count=verified,
            unverified_count=unverified,
            contradicted_count=contradicted,
            hallucination_score=min(1.0, hallucination_score),
            flagged_text=self._flag_response(response, claims),
        )

    def _extract_claims(self, response: str, user_name: str) -> list[Claim]:
        """Extract factual claims from an LLM response using pattern matching."""
        claims = []
        seen = set()  # Avoid duplicate claims

        for pattern, extractor in CLAIM_PATTERNS:
            for match in re.finditer(pattern, response, re.MULTILINE):
                try:
                    subject, predicate, obj = extractor(match)

                    # Normalize subject
                    if subject.lower() in ("you", "your", "user"):
                        subject = user_name

                    claim_key = (subject.lower(), predicate.lower(), obj.lower())
                    if claim_key in seen:
                        continue
                    seen.add(claim_key)

                    claims.append(Claim(
                        text=match.group(0).strip(),
                        subject=subject,
                        predicate=predicate,
                        object_value=obj,
                    ))
                except Exception:
                    continue

        return claims

    def _detect_phantom_entities(
        self, response: str, user_name: str, existing_claims: list[Claim]
    ) -> list[Claim]:
        """
        Find proper nouns in the response that don't exist in the knowledge graph.

        These are "phantom entities" — names the LLM invented. Example:
        "check in on Dexter" when no entity named Dexter exists.

        Only flags names that:
        - Are capitalized proper nouns (not common words)
        - Don't appear in the knowledge graph
        - Don't appear in any stored memories
        - Aren't the user's own name
        - Aren't already captured by claim extraction patterns
        """
        phantom_claims = []

        # Names already captured by claim patterns
        already_found = set()
        for c in existing_claims:
            already_found.add(c.object_value.lower())
            already_found.add(c.subject.lower())

        # Use spaCy to find ACTUAL proper nouns instead of regex guessing.
        # This avoids flagging normal English words as phantom entities.
        proper_nouns = set()
        try:
            from limbiq.graph.entities import _get_nlp
            nlp = _get_nlp()
            if nlp:
                doc = nlp(response)
                skip_labels = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}
                for ent in doc.ents:
                    if ent.label_ in skip_labels:
                        continue
                    name = ent.text.strip()
                    if len(name) >= 3 and name.lower() not in already_found:
                        proper_nouns.add(name)
            else:
                # Fallback: simple regex but with aggressive filtering
                common_words = {
                    "the", "this", "that", "here", "there", "then", "when", "where",
                    "what", "which", "who", "how", "why", "just", "also", "well",
                    "same", "each", "every", "some", "many", "much", "more", "most",
                    "time", "would", "could", "should", "might", "will", "shall",
                    "been", "being", "have", "make", "take", "give", "keep", "find",
                    "note", "notes", "feed", "small", "light", "monitor", "adjust",
                    "position", "elevate", "schedule", "feeding", "sure", "glad",
                    "sorry", "please", "thank", "thanks", "hello",
                }
                for match in re.finditer(r'\b([A-Z][a-z]\w{2,})\b', response):
                    name = match.group(1)
                    if name.lower() not in common_words and name.lower() not in already_found:
                        proper_nouns.add(name)
        except Exception:
            pass  # Skip phantom detection on error

        # Remove the user's own name
        proper_nouns.discard(user_name)

        # Check each proper noun against the graph
        for name in proper_nouns:
            entity = self.graph.find_entity_by_name(name)
            if entity:
                continue  # Known entity — not phantom

            # Check memories too
            try:
                rows = self.store.db.execute(
                    "SELECT COUNT(*) FROM memories WHERE content LIKE ? AND is_suppressed = 0",
                    (f"%{name}%",),
                ).fetchone()
                if rows and rows[0] > 0:
                    continue  # Found in memories — not phantom
            except Exception:
                pass

            # This is a phantom entity — the LLM invented a person/place
            phantom_claims.append(Claim(
                text=f"Reference to '{name}'",
                subject=user_name,
                predicate="knows",
                object_value=name,
                status=ClaimStatus.UNVERIFIED,
                evidence=f"Entity '{name}' does not exist in knowledge graph or memories",
                confidence=0.1,
            ))

        return phantom_claims

    def _detect_narrative_fabrication(
        self, response: str, user_name: str
    ) -> list[Claim]:
        """
        Detect fabricated experiences, events, and shared history.

        8B models love to invent:
        - "I remember you mentioning..."
        - "your last visit..."
        - "you told me about your..."
        - "when we talked about..."
        - "you enjoyed trying my..."

        These are assertions about past interactions that should exist as
        stored memories. If they don't, they're hallucinated.
        """
        fabrication_claims = []

        # Patterns that assert shared history or user experiences
        NARRATIVE_PATTERNS = [
            # "I remember you mentioning/telling/saying X"
            (r"[Ii]\s+remember\s+you\s+(mentioning|telling|saying|talking about)\s+(.+?)(?:\.|,|!)",
             "fabricated_memory"),
            # "you mentioned/told me/said X"
            (r"[Yy]ou\s+(mentioned|told me|said|shared|brought up)\s+(?:that\s+|how\s+)?(.+?)(?:\.|,|!)",
             "fabricated_recall"),
            # "your last/recent visit/appointment/session"
            (r"[Yy]our\s+(last|recent|previous|earlier)\s+(visit|appointment|session|trip|call|conversation|meeting)\b",
             "fabricated_event"),
            # "when we last spoke/met/talked"
            (r"[Ww]hen\s+we\s+(last\s+)?(spoke|met|talked|chatted|discussed)\b",
             "fabricated_interaction"),
            # "you were feeling/doing/having"
            (r"[Yy]ou\s+were\s+(feeling|doing|having|going through|dealing with)\s+(.+?)(?:\.|,|!)",
             "fabricated_state"),
            # "after your X" (surgery, trip, move, etc.)
            (r"after\s+your\s+(surgery|trip|move|vacation|promotion|illness|recovery|accident|wedding|divorce)\b",
             "fabricated_event"),
            # "you enjoyed/liked/loved trying/doing/eating X"
            (r"[Yy]ou\s+(enjoyed|liked|loved|appreciated)\s+(trying|doing|eating|reading|watching|visiting)\s+(.+?)(?:\.|,|!)",
             "fabricated_preference"),
            # "my wife's/husband's/mother's famous/homemade X" (LLM inventing its own backstory)
            (r"my\s+(wife|husband|mother|father|grandmother|grandfather)(?:'s|s)\s+(?:famous\s+|homemade\s+|special\s+)?(\w+(?:\s+\w+){0,3})\s+(recipe|dish|cooking|cake|pie)\b",
             "fabricated_llm_persona"),
            # "doing better" / "feeling better" (asserting health state)
            (r"(?:you'?re?|you\s+are)\s+(doing|feeling)\s+better\b",
             "fabricated_health"),
            # "glad to see/hear that you..." (presupposes shared context)
            (r"(?:glad|happy|pleased)\s+to\s+(?:see|hear)\s+(?:that\s+)?you",
             "fabricated_presupposition"),
            # "I was talking to your dad/wife/X" — LLM claiming real-world interaction
            (r"[Ii]\s+was\s+(?:just\s+)?(?:talking|speaking|chatting)\s+(?:to|with)\s+your\s+(\w+)",
             "fabricated_interaction"),
            # "he/she mentioned/said how..." — attributing fabricated quotes to others
            (r"(?:he|she|they)\s+(mentioned|said|told me|noted|shared)\s+(?:how\s+|that\s+)?(.+?)(?:\.|,|!)",
             "fabricated_attribution"),
            # "we could catch up" / "nice catching up" — implies prior interaction
            (r"(?:we\s+could\s+catch\s+up|catching\s+up|good\s+to\s+catch\s+up|glad\s+we\s+could\s+catch\s+up)",
             "fabricated_interaction"),
            # "that new/recent X at work/school" — fabricating user's life events
            (r"that\s+(new|recent|upcoming|big)\s+(\w+(?:\s+\w+)?)\s+(?:at|in|for)\s+(?:work|school|your)\b",
             "fabricated_event"),
            # "I spoke with / I met with / I ran into your X"
            (r"[Ii]\s+(?:spoke|met|ran into|saw|visited|called)\s+(?:with\s+)?your\s+(\w+)",
             "fabricated_interaction"),
            # "your dad/mom/wife is proud/happy/worried" — attributing emotions to user's relations
            (r"your\s+(dad|father|mother|mom|wife|husband|brother|sister)\s+(?:is|was|seems?)\s+(proud|happy|worried|excited|concerned|upset)",
             "fabricated_attribution"),
            # "our last conversation about X" / "last time we spoke about X"
            (r"(?:our|the)\s+(?:last|previous|recent)\s+(?:conversation|discussion|talk|chat)\s+(?:about|regarding|on)\b",
             "fabricated_interaction"),
            # Third-person "doing/feeling better" — "X is doing better"
            (r"(?:is|was)\s+doing\s+better\b",
             "fabricated_health"),
            # "your father-in-law/mother-in-law is doing/feeling X"
            (r"your\s+(?:father|mother|brother|sister|son|daughter)[-\s]in[-\s]law\b.*?(?:is|was)\s+(?:doing|feeling)\s+\w+",
             "fabricated_health"),
        ]

        seen_types = set()
        resp_lower = response.lower()

        for pattern, fab_type in NARRATIVE_PATTERNS:
            for match in re.finditer(pattern, response, re.IGNORECASE):
                if fab_type in seen_types:
                    continue  # One per fabrication type

                matched_text = match.group(0).strip()

                # Verify: does this event/experience exist in stored memories?
                memory_confirms = self._search_memories_for_narrative(matched_text)

                if not memory_confirms:
                    seen_types.add(fab_type)
                    fabrication_claims.append(Claim(
                        text=matched_text,
                        subject="assistant",
                        predicate=fab_type,
                        object_value=matched_text,
                        status=ClaimStatus.UNVERIFIED,
                        evidence=f"No stored memory confirms this {fab_type.replace('_', ' ')}",
                        confidence=0.1,
                    ))

        return fabrication_claims

    def _search_memories_for_narrative(self, narrative_text: str) -> bool:
        """
        Check if a narrative claim has any supporting evidence in memories
        OR in the recent conversation history.

        If the claim references something the user said in this conversation,
        it's not a hallucination — the model is correctly referencing context.
        """
        # First check conversation history — if mentioned there, it's grounded
        conv_text = getattr(self, "_conversation_text", "")
        if conv_text:
            narrative_lower = narrative_text.lower()
            # Extract key words from the narrative
            key_words = [
                w.lower().strip(".,!?'\"")
                for w in narrative_text.split()
                if len(w) > 4 and w.lower() not in {
                    "about", "could", "would", "should", "their", "there",
                    "where", "which", "these", "those", "other",
                }
            ]
            # If 2+ key words from the narrative appear in conversation, it's grounded
            if key_words:
                matches = sum(1 for w in key_words if w in conv_text)
                if matches >= 2 or (matches >= 1 and any(len(w) > 6 for w in key_words if w in conv_text)):
                    return True
        # Extended stopwords including common words that appear in many contexts
        stopwords = {
            "i", "you", "your", "my", "me", "we", "the", "a", "an",
            "is", "was", "were", "are", "am", "been", "be", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should",
            "that", "this", "it", "to", "of", "in", "on", "at", "for",
            "with", "about", "how", "when", "where", "what", "who",
            "remember", "mentioned", "told", "said", "shared",
            "last", "recent", "previous", "glad", "happy", "pleased",
            "see", "hear", "doing", "feeling", "better", "enjoyed",
            "liked", "loved", "trying", "much", "just", "other", "day",
            # Common context words that appear in many memories
            "new", "work", "project", "good", "great", "really", "time",
            "thing", "way", "got", "going", "want", "like", "know",
            "think", "take", "taking", "make", "come", "look",
        }

        words = [
            w.lower().strip(".,!?'\"")
            for w in narrative_text.split()
            if len(w) > 3 and w.lower().strip(".,!?'\"") not in stopwords
        ]

        if not words:
            return False

        try:
            # Strategy: need 3+ words in the SAME memory for confirmation,
            # or 2+ words if one is a proper noun / distinctive word
            for i, word in enumerate(words[:5]):
                rows = self.store.db.execute(
                    "SELECT content FROM memories "
                    "WHERE content LIKE ? AND is_suppressed = 0 "
                    "LIMIT 10",
                    (f"%{word}%",),
                ).fetchall()

                for (content,) in rows:
                    content_lower = content.lower()
                    # Count how many of our words appear in this same memory
                    match_count = sum(
                        1 for w in words if w in content_lower
                    )
                    # Check for proper noun matches (case-sensitive)
                    has_proper = any(
                        w in content  # case-sensitive
                        for w in narrative_text.split()
                        if w[0:1].isupper() and len(w) > 3
                        and w.lower() not in stopwords
                    )

                    if match_count >= 3:
                        return True
                    if match_count >= 2 and has_proper:
                        return True

            return False
        except Exception:
            return False

    def _verify_claim(self, claim: Claim, user_name: str):
        """
        Verify a single claim against the knowledge graph and memories.
        Updates claim.status, claim.evidence, and claim.confidence in place.
        """
        # Normalize predicate for lookup
        pred_normalized = claim.predicate.lower().replace(" ", "_")

        # Strategy 1: Direct graph lookup
        graph_verified = self._check_graph(
            claim.subject, pred_normalized, claim.object_value, user_name
        )

        if graph_verified is not None:
            if graph_verified["matches"]:
                claim.status = ClaimStatus.VERIFIED
                claim.evidence = f"Graph: {graph_verified['evidence']}"
                claim.confidence = graph_verified.get("confidence", 1.0)
                return
            elif graph_verified["has_different_value"]:
                claim.status = ClaimStatus.CONTRADICTED
                claim.evidence = (
                    f"Graph says: {graph_verified['evidence']} "
                    f"(response claims: {claim.object_value})"
                )
                claim.confidence = graph_verified.get("confidence", 0.9)
                return

        # Strategy 2: Memory search
        memory_verified = self._check_memories(claim)
        if memory_verified is not None:
            claim.status = memory_verified["status"]
            claim.evidence = memory_verified["evidence"]
            claim.confidence = memory_verified.get("confidence", 0.5)
            return

        # Strategy 3: Check if the claimed entity even exists in our graph
        entity = self.graph.find_entity_by_name(claim.object_value)
        if entity:
            # Entity exists, but we can't confirm this specific claim
            claim.status = ClaimStatus.UNVERIFIED
            claim.evidence = f"Entity '{claim.object_value}' known, but relation unconfirmed"
            claim.confidence = 0.3
        else:
            # Unknown entity — likely hallucinated if this is a personal claim
            if claim.subject.lower() == user_name.lower():
                claim.status = ClaimStatus.UNVERIFIED
                claim.evidence = f"Entity '{claim.object_value}' not in knowledge graph"
                claim.confidence = 0.1
            else:
                claim.status = ClaimStatus.UNVERIFIED
                claim.evidence = "Cannot verify — no matching data"
                claim.confidence = 0.2

    def _check_graph(
        self, subject: str, predicate: str, object_value: str, user_name: str
    ) -> Optional[dict]:
        """Check a claim against the knowledge graph."""
        try:
            # Find subject entity
            subj_entity = self.graph.find_entity_by_name(subject)
            if not subj_entity:
                # Try user name
                if subject.lower() in ("user", user_name.lower()):
                    subj_entity = self.graph.find_entity_by_name(user_name)
            if not subj_entity:
                return None

            # Look up all relations from this subject with this predicate
            rows = self.graph.db.execute(
                "SELECT r.id, r.object_id, r.confidence, r.is_inferred, e.name "
                "FROM relations r JOIN entities e ON r.object_id = e.id "
                "WHERE r.subject_id = ? AND r.predicate = ?",
                (subj_entity.id, predicate),
            ).fetchall()

            if not rows:
                # Try synonymous predicates
                synonyms = self._predicate_synonyms(predicate)
                for syn in synonyms:
                    rows = self.graph.db.execute(
                        "SELECT r.id, r.object_id, r.confidence, r.is_inferred, e.name "
                        "FROM relations r JOIN entities e ON r.object_id = e.id "
                        "WHERE r.subject_id = ? AND r.predicate = ?",
                        (subj_entity.id, syn),
                    ).fetchall()
                    if rows:
                        break

            if not rows:
                # Try REVERSE lookup for _of predicates (e.g., father_in_law_of)
                # "your father-in-law is X" → stored as "X → father_in_law_of → user"
                reverse_preds = [predicate] + self._predicate_synonyms(predicate)
                for rev_pred in reverse_preds:
                    rows = self.graph.db.execute(
                        "SELECT r.id, r.subject_id, r.confidence, r.is_inferred, e.name "
                        "FROM relations r JOIN entities e ON r.subject_id = e.id "
                        "WHERE r.object_id = ? AND r.predicate = ?",
                        (subj_entity.id, rev_pred),
                    ).fetchall()
                    if rows:
                        break

            if not rows:
                return None

            # Check if any relation matches the claimed value
            for _, obj_id, confidence, is_inferred, obj_name in rows:
                if obj_name.lower() == object_value.lower():
                    return {
                        "matches": True,
                        "has_different_value": False,
                        "evidence": f"{subject} → {predicate} → {obj_name}",
                        "confidence": confidence,
                    }

            # Relations exist but with different values — CONTRADICTION
            actual_values = [row[4] for row in rows]
            return {
                "matches": False,
                "has_different_value": True,
                "evidence": f"{subject} → {predicate} → {', '.join(actual_values)}",
                "confidence": max(row[2] for row in rows),
            }

        except Exception:
            return None

    def _check_memories(self, claim: Claim) -> Optional[dict]:
        """Check a claim against stored memories via text matching."""
        try:
            # Search for the object value in memories
            obj_lower = claim.object_value.lower()
            pred_lower = claim.predicate.lower()

            # Query memories that mention the claimed value
            rows = self.store.db.execute(
                "SELECT content, confidence FROM memories "
                "WHERE is_suppressed = 0 AND content LIKE ? "
                "ORDER BY confidence DESC LIMIT 5",
                (f"%{claim.object_value}%",),
            ).fetchall()

            if not rows:
                return None

            # Check if any memory confirms the relationship
            for content, confidence in rows:
                content_lower = content.lower()
                # Check if memory contains both the predicate concept and the value
                if obj_lower in content_lower:
                    if pred_lower in content_lower or claim.subject.lower() in content_lower:
                        return {
                            "status": ClaimStatus.VERIFIED,
                            "evidence": f"Memory: '{content[:100]}'",
                            "confidence": confidence,
                        }

            # Found the entity in memories but in a different context
            return {
                "status": ClaimStatus.UNVERIFIED,
                "evidence": f"Entity mentioned in memories but relationship unconfirmed",
                "confidence": 0.3,
            }

        except Exception:
            return None

    def _predicate_synonyms(self, predicate: str) -> list[str]:
        """Return synonym predicates for fuzzy matching."""
        synonyms = {
            "father": ["dad", "parent"],
            "mother": ["mom", "parent"],
            "wife": ["spouse", "married_to", "partner"],
            "husband": ["spouse", "married_to", "partner"],
            "work_at": ["works_at", "employed_at"],
            "works_at": ["work_at", "employed_at"],
            "live_at": ["lives_in", "resides_in"],
            "lives_in": ["live_at", "resides_in", "based_in"],
            "is": ["role", "type", "occupation"],
        }
        return synonyms.get(predicate.lower(), [])

    def _flag_response(self, response: str, claims: list[Claim]) -> str:
        """
        Annotate the response text with verification flags.
        Used for debugging and the playground UI.
        """
        flagged = response
        for claim in claims:
            if claim.status == ClaimStatus.CONTRADICTED:
                flag = f" [⚠ CONTRADICTED: {claim.evidence}]"
                flagged = flagged.replace(claim.text, claim.text + flag)
            elif claim.status == ClaimStatus.UNVERIFIED and claim.confidence < 0.3:
                flag = f" [? UNVERIFIED]"
                flagged = flagged.replace(claim.text, claim.text + flag)
        return flagged

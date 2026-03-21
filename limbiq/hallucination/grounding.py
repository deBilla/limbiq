"""
Grounding Analyzer — pre-generation defense against hallucination.

Determines whether a query can be answered from available context
(knowledge graph + memories) BEFORE the LLM generates a response.

Three grounding levels:
  GROUNDED    — Graph or memories have a direct answer. LLM should use it.
  PARTIAL     — Some relevant context exists but may not fully answer.
  UNGROUNDED  — No relevant information. LLM must abstain or hedge.

The analyzer produces a GroundingReport that the engine uses to:
  - Inject explicit constraints into the prompt
  - Set temperature/sampling parameters
  - Decide whether to allow free generation or force constrained mode
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GroundingLevel(Enum):
    GROUNDED = "grounded"       # Direct answer available
    PARTIAL = "partial"         # Some context, incomplete
    UNGROUNDED = "ungrounded"   # No relevant info


@dataclass
class GroundingReport:
    """Pre-generation analysis of query answerability."""
    level: GroundingLevel
    query_type: str                    # "personal", "factual", "opinion", "meta", "general"
    graph_has_answer: bool = False
    memory_relevance_score: float = 0  # 0-1, how relevant retrieved memories are
    relevant_fact_count: int = 0
    known_entities_mentioned: list = field(default_factory=list)  # Entities from query found in graph
    constraint_prompt: str = ""        # Injected into LLM prompt
    suggested_temperature: float = 0.7 # Lower when grounded, higher for creative


# Patterns that indicate personal queries (about the user)
PERSONAL_PATTERNS = [
    r"\bmy\s+\w+",                                    # my father, my wife, my job
    r"\b(?:who|what|where)\s+(?:is|are|was)\s+my\b",  # who is my X
    r"\bdo\s+(?:you|i)\s+(?:know|remember|have)\b",   # do you know/remember
    r"\babout\s+me\b",
    r"\bwhat(?:'s)?\s+my\b",                           # what's my name
    r"\bwhere\s+do\s+i\b",                             # where do i live
    r"\bam\s+i\b",
    r"\bdo\s+i\b",
    r"\bhave\s+i\b",
    r"\btell\s+me\s+about\s+(?:my|me)\b",
    r"\bremember\s+(?:when|that|my)\b",
    r"\bwhat\s+did\s+(?:i|we)\b",
]

# Patterns that indicate general/factual queries (not about user)
GENERAL_PATTERNS = [
    r"\bwhat\s+is\s+(?:a|an|the)\b",     # what is a/the X (general knowledge)
    r"\bhow\s+(?:does|do|to|can)\b",      # how does X work
    r"\bexplain\b",
    r"\bdefine\b",
    r"\bwhy\s+(?:does|do|is|are)\b",      # why does X happen
    r"\bwhat\s+(?:causes|happens)\b",
    r"\bcan\s+you\s+(?:help|write|code|create|make|explain)\b",
]

# Meta/conversational patterns
META_PATTERNS = [
    r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bthanks\b", r"\bthank you\b",
    r"\bhow\s+are\s+you\b", r"\bwhat\s+can\s+you\b", r"\bwho\s+are\s+you\b",
    r"\bgood\s+(?:morning|afternoon|evening|night)\b",
]


class GroundingAnalyzer:
    """
    Analyzes whether a query can be answered from limbiq's knowledge.

    Usage:
        analyzer = GroundingAnalyzer(graph_store, memory_store, embeddings)
        report = analyzer.analyze(query, graph_result, process_result)

        if report.level == GroundingLevel.UNGROUNDED:
            # Inject abstention constraint
            prompt += report.constraint_prompt
    """

    def __init__(self, graph_store, memory_store, embedding_engine):
        self.graph = graph_store
        self.store = memory_store
        self.embeddings = embedding_engine

    def analyze(
        self,
        query: str,
        graph_result: dict = None,
        memories_retrieved: int = 0,
        memory_context: str = "",
        world_summary: str = "",
    ) -> GroundingReport:
        """
        Analyze a query for grounding level.

        Args:
            query: The user's message
            graph_result: Result from graph_query.try_answer() (if already called)
            memories_retrieved: Count of relevant memories found
            memory_context: The assembled context string
            world_summary: Graph-derived world summary
        """
        query_type = self._classify_query(query)

        # Meta/conversational queries don't need grounding
        if query_type == "meta":
            return GroundingReport(
                level=GroundingLevel.GROUNDED,
                query_type=query_type,
                constraint_prompt="",
                suggested_temperature=0.7,
            )

        # General knowledge queries — LLM can answer from training data
        if query_type == "general":
            return GroundingReport(
                level=GroundingLevel.GROUNDED,
                query_type=query_type,
                constraint_prompt="",
                suggested_temperature=0.7,
            )

        # Personal queries — these MUST be grounded in stored knowledge
        graph_has_answer = False
        if graph_result and graph_result.get("answered"):
            graph_has_answer = True

        # Check which entities from the query exist in our graph
        known_entities = self._find_known_entities(query)

        # Score memory relevance by checking if context contains query keywords
        relevance = self._score_memory_relevance(query, memory_context, world_summary)

        # Count actual relevant facts (not just "User said:" fragments)
        fact_count = self._count_relevant_facts(memory_context)

        # Determine grounding level
        if graph_has_answer:
            level = GroundingLevel.GROUNDED
        elif relevance > 0.6 and fact_count >= 2:
            level = GroundingLevel.GROUNDED
        elif relevance > 0.4 and (fact_count >= 1 or known_entities):
            level = GroundingLevel.PARTIAL
        elif known_entities:
            level = GroundingLevel.PARTIAL
        else:
            level = GroundingLevel.UNGROUNDED

        # Build constraint prompt based on grounding level
        constraint = self._build_constraint(level, query_type, known_entities, query)

        # Adjust temperature
        if level == GroundingLevel.GROUNDED:
            temp = 0.3  # Low creativity, stick to facts
        elif level == GroundingLevel.PARTIAL:
            temp = 0.4  # Slightly more flexible
        else:
            temp = 0.2  # Very constrained for abstention

        return GroundingReport(
            level=level,
            query_type=query_type,
            graph_has_answer=graph_has_answer,
            memory_relevance_score=relevance,
            relevant_fact_count=fact_count,
            known_entities_mentioned=known_entities,
            constraint_prompt=constraint,
            suggested_temperature=temp,
        )

    def _classify_query(self, query: str) -> str:
        """Classify query type to determine grounding requirements."""
        q_lower = query.lower().strip()

        for pattern in META_PATTERNS:
            if re.search(pattern, q_lower):
                return "meta"

        for pattern in PERSONAL_PATTERNS:
            if re.search(pattern, q_lower):
                return "personal"

        for pattern in GENERAL_PATTERNS:
            if re.search(pattern, q_lower):
                return "general"

        # Check for user-specific entity mentions
        if self._find_known_entities(query):
            return "personal"

        # Default: if it's a question, treat as factual
        if "?" in query:
            return "factual"

        return "general"

    def _find_known_entities(self, query: str) -> list[str]:
        """Find entity names from the knowledge graph mentioned in the query."""
        known = []
        try:
            rows = self.graph.db.execute(
                "SELECT name FROM entities WHERE entity_type != 'unknown'"
            ).fetchall()
            for (name,) in rows:
                if name and len(name) > 2:
                    # Case-insensitive search for entity name in query
                    if re.search(r'\b' + re.escape(name) + r'\b', query, re.I):
                        known.append(name)
        except Exception:
            pass
        return known

    def _score_memory_relevance(
        self, query: str, memory_context: str, world_summary: str
    ) -> float:
        """Score how relevant the available context is to the query (0-1)."""
        if not memory_context and not world_summary:
            return 0.0

        # Extract key CONTENT words from query — exclude common stopwords
        # and also exclude personal pronouns/possessives that are always "relevant"
        stopwords = {
            "the", "and", "who", "what", "where", "when", "how",
            "does", "did", "can", "are", "was", "were", "been",
            "have", "has", "will", "would", "could", "should",
            "this", "that", "these", "those", "about", "with",
            "from", "your", "you", "for", "not", "but", "just",
            "my", "me", "i", "am", "is", "do", "tell",
        }
        query_words = set(
            w.lower().strip(".,?!'\"")
            for w in query.split()
            if len(w) > 2 and w.lower().strip(".,?!'\"") not in stopwords
        )

        if not query_words:
            return 0.0  # Only stopwords → can't assess relevance

        combined = (memory_context + " " + world_summary).lower()
        found = sum(1 for w in query_words if w in combined)
        return found / len(query_words)

    def _count_relevant_facts(self, memory_context: str) -> int:
        """Count actual factual statements in the context (not metadata/tags)."""
        if not memory_context:
            return 0

        count = 0
        for line in memory_context.split("\n"):
            line = line.strip()
            # Skip section headers and empty lines
            if not line or line.startswith("[") or line.startswith("<"):
                continue
            # Skip tag lines
            if line.startswith("- [") and "confidence]" in line:
                # This is a memory entry
                count += 1
            elif line.startswith("- "):
                count += 1
            elif "FACT" in line.upper():
                count += 1
        return count

    def _build_constraint(
        self,
        level: GroundingLevel,
        query_type: str,
        known_entities: list[str],
        query: str,
    ) -> str:
        """
        Build the constraint prompt to inject based on grounding analysis.
        This is the key anti-hallucination mechanism.
        """
        if level == GroundingLevel.GROUNDED:
            if query_type == "personal":
                return (
                    "IMPORTANT: Answer ONLY using the facts provided in the context above. "
                    "Do not add any information that is not explicitly stated in the context."
                )
            return ""

        if level == GroundingLevel.PARTIAL:
            if query_type == "personal":
                entity_note = ""
                if known_entities:
                    entity_note = f" You know about: {', '.join(known_entities)}."
                return (
                    f"IMPORTANT: You have SOME information about this topic.{entity_note} "
                    "Answer using ONLY what is provided in the context. "
                    "For anything not covered by the context, explicitly say "
                    "\"I don't have that information stored\" rather than guessing."
                )
            return (
                "If you're not confident in your answer, say so. "
                "Don't present uncertain information as fact."
            )

        # UNGROUNDED — this is where hallucination prevention matters most
        if query_type == "personal":
            return (
                "CRITICAL: You have NO stored information about this topic. "
                "Do NOT guess or invent an answer. "
                "Say something like: \"I don't have any information about that yet. "
                "You can tell me and I'll remember it for next time.\""
            )

        return (
            "If you don't know the answer with certainty, say so clearly. "
            "Don't invent facts."
        )

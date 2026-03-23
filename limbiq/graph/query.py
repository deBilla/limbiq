"""
Natural language query interface for the knowledge graph.

Given a user question, detect if it's a relationship query,
query the graph, and return a computed answer — WITHOUT using
the LLM for reasoning.
"""

import re

from limbiq.graph.store import GraphStore
from limbiq.graph.inference import InferenceEngine


class GraphQuery:
    """
    Detects relationship questions and answers them from the graph.

    Examples:
    - "Who is Upananda to my wife?" → graph lookup + inference
    - "What's my father's name?" → direct graph lookup
    - "Tell me about Prabhashi" → entity description
    """

    RELATIONSHIP_PATTERNS = [
        r"who\s+is\s+(\w+(?:\s+[A-Z]\w+)*)\s+to\s+(?:my\s+)?(\w+)",
        r"how\s+is\s+(\w+(?:\s+[A-Z]\w+)*)\s+related\s+to\s+(\w+(?:\s+[A-Z]\w+)*)",
        r"what\s+is\s+(\w+(?:\s+[A-Z]\w+)*)\s+to\s+(?:my\s+)?(\w+)",
        r"relationship\s+between\s+(\w+(?:\s+[A-Z]\w+)*)\s+and\s+(\w+(?:\s+[A-Z]\w+)*)",
    ]

    FACT_QUERY_PATTERNS = [
        r"(?:what(?:'s)?|who)\s+(?:is\s+)?my\s+(\w+)(?:'s\s+name)?",
    ]

    # Multi-hop: "my father's wife", "who is my sister's husband"
    MULTI_HOP_PATTERNS = [
        r"(?:what(?:'s)?|who)\s+(?:is\s+)?my\s+(\w+)(?:'s|s)\s+(\w+)",
        r"my\s+(\w+)(?:'s|s)\s+(\w+)",
    ]

    ENTITY_QUERY_PATTERNS = [
        r"(?:who|what)\s+is\s+(\w+(?:\s+[A-Z]\w+)*)",
        r"tell\s+me\s+about\s+(\w+(?:\s+[A-Z]\w+)*)",
    ]

    def __init__(self, graph: GraphStore, inference: InferenceEngine, user_name: str):
        self.graph = graph
        self.inference = inference
        self.user_name = user_name
        self._inference_dirty = True  # Run inference on first query

    def mark_dirty(self):
        """Mark inference cache as stale. Call after adding entities/relations."""
        self._inference_dirty = True

    def try_answer(self, question: str) -> dict:
        """
        Attempt to answer a question from the graph.

        Returns:
            {
                "answered": bool,
                "answer": str | None,
                "confidence": float,
                "source": "graph",
            }
        """
        q = question.lower().strip()

        # Ensure inferred relations are up to date (only when graph changed)
        if self._inference_dirty:
            self.inference.run_full_inference()
            self._inference_dirty = False

        # 1. Relationship queries: "who is X to Y"
        for pattern in self.RELATIONSHIP_PATTERNS:
            match = re.search(pattern, q)
            if match:
                entity_a = match.group(1)
                entity_b = self._resolve_reference(match.group(2))

                result = self.inference.query_relationship(entity_a, entity_b)
                if result["found"]:
                    descriptions = [r["description"] for r in result["relations"]]
                    return {
                        "answered": True,
                        "answer": " and ".join(descriptions),
                        "confidence": max(r["confidence"] for r in result["relations"]),
                        "source": "graph",
                    }

        # 2. Fact queries: "what's my father's name"
        for pattern in self.FACT_QUERY_PATTERNS:
            match = re.search(pattern, q)
            if match:
                relation = match.group(1)
                user = self.graph.find_entity_by_name(self.user_name)
                if user:
                    rows = self.graph.db.execute(
                        "SELECT object_id FROM relations WHERE subject_id=? AND predicate=?",
                        (user.id, relation),
                    ).fetchall()
                    if rows:
                        all_entities = {e.id: e for e in self.graph.get_all_entities()}
                        names = [all_entities[r[0]].name for r in rows if r[0] in all_entities]
                        if names:
                            return {
                                "answered": True,
                                "answer": f"Your {relation} is {', '.join(names)}.",
                                "confidence": 1.0,
                                "source": "graph",
                            }

        # 3. Multi-hop queries: "my father's wife", "my sister's husband"
        for pattern in self.MULTI_HOP_PATTERNS:
            match = re.search(pattern, q)
            if match:
                rel1 = match.group(1)  # e.g., "father"
                rel2 = match.group(2)  # e.g., "wife"
                user = self.graph.find_entity_by_name(self.user_name)
                if user:
                    # Hop 1: find target of first relation from user
                    hop1_rows = self.graph.db.execute(
                        "SELECT object_id FROM relations WHERE subject_id=? AND predicate=?",
                        (user.id, rel1),
                    ).fetchall()
                    if hop1_rows:
                        all_entities = {e.id: e for e in self.graph.get_all_entities()}
                        for hop1_row in hop1_rows:
                            hop1_entity_id = hop1_row[0]
                            hop1_name = all_entities.get(hop1_entity_id)
                            if not hop1_name:
                                continue
                            # Hop 2: find target of second relation from hop1 entity
                            # Check both explicit and inferred (include married_to for spouse queries)
                            spouse_preds = {rel2}
                            if rel2 in ("wife", "husband"):
                                spouse_preds.add("married_to")
                                spouse_preds.add("wife")
                                spouse_preds.add("husband")
                            placeholders = ",".join("?" * len(spouse_preds))
                            hop2_rows = self.graph.db.execute(
                                f"SELECT object_id FROM relations WHERE subject_id=? AND predicate IN ({placeholders})",
                                (hop1_entity_id, *spouse_preds),
                            ).fetchall()
                            if hop2_rows:
                                names = [all_entities[r[0]].name for r in hop2_rows
                                         if r[0] in all_entities and r[0] != user.id]
                                if names:
                                    return {
                                        "answered": True,
                                        "answer": f"Your {rel1}'s {rel2} is {', '.join(names)}.",
                                        "confidence": 0.9,
                                        "source": "graph",
                                    }

        # 4. Entity queries: "who is X" / "tell me about X"
        for pattern in self.ENTITY_QUERY_PATTERNS:
            match = re.search(pattern, q)
            if match:
                entity_name = match.group(1)
                # Skip common non-entity words
                if entity_name in ("the", "a", "this", "that", "your", "my", "it"):
                    continue
                description = self.inference.describe_entity(entity_name)
                if description:
                    return {
                        "answered": True,
                        "answer": description,
                        "confidence": 0.9,
                        "source": "graph",
                    }

        return {"answered": False, "answer": None, "confidence": 0, "source": None}

    def _resolve_reference(self, reference: str) -> str:
        """Resolve 'wife', 'father' etc to actual entity names."""
        user = self.graph.find_entity_by_name(self.user_name)
        if not user:
            return reference

        row = self.graph.db.execute(
            "SELECT object_id FROM relations WHERE subject_id=? AND predicate=?",
            (user.id, reference),
        ).fetchone()

        if row:
            entity = self.graph.db.execute(
                "SELECT name FROM entities WHERE id=?", (row[0],),
            ).fetchone()
            if entity:
                return entity[0]

        return reference

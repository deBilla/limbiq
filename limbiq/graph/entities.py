"""
Extract entities and relationships from text memories.

Two modes:
1. LLM-powered (if llm_fn provided) — accurate, handles complex sentences
2. Pattern-based fallback — fast, handles common patterns like "my X is Y"
"""

import re

from limbiq.graph.store import Entity, Relation, GraphStore


# Patterns for heuristic extraction.
# Each entry: (regex, extractor_fn) where extractor_fn returns
# a single (subject, predicate, object) tuple or a list of them.
RELATION_PATTERNS = [
    # "my father is Upananda" / "my wife is Prabhashi"
    (r"(?:my|our)\s+(father|mother|wife|husband|partner|brother|sister|son|daughter|"
     r"boss|manager|friend|colleague)\s+(?:is|was)\s+(\w+)",
     lambda m: ("user", m.group(1), m.group(2))),

    # "Upananda is my father"
    (r"(\w+)\s+is\s+(?:my|our)\s+(father|mother|wife|husband|partner|brother|sister|"
     r"son|daughter|boss|manager|friend|colleague)",
     lambda m: ("user", m.group(2), m.group(1))),

    # "my father's name is Upananda"
    (r"(?:my|our)\s+(father|mother|wife|husband|partner|brother|sister|son|daughter)"
     r"'?s?\s+name\s+is\s+(\w+)",
     lambda m: ("user", m.group(1), m.group(2))),

    # "User's father is Upananda" (compressed memory format)
    (r"(?:The\s+)?[Uu]ser'?s?\s+(father|mother|wife|husband|partner|brother|sister|"
     r"son|daughter|boss|manager|friend|colleague)\s+(?:is|was)\s+(?:named?\s+)?(\w+)",
     lambda m: ("user", m.group(1), m.group(2))),

    # "my wife's father is Chandrasiri" → (wife_entity, father, Chandrasiri)
    # Chained possessive: my <rel1>'s <rel2> is <name>
    (r"(?:my|our)\s+(father|mother|wife|husband|partner|brother|sister|son|daughter)"
     r"(?:'s|s)\s+(father|mother|wife|husband|partner|brother|sister|son|daughter)"
     r"\s+(?:is|was)\s+(\w+)",
     lambda m: (m.group(1), m.group(2), m.group(3))),

    # "my wife's father's name is Chandrasiri"
    (r"(?:my|our)\s+(father|mother|wife|husband|partner|brother|sister|son|daughter)"
     r"(?:'s|s)\s+(father|mother|wife|husband|partner|brother|sister|son|daughter)"
     r"'?s?\s+name\s+is\s+(\w+)",
     lambda m: (m.group(1), m.group(2), m.group(3))),

    # "I work at Bitsmedia" / "I work at Bitsmedia as a software engineer"
    (r"(?:I|[Uu]ser)\s+(?:work|works|worked)\s+(?:at|for)\s+(.+?)(?:\s+as\s+(?:a\s+)?(.+?))?[.\s,]?$",
     lambda m: [("user", "works_at", m.group(1).strip())] +
               ([("user", "role", m.group(2).strip())] if m.group(2) else [])),

    # "I live in Singapore"
    (r"(?:I|[Uu]ser)\s+(?:live|lives|lived|am based|stay|stays)\s+(?:in|at)\s+(.+?)[\s.,]?$",
     lambda m: ("user", "lives_in", m.group(1).strip())),
]


class EntityExtractor:
    def __init__(self, graph: GraphStore, user_name: str = "user", llm_fn=None):
        self.graph = graph
        self.user_name = user_name
        self.llm_fn = llm_fn

        # Ensure the user entity exists
        self.user_entity = graph.add_entity(Entity(
            name=user_name, entity_type="person",
        ))

    def extract_from_memory(self, memory_content: str, memory_id: str = "") -> dict:
        """
        Extract entities and relations from a memory string.
        Returns {"entities": [...], "relations": [...]}.
        """
        if self.llm_fn:
            return self._extract_llm(memory_content, memory_id)
        return self._extract_heuristic(memory_content, memory_id)

    # ── LLM extraction ────────────────────────────────────────

    def _extract_llm(self, text: str, memory_id: str) -> dict:
        prompt = (
            f"Extract entities and relationships from this text about a user "
            f"named {self.user_name}.\n\n"
            f'Text: "{text}"\n\n'
            "Output ONLY in this exact format (one per line, no other text):\n"
            "ENTITY: name | type\n"
            "RELATION: subject | predicate | object\n\n"
            "Entity types: person, company, place, project, concept\n"
            "Relation predicates: father, mother, wife, husband, brother, sister, "
            "son, daughter, works_at, lives_in, role, friend, colleague, interested_in\n\n"
            f"Example output:\n"
            f"ENTITY: Prabhashi | person\n"
            f"RELATION: {self.user_name} | wife | Prabhashi\n\n"
            "If no entities or relations can be extracted, respond: NONE"
        )

        result = self.llm_fn(prompt)
        if "NONE" in result.upper()[:20]:
            return {"entities": [], "relations": []}

        entities = []
        relations = []

        for line in result.strip().split("\n"):
            line = line.strip()

            if line.startswith("ENTITY:"):
                parts = line[7:].strip().split("|")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    etype = parts[1].strip()
                    entity = self.graph.add_entity(Entity(
                        name=name, entity_type=etype,
                        source_memory_id=memory_id,
                    ))
                    entities.append(entity)

            elif line.startswith("RELATION:"):
                parts = line[9:].strip().split("|")
                if len(parts) >= 3:
                    subj_name = parts[0].strip()
                    predicate = parts[1].strip()
                    obj_name = parts[2].strip()

                    subj = self.graph.find_entity_by_name(subj_name)
                    obj = self.graph.find_entity_by_name(obj_name)
                    if not subj:
                        subj = self.graph.add_entity(Entity(name=subj_name, entity_type="unknown"))
                    if not obj:
                        obj = self.graph.add_entity(Entity(name=obj_name, entity_type="unknown"))

                    rel = self.graph.add_relation(Relation(
                        subject_id=subj.id, predicate=predicate,
                        object_id=obj.id, source_memory_id=memory_id,
                    ))
                    relations.append(rel)

        return {"entities": entities, "relations": relations}

    # ── Heuristic extraction ──────────────────────────────────

    def _extract_heuristic(self, text: str, memory_id: str) -> dict:
        entities = []
        relations = []

        for pattern, extractor in RELATION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue

            result = extractor(match)
            if isinstance(result, tuple):
                result = [result]

            for subj_name, predicate, obj_name in result:
                if subj_name == "user":
                    subj_name = self.user_name

                # Resolve relation-as-subject (chained possessive):
                # "wife" → look up user's wife entity (e.g., Prabhashi)
                FAMILY_RELS = {"father", "mother", "wife", "husband", "partner",
                               "brother", "sister", "son", "daughter"}
                if subj_name.lower() in FAMILY_RELS:
                    user_entity = self.graph.find_entity_by_name(self.user_name)
                    if user_entity:
                        row = self.graph.db.execute(
                            "SELECT object_id FROM relations WHERE subject_id=? AND predicate=?",
                            (user_entity.id, subj_name.lower()),
                        ).fetchone()
                        if row:
                            resolved = self.graph.db.execute(
                                "SELECT name FROM entities WHERE id=?", (row[0],),
                            ).fetchone()
                            if resolved:
                                subj_name = resolved[0]

                subj = self.graph.find_entity_by_name(subj_name)
                if not subj:
                    subj = self.graph.add_entity(Entity(
                        name=subj_name, entity_type="person",
                        source_memory_id=memory_id,
                    ))
                    entities.append(subj)

                obj = self.graph.find_entity_by_name(obj_name)
                if not obj:
                    obj = self.graph.add_entity(Entity(
                        name=obj_name, source_memory_id=memory_id,
                    ))
                    entities.append(obj)

                rel = self.graph.add_relation(Relation(
                    subject_id=subj.id, predicate=predicate,
                    object_id=obj.id, source_memory_id=memory_id,
                ))
                relations.append(rel)

        return {"entities": entities, "relations": relations}

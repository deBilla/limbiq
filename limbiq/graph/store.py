"""
Knowledge graph stored in SQLite.

Lightweight — personal knowledge graphs are small (tens to hundreds of nodes).
No need for Neo4j or any graph database.

Thread-safe: uses the MemoryStore's per-thread connection via store.db property.
"""

import json
import logging
import uuid
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    entity_type: str = ""           # person, company, place, concept, project
    properties: dict = field(default_factory=dict)
    source_memory_id: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class Relation:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    subject_id: str = ""
    predicate: str = ""             # father, wife, works_at, lives_in, etc.
    object_id: str = ""
    confidence: float = 1.0
    is_inferred: bool = False
    source_memory_id: str = ""
    created_at: float = field(default_factory=time.time)


class GraphStore:
    """
    SQLite-backed knowledge graph.

    Uses the MemoryStore's thread-safe db property for all access,
    so it works correctly under Gradio's worker threads.
    """

    def __init__(self, memory_store, entity_state_store=None):
        """
        Args:
            memory_store: A MemoryStore instance — we use its .db property
                          for thread-safe per-thread SQLite connections.
            entity_state_store: Optional EntityStateStore — if provided,
                                auto-creates entity state on add_entity().
        """
        self._store = memory_store
        self._entity_state_store = entity_state_store
        self._init_tables()

    def heal(self):
        """Public self-healing entry point. Safe to call repeatedly.

        Detects junk entities (predicate words stored as entity names),
        resolves them from the graph or memories, re-points relations,
        and deletes the junk. Called after every observe() and at startup.
        """
        self._cleanup_junk_entities()

    @property
    def db(self):
        return self._store.db

    def _init_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT DEFAULT '',
                properties TEXT DEFAULT '{}',
                source_memory_id TEXT DEFAULT '',
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_id TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                is_inferred INTEGER DEFAULT 0,
                source_memory_id TEXT DEFAULT '',
                created_at REAL NOT NULL,
                FOREIGN KEY (subject_id) REFERENCES entities(id),
                FOREIGN KEY (object_id) REFERENCES entities(id)
            );

            CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_id);
            CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_id);
            CREATE INDEX IF NOT EXISTS idx_relations_predicate ON relations(predicate);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name COLLATE NOCASE);

            CREATE TABLE IF NOT EXISTS relation_corrections (
                id TEXT PRIMARY KEY,
                sentence TEXT NOT NULL,
                subject_name TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_name TEXT NOT NULL,
                is_positive INTEGER DEFAULT 1,
                created_at REAL NOT NULL
            );
        """)
        self.db.commit()
        self._cleanup_junk_entities()

    def _cleanup_junk_entities(self):
        """Self-healing: detect, resolve, and fix junk entities on every startup.

        For each entity whose name is a known predicate/relationship word
        (e.g., "Wife", "Dog", "Boss"):
        1. Try to resolve via graph (another relation to the real entity)
        2. If graph can't resolve, mine priority MEMORIES for the real name
        3. Re-point relations from junk → real entity (preserving graph data)
        4. Delete the junk entity

        This means past extraction mistakes are automatically repaired
        every time the store is opened — even if the only data source
        is in the memory store, not the graph.
        """
        try:
            from limbiq.graph.entities import VALID_PREDICATES, RELATION_ALIASES

            entities = self.db.execute("SELECT id, name FROM entities").fetchall()
            healed = 0
            deleted = 0

            for eid, name in entities:
                # Check static junk first (fast path)
                if self._is_junk_name(name):
                    self._delete_entity_and_relations(eid)
                    deleted += 1
                    continue

                # Check if name is a predicate/relationship word
                normalized = name.lower().strip().replace("-", "_").replace(" ", "_")
                alias_resolved = RELATION_ALIASES.get(normalized, normalized)
                if alias_resolved not in VALID_PREDICATES:
                    continue

                # Protect pet entities: if "Dog"/"Cat" is typed as "animal"
                # and has a "pet" relation from the user, it's a real entity
                pet_preds = {"dog", "cat", "pet"}
                if alias_resolved in pet_preds:
                    etype_row = self.db.execute(
                        "SELECT entity_type FROM entities WHERE id=?", (eid,)
                    ).fetchone()
                    if etype_row and etype_row[0] == "animal":
                        continue  # Keep it — it's a real pet entity

                # This entity name IS a relationship word (e.g., "Wife").
                # Try to find the real entity it should point to.
                real_entity_id = self._resolve_from_graph(eid, alias_resolved)

                # If graph can't resolve, try mining memories
                if not real_entity_id:
                    real_entity_id = self._resolve_from_memories(eid, name, alias_resolved)

                if real_entity_id:
                    self._repoint_relations(eid, real_entity_id)
                    healed += 1

                # Delete the junk entity either way
                self._delete_entity_and_relations(eid)
                deleted += 1

            if healed or deleted:
                self.db.commit()
                logger.info(f"Graph self-heal: {healed} re-pointed, {deleted} junk entities removed")
        except Exception as e:
            logger.warning(f"Failed to cleanup junk entities: {e}")

    def _get_user_entity_id(self) -> Optional[str]:
        """Find the primary user entity (person with most outgoing explicit relations)."""
        row = self.db.execute(
            "SELECT e.id FROM entities e "
            "JOIN relations r ON r.subject_id = e.id "
            "WHERE e.entity_type = 'person' AND r.is_inferred = 0 "
            "GROUP BY e.id ORDER BY COUNT(r.id) DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def _resolve_from_graph(self, junk_id: str, predicate: str) -> Optional[str]:
        """Try to find the real entity via existing graph relations.

        Looks for: UserEntity --predicate--> RealEntity (where RealEntity != junk).
        """
        user_id = self._get_user_entity_id()
        if not user_id:
            return None
        row = self.db.execute(
            "SELECT object_id FROM relations WHERE subject_id=? AND predicate=?",
            (user_id, predicate),
        ).fetchone()
        if row and row[0] != junk_id:
            return row[0]
        return None

    def _resolve_from_memories(self, junk_id: str, junk_name: str, predicate: str) -> Optional[str]:
        """Mine priority memories for the real entity name.

        Searches memories for patterns like:
          "wife is Prabhashi", "wife's name is Prabhashi",
          "Prabhashi is my wife", "married to Prabhashi"

        If found, creates the entity in the graph and returns its ID.
        """
        import re

        relationship = junk_name.lower()
        # Search priority + mid-tier memories for relationship mentions
        rows = self.db.execute(
            "SELECT content FROM memories "
            "WHERE is_suppressed = 0 AND (is_priority = 1 OR tier IN ('priority', 'mid')) "
            "ORDER BY is_priority DESC, created_at DESC"
        ).fetchall()

        # Patterns that extract a proper name from relationship context
        patterns = [
            # "wife is Prabhashi" / "wife's name is Prabhashi"
            re.compile(
                rf"\b{re.escape(relationship)}(?:'s)?\s+(?:name\s+)?is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                re.I,
            ),
            # "Prabhashi is my/his/the wife"
            re.compile(
                rf"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+is\s+(?:my|his|her|the|user'?s?)\s+{re.escape(relationship)}",
                re.I,
            ),
            # "married to Prabhashi" (for wife/husband)
            re.compile(
                r"married\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                re.I,
            ) if relationship in ("wife", "husband", "partner") else None,
            # "my wife Prabhashi" / "his dog Max"
            re.compile(
                rf"(?:my|his|her|user'?s?)\s+{re.escape(relationship)}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                re.I,
            ),
        ]
        patterns = [p for p in patterns if p is not None]

        for (content,) in rows:
            for pattern in patterns:
                match = pattern.search(content)
                if match:
                    real_name = match.group(1).strip()
                    # Validate: the extracted name shouldn't itself be a junk name
                    if self._is_junk_name(real_name) or len(real_name) < 2:
                        continue

                    logger.info(f"Graph self-heal: resolved '{junk_name}' → '{real_name}' from memory")

                    # Find or create the real entity
                    existing = self.find_entity_by_name(real_name)
                    if existing:
                        return existing.id

                    # Create the entity
                    entity = self.add_entity(Entity(
                        name=real_name, entity_type="person",
                    ))
                    if entity:
                        # Also create the correct relation: user --predicate--> real_entity
                        user_id = self._get_user_entity_id()
                        if user_id:
                            self.add_relation(Relation(
                                subject_id=user_id, predicate=predicate,
                                object_id=entity.id,
                            ))
                        return entity.id
        return None

    def _repoint_relations(self, old_id: str, new_id: str):
        """Re-point all relations from old entity to new entity."""
        self.db.execute(
            "UPDATE relations SET subject_id=? WHERE subject_id=?",
            (new_id, old_id),
        )
        self.db.execute(
            "UPDATE relations SET object_id=? WHERE object_id=?",
            (new_id, old_id),
        )

    def _delete_entity_and_relations(self, eid: str):
        """Delete an entity and all its relations."""
        self.db.execute("DELETE FROM relations WHERE subject_id=? OR object_id=?", (eid, eid))
        self.db.execute("DELETE FROM entities WHERE id=?", (eid,))
        self.db.execute("DELETE FROM entities WHERE id=?", (eid,))

    # ── Entity operations ─────────────────────────────────────

    # Entity names that should never be stored — typically LLM extraction artifacts.
    # NOTE: "user" is NOT in this list — EntityExtractor maps "user" → real
    # user_name (e.g. "Dimuthu") at extraction time (entities.py:448-449).
    # Blocking "user" here silently kills ALL user-centric relations.
    _JUNK_NAMES = {
        "none", "null", "n/a", "unknown", "undefined", "default", "",
        "?", "topic", "the user", "assistant", "ai", "bot",
        "feeding schedule care", "well-being", "feeding schedule",
        # Pronouns and common words that slip through as entities
        "he", "she", "it", "they", "we", "you", "i", "me", "him", "her",
        "my", "his", "our", "your", "their", "us", "them",
        "if", "but", "and", "or", "so", "yet", "nor",
        "based", "however", "therefore", "also", "just", "really",
        "actually", "yes", "no", "ok", "sure", "well",
        "today", "tomorrow", "yesterday", "now", "then", "here", "there",
        "everything", "something", "nothing", "anything",
        "everyone", "someone", "anyone",
    }

    # Reject entities with names shorter than 2 chars or that look like numbers/dates
    @staticmethod
    def _is_junk_name(name: str) -> bool:
        import re
        stripped = name.strip()
        # Strip possessives
        if stripped.endswith(("\u2019s", "'s")):
            stripped = stripped[:-2].strip()
        # Strip parenthetical content: "User (Dimuthu)" → "User"
        stripped = re.sub(r'\s*\([^)]*\)\s*$', '', stripped).strip()
        if stripped.lower() in GraphStore._JUNK_NAMES:
            return True
        if len(stripped) < 2:
            return True
        # Reject pure numbers, dates, time expressions
        if re.match(r'^[\d\s.,:/-]+$', stripped):
            return True
        if re.match(r'^\d+\s+(days?|weeks?|months?|years?|hours?|minutes?)\s+ago$', stripped, re.I):
            return True
        # Reject multi-word names starting with sentence particles
        words = stripped.split()
        if len(words) >= 2:
            _STARTERS = {"if", "no", "yes", "ok", "so", "and", "but", "or", "thank",
                         "thanks", "please", "hi", "hello", "hey", "not", "nor", "yet"}
            if words[0].lower() in _STARTERS:
                return True
        return False

    @staticmethod
    def _name_similarity(a: str, b: str) -> float:
        """Simple character-level similarity for fuzzy entity matching.
        Uses Levenshtein-like ratio: 1.0 = identical, 0.0 = no overlap."""
        a, b = a.lower().strip(), b.lower().strip()
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0
        # Character bigram overlap (Dice coefficient)
        def bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1)) if len(s) > 1 else {s}
        ba, bb = bigrams(a), bigrams(b)
        if not ba or not bb:
            return 0.0
        return 2.0 * len(ba & bb) / (len(ba) + len(bb))

    def add_entity(self, entity: Entity) -> Optional[Entity]:
        """Add entity. If name already exists (case-insensitive or fuzzy match),
        return existing. Returns None if entity name is rejected as junk."""
        # Reject junk names from LLM extraction
        if self._is_junk_name(entity.name):
            return None

        # Exact match first
        existing = self.find_entity_by_name(entity.name)
        if existing:
            return existing

        # Fuzzy match: catch typos like "Prabhasi" ≈ "Prabhashi"
        # Only for short names (< 20 chars) to avoid false positives
        if len(entity.name) < 20:
            all_entities = self.get_all_entities()
            for e in all_entities:
                sim = self._name_similarity(entity.name, e.name)
                if sim >= 0.75 and e.entity_type == entity.entity_type:
                    logger.info(
                        f"Fuzzy entity match: '{entity.name}' ≈ '{e.name}' "
                        f"(sim={sim:.2f}), merging"
                    )
                    return e

        self.db.execute(
            "INSERT INTO entities (id, name, entity_type, properties, source_memory_id, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (entity.id, entity.name, entity.entity_type, json.dumps(entity.properties),
             entity.source_memory_id, entity.created_at),
        )
        self.db.commit()

        # Auto-create entity state (distributed cellular memory)
        if self._entity_state_store is not None:
            try:
                self._entity_state_store.ensure_state_exists(entity.id)
            except Exception as e:
                logger.warning(f"Failed to create entity state for {entity.name}: {e}")

        return entity

    def find_entity_by_name(self, name: str) -> Optional[Entity]:
        """Case-insensitive entity lookup."""
        row = self.db.execute(
            "SELECT * FROM entities WHERE name = ? COLLATE NOCASE", (name,)
        ).fetchone()
        if row:
            return self._row_to_entity(row)
        return None

    def get_all_entities(self) -> list[Entity]:
        rows = self.db.execute(
            "SELECT * FROM entities ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    # ── Relation operations ───────────────────────────────────

    def add_relation(self, relation: Relation) -> Relation:
        """Add relation. Skip if identical relation already exists."""
        existing = self.db.execute(
            "SELECT id FROM relations WHERE subject_id=? AND predicate=? AND object_id=? AND is_inferred=?",
            (relation.subject_id, relation.predicate, relation.object_id, int(relation.is_inferred)),
        ).fetchone()
        if existing:
            return relation

        self.db.execute(
            "INSERT INTO relations (id, subject_id, predicate, object_id, confidence, "
            "is_inferred, source_memory_id, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (relation.id, relation.subject_id, relation.predicate, relation.object_id,
             relation.confidence, int(relation.is_inferred), relation.source_memory_id,
             relation.created_at),
        )
        self.db.commit()
        return relation

    def get_relations_for(self, entity_id: str) -> list[Relation]:
        """Get all relations where this entity is subject OR object."""
        rows = self.db.execute(
            "SELECT * FROM relations WHERE subject_id=? OR object_id=?",
            (entity_id, entity_id),
        ).fetchall()
        return [self._row_to_relation(r) for r in rows]

    def get_all_relations(self, include_inferred: bool = True) -> list[Relation]:
        query = "SELECT * FROM relations"
        if not include_inferred:
            query += " WHERE is_inferred = 0"
        rows = self.db.execute(query).fetchall()
        return [self._row_to_relation(r) for r in rows]

    def remove_inferred(self):
        """Clear all inferred relations (before re-running inference)."""
        self.db.execute("DELETE FROM relations WHERE is_inferred = 1")
        self.db.commit()

    def suppress_relations_for_memory(self, memory_id: str):
        """When GABA suppresses a memory, remove its graph relations too."""
        self.db.execute(
            "DELETE FROM relations WHERE source_memory_id = ?", (memory_id,)
        )
        self.db.commit()

    def delete_relation(self, subject_name: str, predicate: str, object_name: str):
        """Delete a specific relation by entity names and predicate."""
        subj = self.find_entity_by_name(subject_name)
        obj = self.find_entity_by_name(object_name)
        if subj and obj:
            self.db.execute(
                "DELETE FROM relations WHERE subject_id=? AND predicate=? AND object_id=?",
                (subj.id, predicate, obj.id),
            )
            self.db.commit()

    def delete_relations_between(self, name_a: str, name_b: str):
        """Delete ALL relations between two entities (both directions)."""
        ent_a = self.find_entity_by_name(name_a)
        ent_b = self.find_entity_by_name(name_b)
        if ent_a and ent_b:
            self.db.execute(
                "DELETE FROM relations WHERE "
                "(subject_id=? AND object_id=?) OR (subject_id=? AND object_id=?)",
                (ent_a.id, ent_b.id, ent_b.id, ent_a.id),
            )
            self.db.commit()

    # ── Relation corrections (training feedback) ────────────

    def store_relation_correction(
        self, sentence: str, subject_name: str, predicate: str,
        object_name: str, is_positive: bool = True,
    ) -> None:
        """Store a relation correction as a training pair.

        Args:
            sentence: The original sentence text.
            subject_name: Subject entity name.
            predicate: The relation predicate.
            object_name: Object entity name.
            is_positive: True = correct relation, False = incorrect relation.
        """
        self.db.execute(
            "INSERT INTO relation_corrections "
            "(id, sentence, subject_name, predicate, object_name, is_positive, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4())[:12], sentence, subject_name, predicate,
             object_name, int(is_positive), time.time()),
        )
        self.db.commit()

    def get_relation_corrections(self) -> list[dict]:
        """Get all stored relation corrections for training."""
        rows = self.db.execute(
            "SELECT sentence, subject_name, predicate, object_name, is_positive "
            "FROM relation_corrections ORDER BY created_at"
        ).fetchall()
        return [
            {
                "sentence": r[0], "subject_name": r[1], "predicate": r[2],
                "object_name": r[3], "is_positive": bool(r[4]),
            }
            for r in rows
        ]

    def count_corrections_since(self, since_timestamp: float) -> int:
        """Count corrections stored since a given timestamp."""
        row = self.db.execute(
            "SELECT COUNT(*) FROM relation_corrections WHERE created_at > ?",
            (since_timestamp,),
        ).fetchone()
        return row[0] if row else 0

    # ── Stats ─────────────────────────────────────────────────

    def get_stats(self) -> dict:
        entities = self.db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        relations = self.db.execute(
            "SELECT COUNT(*) FROM relations WHERE is_inferred=0"
        ).fetchone()[0]
        inferred = self.db.execute(
            "SELECT COUNT(*) FROM relations WHERE is_inferred=1"
        ).fetchone()[0]
        return {"entities": entities, "relations": relations, "inferred": inferred}

    # ── Internal ──────────────────────────────────────────────

    def _row_to_entity(self, row) -> Entity:
        return Entity(
            id=row[0], name=row[1], entity_type=row[2],
            properties=json.loads(row[3]), source_memory_id=row[4],
            created_at=row[5],
        )

    def _row_to_relation(self, row) -> Relation:
        return Relation(
            id=row[0], subject_id=row[1], predicate=row[2],
            object_id=row[3], confidence=row[4], is_inferred=bool(row[5]),
            source_memory_id=row[6], created_at=row[7],
        )

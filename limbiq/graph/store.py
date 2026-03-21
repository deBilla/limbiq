"""
Knowledge graph stored in SQLite.

Lightweight — personal knowledge graphs are small (tens to hundreds of nodes).
No need for Neo4j or any graph database.

Thread-safe: uses the MemoryStore's per-thread connection via store.db property.
"""

import json
import uuid
import time
from dataclasses import dataclass, field
from typing import Optional


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

    def __init__(self, memory_store):
        """
        Args:
            memory_store: A MemoryStore instance — we use its .db property
                          for thread-safe per-thread SQLite connections.
        """
        self._store = memory_store
        self._init_tables()

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
        """)
        self.db.commit()
        self._cleanup_junk_entities()

    def _cleanup_junk_entities(self):
        """Remove junk entities and their relations on startup."""
        try:
            entities = self.db.execute("SELECT id, name FROM entities").fetchall()
            junk_ids = [eid for eid, name in entities if self._is_junk_name(name)]
            if junk_ids:
                placeholders = ",".join("?" * len(junk_ids))
                self.db.execute(f"DELETE FROM relations WHERE subject_id IN ({placeholders})", junk_ids)
                self.db.execute(f"DELETE FROM relations WHERE object_id IN ({placeholders})", junk_ids)
                self.db.execute(f"DELETE FROM entities WHERE id IN ({placeholders})", junk_ids)
                self.db.commit()
        except Exception:
            pass

    # ── Entity operations ─────────────────────────────────────

    # Entity names that should never be stored — typically LLM extraction artifacts
    _JUNK_NAMES = {"none", "null", "n/a", "unknown", "undefined", "default", "",
                   "?", "topic", "user", "the user", "assistant", "ai",
                   "wife", "husband", "father", "mother", "brother", "sister",
                   "son", "daughter", "dog", "cat", "pet", "boss", "friend",
                   "feeding schedule care", "well-being", "feeding schedule"}

    # Reject entities with names shorter than 2 chars or that look like numbers/dates
    @staticmethod
    def _is_junk_name(name: str) -> bool:
        stripped = name.strip()
        if stripped.lower() in GraphStore._JUNK_NAMES:
            return True
        if len(stripped) < 2:
            return True
        # Reject pure numbers, dates, time expressions
        import re
        if re.match(r'^[\d\s.,:/-]+$', stripped):
            return True
        if re.match(r'^\d+\s+(days?|weeks?|months?|years?|hours?|minutes?)\s+ago$', stripped, re.I):
            return True
        return False

    def add_entity(self, entity: Entity) -> Optional[Entity]:
        """Add entity. If name already exists (case-insensitive), return existing.
        Returns None if entity name is rejected as junk."""
        # Reject junk names from LLM extraction
        if self._is_junk_name(entity.name):
            return None

        existing = self.find_entity_by_name(entity.name)
        if existing:
            return existing

        self.db.execute(
            "INSERT INTO entities (id, name, entity_type, properties, source_memory_id, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (entity.id, entity.name, entity.entity_type, json.dumps(entity.properties),
             entity.source_memory_id, entity.created_at),
        )
        self.db.commit()
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

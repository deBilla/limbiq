"""Behavioral rules store for Serotonin signal.

Thread-safe: uses MemoryStore's per-thread db property.
"""

import json
import uuid
import time

from limbiq.types import BehavioralRule


class RuleStore:
    def __init__(self, memory_store):
        self._store = memory_store
        self._init_tables()

    def _init_tables(self):
        self._store.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS pattern_observations (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                pattern_key TEXT NOT NULL,
                observation TEXT NOT NULL,
                session_id TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS behavioral_rules (
                id TEXT PRIMARY KEY,
                pattern_key TEXT NOT NULL UNIQUE,
                rule_text TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                observation_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                is_active INTEGER DEFAULT 1
            );
            """
        )
        self._store.db.commit()

    def add_observation(
        self, category: str, pattern_key: str, observation: str, session_id: str
    ) -> None:
        db = self._store.db
        db.execute(
            "INSERT INTO pattern_observations (id, category, pattern_key, observation, session_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), category, pattern_key, observation, session_id, time.time()),
        )
        db.commit()

    def get_observation_count(self, pattern_key: str) -> int:
        db = self._store.db
        cursor = db.execute(
            "SELECT COUNT(*) FROM pattern_observations WHERE pattern_key = ?",
            (pattern_key,),
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def get_distinct_session_count(self, pattern_key: str) -> int:
        db = self._store.db
        cursor = db.execute(
            "SELECT COUNT(DISTINCT session_id) FROM pattern_observations WHERE pattern_key = ?",
            (pattern_key,),
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def get_observations(self, pattern_key: str) -> list[dict]:
        db = self._store.db
        cursor = db.execute(
            "SELECT category, pattern_key, observation, session_id, created_at "
            "FROM pattern_observations WHERE pattern_key = ? ORDER BY created_at DESC",
            (pattern_key,),
        )
        return [
            {
                "category": row[0],
                "pattern_key": row[1],
                "observation": row[2],
                "session_id": row[3],
                "created_at": row[4],
            }
            for row in cursor.fetchall()
        ]

    def rule_exists(self, pattern_key: str) -> bool:
        db = self._store.db
        cursor = db.execute(
            "SELECT COUNT(*) FROM behavioral_rules WHERE pattern_key = ?",
            (pattern_key,),
        )
        row = cursor.fetchone()
        return (row[0] if row else 0) > 0

    def create_rule(
        self, pattern_key: str, rule_text: str, observation_count: int
    ) -> BehavioralRule:
        db = self._store.db
        rule_id = str(uuid.uuid4())
        now = time.time()
        db.execute(
            "INSERT OR REPLACE INTO behavioral_rules "
            "(id, pattern_key, rule_text, confidence, observation_count, created_at, is_active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rule_id, pattern_key, rule_text, 1.0, observation_count, now, 1),
        )
        db.commit()
        return BehavioralRule(
            id=rule_id,
            pattern_key=pattern_key,
            rule_text=rule_text,
            confidence=1.0,
            observation_count=observation_count,
            created_at=now,
            is_active=True,
        )

    def get_active_rules(self) -> list[BehavioralRule]:
        db = self._store.db
        cursor = db.execute(
            "SELECT id, pattern_key, rule_text, confidence, observation_count, created_at, is_active "
            "FROM behavioral_rules WHERE is_active = 1"
        )
        return [
            BehavioralRule(
                id=row[0],
                pattern_key=row[1],
                rule_text=row[2],
                confidence=row[3],
                observation_count=row[4],
                created_at=row[5],
                is_active=bool(row[6]),
            )
            for row in cursor.fetchall()
        ]

    def deactivate_rule(self, rule_id: str) -> None:
        db = self._store.db
        db.execute("UPDATE behavioral_rules SET is_active = 0 WHERE id = ?", (rule_id,))
        db.commit()

    def reactivate_rule(self, rule_id: str) -> None:
        db = self._store.db
        db.execute("UPDATE behavioral_rules SET is_active = 1 WHERE id = ?", (rule_id,))
        db.commit()

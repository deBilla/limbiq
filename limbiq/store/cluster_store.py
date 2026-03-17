"""Knowledge cluster store for Acetylcholine signal.

Thread-safe: uses MemoryStore's per-thread db property.
"""

import json
import uuid
import time

from limbiq.types import KnowledgeCluster, Memory


class ClusterStore:
    MAX_CLUSTER_SIZE = 15

    def __init__(self, memory_store):
        self._store = memory_store
        self._init_tables()

    def _init_tables(self):
        self._store.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_clusters (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                description TEXT,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                memory_ids TEXT DEFAULT '[]'
            );
            """
        )
        self._store.db.commit()

    def get_by_topic(self, topic: str) -> KnowledgeCluster | None:
        db = self._store.db
        cursor = db.execute(
            "SELECT id, topic, description, created_at, last_accessed, access_count, memory_ids "
            "FROM knowledge_clusters WHERE LOWER(topic) = LOWER(?)",
            (topic,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_cluster(row)

    def find_matching_cluster(self, topic: str) -> KnowledgeCluster | None:
        """Find a cluster whose topic is contained in or contains the given topic."""
        db = self._store.db
        cursor = db.execute(
            "SELECT id, topic, description, created_at, last_accessed, access_count, memory_ids "
            "FROM knowledge_clusters"
        )
        topic_lower = topic.lower()
        for row in cursor.fetchall():
            cluster_topic = row[1].lower()
            if cluster_topic in topic_lower or topic_lower in cluster_topic:
                return self._row_to_cluster(row)
        return None

    def create_cluster(self, topic: str, description: str = "") -> KnowledgeCluster:
        db = self._store.db
        cluster_id = str(uuid.uuid4())
        now = time.time()
        db.execute(
            "INSERT INTO knowledge_clusters (id, topic, description, created_at, last_accessed, access_count, memory_ids) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cluster_id, topic, description, now, now, 0, "[]"),
        )
        db.commit()
        return KnowledgeCluster(
            id=cluster_id, topic=topic, description=description,
            created_at=now, last_accessed=now,
        )

    def add_memory_to_cluster(self, cluster_id: str, memory_id: str) -> None:
        db = self._store.db
        cursor = db.execute(
            "SELECT memory_ids FROM knowledge_clusters WHERE id = ?", (cluster_id,)
        )
        row = cursor.fetchone()
        if not row:
            return
        ids = json.loads(row[0])
        if memory_id not in ids:
            if len(ids) >= self.MAX_CLUSTER_SIZE:
                ids = ids[1:]  # Drop oldest
            ids.append(memory_id)
            db.execute(
                "UPDATE knowledge_clusters SET memory_ids = ? WHERE id = ?",
                (json.dumps(ids), cluster_id),
            )
            db.commit()

    def touch_cluster(self, cluster_id: str) -> None:
        db = self._store.db
        db.execute(
            "UPDATE knowledge_clusters SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
            (time.time(), cluster_id),
        )
        db.commit()

    def get_cluster_memories(self, cluster_id: str) -> list[Memory]:
        db = self._store.db
        cursor = db.execute(
            "SELECT memory_ids FROM knowledge_clusters WHERE id = ?", (cluster_id,)
        )
        row = cursor.fetchone()
        if not row:
            return []
        ids = json.loads(row[0])
        if not ids:
            return []

        placeholders = ",".join("?" for _ in ids)
        cursor = db.execute(
            f"SELECT id, content, tier, confidence, created_at, session_count, "
            f"access_count, is_priority, is_suppressed, suppression_reason, "
            f"source, metadata, embedding FROM memories "
            f"WHERE id IN ({placeholders}) AND is_suppressed = 0",
            ids,
        )
        return [self._store._row_to_memory(r) for r in cursor.fetchall()]

    def get_all_clusters(self) -> list[KnowledgeCluster]:
        db = self._store.db
        cursor = db.execute(
            "SELECT id, topic, description, created_at, last_accessed, access_count, memory_ids "
            "FROM knowledge_clusters ORDER BY last_accessed DESC"
        )
        return [self._row_to_cluster(row) for row in cursor.fetchall()]

    def _row_to_cluster(self, row) -> KnowledgeCluster:
        return KnowledgeCluster(
            id=row[0],
            topic=row[1],
            description=row[2] or "",
            created_at=row[3],
            last_accessed=row[4],
            access_count=row[5],
            memory_ids=json.loads(row[6]) if row[6] else [],
        )

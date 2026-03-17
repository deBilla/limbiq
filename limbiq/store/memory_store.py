"""SQLite-backed memory store with tiers."""

import json
import os
import sqlite3
import uuid
import time

from limbiq.types import Memory, MemoryTier, SuppressionReason


def _serialize_embedding(embedding: list[float] | None) -> bytes | None:
    if embedding is None:
        return None
    import struct
    return struct.pack(f"{len(embedding)}f", *embedding)


def _deserialize_embedding(data: bytes | None) -> list[float] | None:
    if data is None:
        return None
    import struct
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


class MemoryStore:
    def __init__(self, store_path: str, user_id: str):
        self.store_path = store_path
        self.user_id = user_id

        os.makedirs(store_path, exist_ok=True)
        db_path = os.path.join(store_path, f"{user_id}.db")
        self.db = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'short',
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL,
                session_count INTEGER DEFAULT 0,
                access_count INTEGER DEFAULT 0,
                is_priority INTEGER DEFAULT 0,
                is_suppressed INTEGER DEFAULT 0,
                suppression_reason TEXT,
                source TEXT DEFAULT 'conversation',
                metadata TEXT DEFAULT '{}',
                embedding BLOB
            );

            CREATE TABLE IF NOT EXISTS signal_log (
                id TEXT PRIMARY KEY,
                signal_type TEXT NOT NULL,
                trigger TEXT NOT NULL,
                timestamp REAL NOT NULL,
                details TEXT DEFAULT '{}',
                memory_ids_affected TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                messages TEXT NOT NULL,
                created_at REAL NOT NULL,
                compressed INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                started_at REAL NOT NULL,
                ended_at REAL,
                turns INTEGER DEFAULT 0,
                signals_fired INTEGER DEFAULT 0
            );
            """
        )
        self.db.commit()

    def store(
        self,
        content: str,
        tier: MemoryTier = MemoryTier.SHORT,
        confidence: float = 1.0,
        is_priority: bool = False,
        source: str = "conversation",
        metadata: dict = None,
        embedding: list[float] = None,
    ) -> Memory:
        memory_id = str(uuid.uuid4())
        now = time.time()
        meta_str = json.dumps(metadata or {})
        emb_bytes = _serialize_embedding(embedding)

        self.db.execute(
            """INSERT INTO memories
               (id, content, tier, confidence, created_at, is_priority, source, metadata, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory_id,
                content,
                tier.value if isinstance(tier, MemoryTier) else tier,
                confidence,
                now,
                1 if is_priority else 0,
                source,
                meta_str,
                emb_bytes,
            ),
        )
        self.db.commit()

        return Memory(
            id=memory_id,
            content=content,
            tier=tier,
            confidence=confidence,
            created_at=now,
            is_priority=is_priority,
            source=source,
            metadata=metadata or {},
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        include_suppressed: bool = False,
    ) -> list[Memory]:
        """Search memories by embedding similarity."""
        clause = "" if include_suppressed else "WHERE is_suppressed = 0"
        cursor = self.db.execute(
            f"SELECT id, content, tier, confidence, created_at, session_count, "
            f"access_count, is_priority, is_suppressed, suppression_reason, "
            f"source, metadata, embedding FROM memories {clause}"
        )

        from limbiq.store.embeddings import EmbeddingEngine

        scored = []
        for row in cursor.fetchall():
            emb = _deserialize_embedding(row[12])
            if emb is None:
                continue
            # Inline cosine similarity for performance
            dot = sum(a * b for a, b in zip(query_embedding, emb))
            norm_a = sum(a * a for a in query_embedding) ** 0.5
            norm_b = sum(b * b for b in emb) ** 0.5
            if norm_a == 0 or norm_b == 0:
                sim = 0.0
            else:
                sim = dot / (norm_a * norm_b)

            mem = self._row_to_memory(row)
            scored.append((sim, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def get_priority_memories(self) -> list[Memory]:
        cursor = self.db.execute(
            "SELECT id, content, tier, confidence, created_at, session_count, "
            "access_count, is_priority, is_suppressed, suppression_reason, "
            "source, metadata, embedding FROM memories "
            "WHERE is_priority = 1 AND is_suppressed = 0"
        )
        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def get_suppressed(self) -> list[Memory]:
        cursor = self.db.execute(
            "SELECT id, content, tier, confidence, created_at, session_count, "
            "access_count, is_priority, is_suppressed, suppression_reason, "
            "source, metadata, embedding FROM memories WHERE is_suppressed = 1"
        )
        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def suppress(self, memory_id: str, reason: SuppressionReason | str) -> None:
        reason_val = reason.value if isinstance(reason, SuppressionReason) else reason
        self.db.execute(
            "UPDATE memories SET is_suppressed = 1, confidence = 0.05, "
            "suppression_reason = ? WHERE id = ?",
            (reason_val, memory_id),
        )
        self.db.commit()

    def restore(self, memory_id: str) -> None:
        self.db.execute(
            "UPDATE memories SET is_suppressed = 0, confidence = 0.8, "
            "suppression_reason = NULL WHERE id = ?",
            (memory_id,),
        )
        self.db.commit()

    def age_all(self) -> None:
        self.db.execute("UPDATE memories SET session_count = session_count + 1")
        self.db.commit()

    def increment_access(self, memory_id: str) -> None:
        self.db.execute(
            "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
            (memory_id,),
        )
        self.db.commit()

    def boost_confidence(self, memory_id: str, new_confidence: float) -> None:
        self.db.execute(
            "UPDATE memories SET confidence = ? WHERE id = ?",
            (new_confidence, memory_id),
        )
        self.db.commit()

    def get_stale(self, min_sessions: int = 10) -> list[Memory]:
        cursor = self.db.execute(
            "SELECT id, content, tier, confidence, created_at, session_count, "
            "access_count, is_priority, is_suppressed, suppression_reason, "
            "source, metadata, embedding FROM memories "
            "WHERE session_count >= ? AND access_count = 0 "
            "AND is_suppressed = 0 AND is_priority = 0",
            (min_sessions,),
        )
        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def delete_old_suppressed(self, min_sessions: int = 30) -> int:
        cursor = self.db.execute(
            "DELETE FROM memories WHERE is_suppressed = 1 AND session_count >= ?",
            (min_sessions,),
        )
        self.db.commit()
        return cursor.rowcount

    def store_conversation(self, messages: list[dict], session_id: str = None) -> None:
        self.db.execute(
            "INSERT INTO conversations (id, session_id, messages, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                session_id or "default",
                json.dumps(messages),
                time.time(),
            ),
        )
        self.db.commit()

    def get_stats(self) -> dict:
        stats = {}
        for tier in MemoryTier:
            cursor = self.db.execute(
                "SELECT COUNT(*) FROM memories WHERE tier = ? AND is_suppressed = 0",
                (tier.value,),
            )
            stats[tier.value] = cursor.fetchone()[0]

        cursor = self.db.execute(
            "SELECT COUNT(*) FROM memories WHERE is_suppressed = 1"
        )
        stats["suppressed"] = cursor.fetchone()[0]

        cursor = self.db.execute(
            "SELECT COUNT(*) FROM memories WHERE is_priority = 1 AND is_suppressed = 0"
        )
        stats["priority"] = cursor.fetchone()[0]

        cursor = self.db.execute("SELECT COUNT(*) FROM memories")
        stats["total"] = cursor.fetchone()[0]

        return stats

    def export_all(self) -> dict:
        cursor = self.db.execute(
            "SELECT id, content, tier, confidence, created_at, session_count, "
            "access_count, is_priority, is_suppressed, suppression_reason, "
            "source, metadata, embedding FROM memories"
        )
        memories = []
        for row in cursor.fetchall():
            mem = self._row_to_memory(row)
            memories.append(
                {
                    "id": mem.id,
                    "content": mem.content,
                    "tier": mem.tier if isinstance(mem.tier, str) else mem.tier.value,
                    "confidence": mem.confidence,
                    "created_at": mem.created_at,
                    "session_count": mem.session_count,
                    "access_count": mem.access_count,
                    "is_priority": mem.is_priority,
                    "is_suppressed": mem.is_suppressed,
                    "suppression_reason": mem.suppression_reason,
                    "source": mem.source,
                    "metadata": mem.metadata,
                }
            )

        return {"user_id": self.user_id, "memories": memories}

    def _row_to_memory(self, row) -> Memory:
        return Memory(
            id=row[0],
            content=row[1],
            tier=row[2],
            confidence=row[3],
            created_at=row[4],
            session_count=row[5],
            access_count=row[6],
            is_priority=bool(row[7]),
            is_suppressed=bool(row[8]),
            suppression_reason=row[9],
            source=row[10],
            metadata=json.loads(row[11]) if row[11] else {},
        )

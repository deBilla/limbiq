"""SQLite-backed memory store with tiers.

Thread-safe: uses threading.local() for per-thread SQLite connections.
This is required for multi-threaded frameworks like Gradio where handlers
run in worker threads separate from the main thread.
"""

import json
import os
import sqlite3
import threading
import uuid
import time
import numpy as np

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
        self._local = threading.local()

        os.makedirs(store_path, exist_ok=True)
        self._db_path = os.path.join(store_path, f"{user_id}.db")
        self._init_tables()

        # Embedding cache for vectorized search
        self._emb_matrix = None  # Cached numpy matrix of all embeddings
        self._emb_ids = None     # Corresponding memory IDs
        self._emb_dirty = True   # Invalidation flag
        self._emb_include_suppressed = None  # Track last include_suppressed value
        self._emb_lock = threading.Lock()    # Thread-safe cache rebuilds

    @property
    def db(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection."""
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self._db_path)
            self._local.conn = conn
        return conn

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

            CREATE INDEX IF NOT EXISTS idx_memories_suppressed ON memories(is_suppressed);
            CREATE INDEX IF NOT EXISTS idx_memories_priority ON memories(is_priority);
            CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(tier);
            CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence);
            CREATE INDEX IF NOT EXISTS idx_signal_log_type ON signal_log(signal_type);
            CREATE INDEX IF NOT EXISTS idx_signal_log_timestamp ON signal_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_compressed ON conversations(compressed);
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

        # Incremental cache update: append to existing cache instead of
        # triggering a full rebuild (which blocks 50-200ms for 1000 memories)
        if embedding is not None:
            self._append_to_cache(memory_id, embedding)
        else:
            self._emb_dirty = True

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

    def _append_to_cache(self, memory_id: str, embedding: list[float]) -> None:
        """Incrementally append a single embedding to the cache without full rebuild."""
        with self._emb_lock:
            # Only append if cache is initialized and not dirty
            # (if dirty, next search will do a full rebuild anyway)
            if self._emb_dirty or self._emb_matrix is None:
                self._emb_dirty = True
                return

            # Skip if include_suppressed tracking would be wrong
            # (new memories are never suppressed, so this is safe for both modes)
            emb_array = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(emb_array)
            if norm == 0:
                return
            emb_array = emb_array / norm

            # Check dimension compatibility
            if self._emb_matrix.shape[1] != emb_array.shape[0]:
                self._emb_dirty = True
                return

            self._emb_matrix = np.vstack([self._emb_matrix, emb_array.reshape(1, -1)])
            self._emb_ids = np.append(self._emb_ids, memory_id)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        include_suppressed: bool = False,
    ) -> list[Memory]:
        """Search memories by embedding similarity using vectorized numpy computation."""
        # Rebuild cache if dirty, uninitialized, or include_suppressed flag changed
        if self._emb_dirty or self._emb_matrix is None or self._emb_include_suppressed != include_suppressed:
            self._rebuild_embedding_cache(include_suppressed)

        # Return empty if no embeddings available
        if self._emb_matrix is None or len(self._emb_matrix) == 0:
            return []

        # Normalize query embedding
        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm

        # Handle dimension mismatch (e.g. model changed since embeddings were stored)
        cache_dim = self._emb_matrix.shape[1]
        query_dim = query.shape[0]
        if cache_dim != query_dim:
            # Invalidate cache and fall back to row-by-row search
            self._emb_dirty = True
            self._emb_matrix = None
            self._emb_ids = None
            return self._search_fallback(query_embedding, top_k, include_suppressed)

        # Batch cosine similarity (matrix already normalized in cache)
        sims = self._emb_matrix @ query

        # Get top-k indices by similarity
        top_indices = np.argsort(sims)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if sims[idx] <= 0:
                break
            mem_id = self._emb_ids[idx]
            mem = self._get_memory_by_id(mem_id)
            if mem:
                results.append(mem)

        return results

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
        self._emb_dirty = True

    def restore(self, memory_id: str) -> None:
        self.db.execute(
            "UPDATE memories SET is_suppressed = 0, confidence = 0.8, "
            "suppression_reason = NULL WHERE id = ?",
            (memory_id,),
        )
        self.db.commit()
        self._emb_dirty = True

    def age_all(self) -> None:
        self.db.execute("UPDATE memories SET session_count = session_count + 1")
        self.db.commit()

    def increment_access(self, memory_id: str) -> None:
        self.db.execute(
            "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
            (memory_id,),
        )
        self.db.commit()

    def increment_access_batch(self, memory_ids: list[str]) -> None:
        """Increment access count for multiple memories in a single transaction."""
        if not memory_ids:
            return
        placeholders = ",".join("?" * len(memory_ids))
        self.db.execute(
            f"UPDATE memories SET access_count = access_count + 1 WHERE id IN ({placeholders})",
            memory_ids,
        )
        self.db.commit()

    def boost_confidence(self, memory_id: str, new_confidence: float) -> None:
        self.db.execute(
            "UPDATE memories SET confidence = ? WHERE id = ?",
            (new_confidence, memory_id),
        )
        self.db.commit()
        self._emb_dirty = True

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
        self._emb_dirty = True
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
            row = cursor.fetchone()
            stats[tier.value] = row[0] if row else 0

        cursor = self.db.execute(
            "SELECT COUNT(*) FROM memories WHERE is_suppressed = 1"
        )
        row = cursor.fetchone()
        stats["suppressed"] = row[0] if row else 0

        cursor = self.db.execute(
            "SELECT COUNT(*) FROM memories WHERE is_priority = 1 AND is_suppressed = 0"
        )
        row = cursor.fetchone()
        stats["priority"] = row[0] if row else 0

        cursor = self.db.execute("SELECT COUNT(*) FROM memories")
        row = cursor.fetchone()
        stats["total"] = row[0] if row else 0

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

    def _search_fallback(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        include_suppressed: bool = False,
    ) -> list[Memory]:
        """Row-by-row cosine similarity search — used when dimensions mismatch."""
        clause = "" if include_suppressed else "WHERE is_suppressed = 0"
        cursor = self.db.execute(
            f"SELECT id, embedding FROM memories {clause}"
        )

        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm
        query_dim = query.shape[0]

        scored = []
        for row_id, emb_bytes in cursor.fetchall():
            emb = _deserialize_embedding(emb_bytes)
            if emb is None or len(emb) != query_dim:
                continue
            emb_arr = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(emb_arr)
            if norm == 0:
                continue
            sim = float(np.dot(emb_arr / norm, query))
            if sim > 0:
                scored.append((row_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for mem_id, _ in scored[:top_k]:
            mem = self._get_memory_by_id(mem_id)
            if mem:
                results.append(mem)
        return results

    def _rebuild_embedding_cache(self, include_suppressed: bool = False) -> None:
        """Rebuild the normalized embedding matrix cache."""
        with self._emb_lock:
            clause = "" if include_suppressed else "WHERE is_suppressed = 0"
            cursor = self.db.execute(
                f"SELECT id, embedding FROM memories {clause}"
            )

            embeddings = []
            memory_ids = []

            for row_id, emb_bytes in cursor.fetchall():
                emb = _deserialize_embedding(emb_bytes)
                if emb is None:
                    continue

                # Normalize embedding
                emb_array = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    emb_array = emb_array / norm
                    embeddings.append(emb_array)
                    memory_ids.append(row_id)

            if embeddings:
                self._emb_matrix = np.array(embeddings, dtype=np.float32)
                self._emb_ids = np.array(memory_ids)
            else:
                self._emb_matrix = None
                self._emb_ids = None

            self._emb_dirty = False
            self._emb_include_suppressed = include_suppressed

    def _get_memory_by_id(self, memory_id: str) -> Memory | None:
        """Retrieve a single memory by ID."""
        cursor = self.db.execute(
            "SELECT id, content, tier, confidence, created_at, session_count, "
            "access_count, is_priority, is_suppressed, suppression_reason, "
            "source, metadata, embedding FROM memories WHERE id = ?",
            (memory_id,),
        )
        row = cursor.fetchone()
        return self._row_to_memory(row) if row else None

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

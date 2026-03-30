"""SQLite-backed memory store with tiers.

Thread-safe: uses threading.local() for per-thread SQLite connections.
This is required for multi-threaded frameworks like Gradio where handlers
run in worker threads separate from the main thread.

Vector search: uses FAISS (if available) for fast approximate nearest
neighbor search, falling back to numpy brute-force if not installed.
"""

import json
import logging
import os
import sqlite3
import threading
import uuid
import time
import numpy as np

from limbiq.types import Memory, MemoryTier, SuppressionReason

logger = logging.getLogger(__name__)

# Try to import FAISS — optional dependency for fast vector search.
# On macOS, FAISS and PyTorch both link libomp which can segfault
# under concurrent threading. We mitigate by setting OMP_NUM_THREADS=1
# and KMP_DUPLICATE_LIB_OK=TRUE before either library initializes OpenMP.
_HAS_FAISS = False
try:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


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

        # FAISS vector index (preferred) or numpy fallback
        self._index_lock = threading.Lock()  # Thread-safe index writes

        if _HAS_FAISS:
            self._use_faiss = True
            self._faiss_index = None       # faiss.IndexIDMap wrapping IndexFlatIP
            self._faiss_dim = None          # Embedding dimension (set on first add)
            self._id_to_int: dict[str, int] = {}   # UUID str → int64
            self._int_to_id: dict[int, str] = {}   # int64 → UUID str
            self._next_int_id = 1
            self._faiss_path = os.path.join(store_path, f"{user_id}.faiss")
            self._faiss_map_path = os.path.join(store_path, f"{user_id}.faiss_map.json")
            self._load_faiss_index()
            logger.info("FAISS vector index enabled")
        else:
            self._use_faiss = False
            # Numpy fallback state
            self._emb_matrix = None
            self._emb_ids = None
            self._emb_dirty = True
            self._emb_include_suppressed = None
            self._emb_lock = threading.Lock()
            logger.info("FAISS not available, using numpy fallback for vector search")

    @property
    def db(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection."""
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
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

    # ── FAISS index management ─────────────────────────────────

    def _load_faiss_index(self) -> None:
        """Load FAISS index and ID mapping from disk if they exist."""
        if not self._use_faiss:
            return

        if os.path.exists(self._faiss_path) and os.path.exists(self._faiss_map_path):
            try:
                self._faiss_index = faiss.read_index(self._faiss_path)
                with open(self._faiss_map_path, 'r') as f:
                    map_data = json.load(f)
                self._id_to_int = map_data.get("id_to_int", {})
                self._int_to_id = {int(k): v for k, v in map_data.get("int_to_id", {}).items()}
                self._next_int_id = map_data.get("next_int_id", 1)
                self._faiss_dim = map_data.get("dim", None)
                logger.info(
                    f"Loaded FAISS index: {self._faiss_index.ntotal} vectors, "
                    f"dim={self._faiss_dim}"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load FAISS index, rebuilding: {e}")

        # No saved index — will be built on first add or search
        self._faiss_index = None
        self._id_to_int = {}
        self._int_to_id = {}
        self._next_int_id = 1
        self._faiss_dim = None

    def _ensure_faiss_index(self, dim: int) -> None:
        """Create a FAISS index if one doesn't exist yet."""
        if self._faiss_index is not None:
            return
        # IndexFlatIP = inner product on normalized vectors = cosine similarity
        base_index = faiss.IndexFlatIP(dim)
        self._faiss_index = faiss.IndexIDMap(base_index)
        self._faiss_dim = dim

    def _get_int_id(self, memory_id: str) -> int:
        """Get or assign an int64 ID for a UUID string."""
        if memory_id in self._id_to_int:
            return self._id_to_int[memory_id]
        int_id = self._next_int_id
        self._next_int_id += 1
        self._id_to_int[memory_id] = int_id
        self._int_to_id[int_id] = memory_id
        return int_id

    def _faiss_add(self, memory_id: str, embedding: list[float]) -> None:
        """Add a single embedding to the FAISS index."""
        emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return
        emb = emb / norm

        dim = emb.shape[1]
        self._ensure_faiss_index(dim)

        # Dimension mismatch — skip (model changed)
        if self._faiss_dim != dim:
            logger.warning(
                f"Embedding dim {dim} != index dim {self._faiss_dim}, skipping add"
            )
            return

        int_id = self._get_int_id(memory_id)
        ids = np.array([int_id], dtype=np.int64)
        self._faiss_index.add_with_ids(emb, ids)

    def _faiss_remove(self, memory_id: str) -> None:
        """Remove a single embedding from the FAISS index."""
        if memory_id not in self._id_to_int:
            return
        if self._faiss_index is None:
            return
        int_id = self._id_to_int[memory_id]
        ids = np.array([int_id], dtype=np.int64)
        self._faiss_index.remove_ids(ids)

    def _rebuild_faiss_from_db(self, include_suppressed: bool = False) -> None:
        """Rebuild the entire FAISS index from SQLite embeddings."""
        clause = "" if include_suppressed else "WHERE is_suppressed = 0"
        cursor = self.db.execute(
            f"SELECT id, embedding FROM memories {clause}"
        )

        embeddings = []
        int_ids = []

        for row_id, emb_bytes in cursor.fetchall():
            emb = _deserialize_embedding(emb_bytes)
            if emb is None:
                continue
            emb_array = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(emb_array)
            if norm == 0:
                continue
            emb_array = emb_array / norm
            embeddings.append(emb_array)
            int_ids.append(self._get_int_id(row_id))

        if embeddings:
            dim = embeddings[0].shape[0]
            base_index = faiss.IndexFlatIP(dim)
            self._faiss_index = faiss.IndexIDMap(base_index)
            self._faiss_dim = dim

            emb_matrix = np.array(embeddings, dtype=np.float32)
            id_array = np.array(int_ids, dtype=np.int64)
            self._faiss_index.add_with_ids(emb_matrix, id_array)
            logger.info(f"Rebuilt FAISS index: {len(embeddings)} vectors, dim={dim}")
        else:
            self._faiss_index = None
            self._faiss_dim = None

    def save_index(self) -> None:
        """Persist FAISS index and ID mapping to disk."""
        if not self._use_faiss:
            return
        with self._index_lock:
            if self._faiss_index is None:
                return
            try:
                faiss.write_index(self._faiss_index, self._faiss_path)
                map_data = {
                    "id_to_int": self._id_to_int,
                    "int_to_id": {str(k): v for k, v in self._int_to_id.items()},
                    "next_int_id": self._next_int_id,
                    "dim": self._faiss_dim,
                }
                with open(self._faiss_map_path, 'w') as f:
                    json.dump(map_data, f)
                logger.debug(
                    f"Saved FAISS index: {self._faiss_index.ntotal} vectors"
                )
            except Exception as e:
                logger.warning(f"Failed to save FAISS index: {e}")

    def invalidate_index(self) -> None:
        """Invalidate the vector index, forcing a rebuild on next search.

        Use this instead of accessing _emb_dirty directly.
        """
        if self._use_faiss:
            with self._index_lock:
                self._rebuild_faiss_from_db()
        else:
            self._emb_dirty = True

    # ── Core CRUD ──────────────────────────────────────────────

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

        # Update vector index
        if embedding is not None:
            if self._use_faiss:
                with self._index_lock:
                    self._faiss_add(memory_id, embedding)
            else:
                self._append_to_cache(memory_id, embedding)
        else:
            if not self._use_faiss:
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

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        include_suppressed: bool = False,
    ) -> list[Memory]:
        """Search memories by embedding similarity."""
        if self._use_faiss:
            return self._search_faiss(query_embedding, top_k, include_suppressed)
        else:
            return self._search_numpy(query_embedding, top_k, include_suppressed)

    def search_with_scores(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        include_suppressed: bool = False,
    ) -> list[tuple[str, float]]:
        """Search and return (memory_id, similarity_score) pairs.

        Used by ActivationRetrieval to get embedding similarities without
        re-computing them from raw embeddings.
        """
        if self._use_faiss:
            return self._search_faiss_with_scores(query_embedding, top_k, include_suppressed)
        else:
            return self._search_numpy_with_scores(query_embedding, top_k, include_suppressed)

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

        if self._use_faiss:
            with self._index_lock:
                self._faiss_remove(memory_id)
        else:
            self._emb_dirty = True

    def restore(self, memory_id: str) -> None:
        self.db.execute(
            "UPDATE memories SET is_suppressed = 0, confidence = 0.8, "
            "suppression_reason = NULL WHERE id = ?",
            (memory_id,),
        )
        self.db.commit()

        if self._use_faiss:
            # Re-add embedding to index
            cursor = self.db.execute(
                "SELECT embedding FROM memories WHERE id = ?", (memory_id,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                emb = _deserialize_embedding(row[0])
                if emb:
                    with self._index_lock:
                        self._faiss_add(memory_id, emb)
        else:
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
        # Confidence changes don't affect the vector index — no invalidation needed

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
        # Get IDs before deleting so we can remove from FAISS
        if self._use_faiss:
            cursor = self.db.execute(
                "SELECT id FROM memories WHERE is_suppressed = 1 AND session_count >= ?",
                (min_sessions,),
            )
            ids_to_delete = [row[0] for row in cursor.fetchall()]

        cursor = self.db.execute(
            "DELETE FROM memories WHERE is_suppressed = 1 AND session_count >= ?",
            (min_sessions,),
        )
        self.db.commit()

        if self._use_faiss:
            with self._index_lock:
                for mid in ids_to_delete:
                    self._faiss_remove(mid)
        else:
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

    # ── FAISS search implementation ────────────────────────────

    def _search_faiss(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        include_suppressed: bool = False,
    ) -> list[Memory]:
        """Search using FAISS index."""
        scored = self._search_faiss_with_scores(query_embedding, top_k, include_suppressed)
        results = []
        for mem_id, score in scored:
            mem = self._get_memory_by_id(mem_id)
            if mem:
                results.append(mem)
        return results

    def _search_faiss_with_scores(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        include_suppressed: bool = False,
    ) -> list[tuple[str, float]]:
        """Search FAISS and return (memory_id, similarity) pairs."""
        # Build index if not yet initialized
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            with self._index_lock:
                self._rebuild_faiss_from_db(include_suppressed)
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                return []

        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm

        # Dimension mismatch check
        if self._faiss_dim and query.shape[1] != self._faiss_dim:
            return self._search_fallback_with_scores(
                query_embedding, top_k, include_suppressed
            )

        k = min(top_k, self._faiss_index.ntotal)
        distances, ids = self._faiss_index.search(query, k)

        results = []
        for i in range(k):
            int_id = int(ids[0][i])
            score = float(distances[0][i])
            if int_id == -1 or score <= 0:
                continue
            mem_id = self._int_to_id.get(int_id)
            if mem_id:
                results.append((mem_id, score))

        # If not including suppressed, filter out suppressed memories
        if not include_suppressed and results:
            suppressed_ids = self._get_suppressed_ids()
            results = [(mid, s) for mid, s in results if mid not in suppressed_ids]

        return results

    def _get_suppressed_ids(self) -> set[str]:
        """Get set of suppressed memory IDs."""
        cursor = self.db.execute("SELECT id FROM memories WHERE is_suppressed = 1")
        return {row[0] for row in cursor.fetchall()}

    # ── Numpy fallback search ──────────────────────────────────

    def _search_numpy(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        include_suppressed: bool = False,
    ) -> list[Memory]:
        """Search memories using numpy vectorized computation (fallback when FAISS unavailable)."""
        if self._emb_dirty or self._emb_matrix is None or self._emb_include_suppressed != include_suppressed:
            self._rebuild_embedding_cache(include_suppressed)

        if self._emb_matrix is None or len(self._emb_matrix) == 0:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm

        cache_dim = self._emb_matrix.shape[1]
        query_dim = query.shape[0]
        if cache_dim != query_dim:
            self._emb_dirty = True
            self._emb_matrix = None
            self._emb_ids = None
            return self._search_fallback(query_embedding, top_k, include_suppressed)

        sims = self._emb_matrix @ query
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

    def _search_numpy_with_scores(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        include_suppressed: bool = False,
    ) -> list[tuple[str, float]]:
        """Numpy search returning (memory_id, similarity) pairs."""
        if self._emb_dirty or self._emb_matrix is None or self._emb_include_suppressed != include_suppressed:
            self._rebuild_embedding_cache(include_suppressed)

        if self._emb_matrix is None or len(self._emb_matrix) == 0:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm

        cache_dim = self._emb_matrix.shape[1]
        query_dim = query.shape[0]
        if cache_dim != query_dim:
            self._emb_dirty = True
            self._emb_matrix = None
            self._emb_ids = None
            return self._search_fallback_with_scores(query_embedding, top_k, include_suppressed)

        sims = self._emb_matrix @ query
        top_indices = np.argsort(sims)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if sims[idx] <= 0:
                break
            results.append((self._emb_ids[idx], float(sims[idx])))

        return results

    def _search_fallback(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        include_suppressed: bool = False,
    ) -> list[Memory]:
        """Row-by-row cosine similarity search — used when dimensions mismatch."""
        scored = self._search_fallback_with_scores(query_embedding, top_k, include_suppressed)
        results = []
        for mem_id, _ in scored:
            mem = self._get_memory_by_id(mem_id)
            if mem:
                results.append(mem)
        return results

    def _search_fallback_with_scores(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        include_suppressed: bool = False,
    ) -> list[tuple[str, float]]:
        """Row-by-row cosine similarity returning (memory_id, score) pairs."""
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
        return scored[:top_k]

    # ── Numpy cache management (fallback only) ─────────────────

    def _append_to_cache(self, memory_id: str, embedding: list[float]) -> None:
        """Incrementally append a single embedding to the numpy cache."""
        with self._emb_lock:
            if self._emb_dirty or self._emb_matrix is None:
                self._emb_dirty = True
                return

            emb_array = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(emb_array)
            if norm == 0:
                return
            emb_array = emb_array / norm

            if self._emb_matrix.shape[1] != emb_array.shape[0]:
                self._emb_dirty = True
                return

            self._emb_matrix = np.vstack([self._emb_matrix, emb_array.reshape(1, -1)])
            self._emb_ids = np.append(self._emb_ids, memory_id)

    def _rebuild_embedding_cache(self, include_suppressed: bool = False) -> None:
        """Rebuild the normalized embedding matrix cache (numpy fallback)."""
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

    # ── Internal helpers ───────────────────────────────────────

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

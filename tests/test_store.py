from limbiq.store.memory_store import MemoryStore
from limbiq.types import MemoryTier, SuppressionReason


class TestMemoryStore:
    def setup_method(self, method, tmp_dir=None):
        pass

    def test_store_and_retrieve(self, tmp_dir):
        store = MemoryStore(tmp_dir, "test_user")
        embedding = [0.1] * 10

        mem = store.store(
            content="User's name is Dimuthu",
            tier=MemoryTier.SHORT,
            confidence=0.9,
            is_priority=False,
            source="conversation",
            metadata={"key": "value"},
            embedding=embedding,
        )

        assert mem.id is not None
        assert mem.content == "User's name is Dimuthu"

        results = store.search([0.1] * 10, top_k=5)
        assert len(results) >= 1
        assert results[0].content == "User's name is Dimuthu"

    def test_priority_memories(self, tmp_dir):
        store = MemoryStore(tmp_dir, "test_user")
        embedding = [0.1] * 10

        store.store(
            content="Priority fact",
            tier=MemoryTier.PRIORITY,
            confidence=1.0,
            is_priority=True,
            source="manual",
            embedding=embedding,
        )

        store.store(
            content="Normal fact",
            tier=MemoryTier.SHORT,
            confidence=0.5,
            is_priority=False,
            source="conversation",
            embedding=embedding,
        )

        priority = store.get_priority_memories()
        assert len(priority) == 1
        assert priority[0].content == "Priority fact"

    def test_suppress_and_restore(self, tmp_dir):
        store = MemoryStore(tmp_dir, "test_user")
        embedding = [0.1] * 10

        mem = store.store(
            content="Fact to suppress",
            tier=MemoryTier.SHORT,
            confidence=0.9,
            embedding=embedding,
        )

        store.suppress(mem.id, SuppressionReason.MANUAL)

        suppressed = store.get_suppressed()
        assert len(suppressed) == 1
        assert suppressed[0].confidence == 0.05

        # Should not appear in normal search
        results = store.search([0.1] * 10, top_k=5, include_suppressed=False)
        assert all(r.id != mem.id for r in results)

        store.restore(mem.id)
        assert len(store.get_suppressed()) == 0

    def test_age_all(self, tmp_dir):
        store = MemoryStore(tmp_dir, "test_user")
        embedding = [0.1] * 10

        mem = store.store(content="Aging fact", embedding=embedding)
        assert mem.session_count == 0

        store.age_all()

        results = store.search([0.1] * 10, top_k=1)
        assert results[0].session_count == 1

    def test_get_stale(self, tmp_dir):
        store = MemoryStore(tmp_dir, "test_user")
        embedding = [0.1] * 10

        store.store(content="Stale fact", embedding=embedding)

        for _ in range(11):
            store.age_all()

        stale = store.get_stale(min_sessions=10)
        assert len(stale) >= 1

    def test_delete_old_suppressed(self, tmp_dir):
        store = MemoryStore(tmp_dir, "test_user")
        embedding = [0.1] * 10

        mem = store.store(content="To delete", embedding=embedding)
        store.suppress(mem.id, SuppressionReason.MANUAL)

        for _ in range(31):
            store.age_all()

        deleted = store.delete_old_suppressed(min_sessions=30)
        assert deleted >= 1

    def test_stats(self, tmp_dir):
        store = MemoryStore(tmp_dir, "test_user")
        embedding = [0.1] * 10

        store.store(
            content="Short memory",
            tier=MemoryTier.SHORT,
            embedding=embedding,
        )
        store.store(
            content="Priority memory",
            tier=MemoryTier.PRIORITY,
            is_priority=True,
            embedding=embedding,
        )

        stats = store.get_stats()
        assert stats["short"] == 1
        assert stats["priority"] >= 1
        assert stats["total"] >= 2

    def test_export_all(self, tmp_dir):
        store = MemoryStore(tmp_dir, "test_user")
        embedding = [0.1] * 10

        store.store(content="Export test", embedding=embedding)

        export = store.export_all()
        assert export["user_id"] == "test_user"
        assert len(export["memories"]) >= 1

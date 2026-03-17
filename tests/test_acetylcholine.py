"""Tests for the Acetylcholine signal -- domain focus."""

from limbiq import Limbiq
from limbiq.signals.acetylcholine import AcetylcholineSignal


class TestAcetylcholineDetection:
    def test_heuristic_topic_detection(self):
        signal = AcetylcholineSignal()
        topic = signal._detect_topic_heuristic("Tell me about Python decorators")
        assert topic is not None
        assert "python" in topic or "decorators" in topic

    def test_no_topic_on_empty(self):
        signal = AcetylcholineSignal()
        topic = signal._detect_topic_heuristic("hi")
        # "hi" is too short / stopword-like
        assert topic is None or len(topic) <= 3

    def test_depth_request_triggers_cluster(self, tmp_dir):
        """Explicit depth request creates a cluster."""
        lq = Limbiq(store_path=tmp_dir, user_id="test")
        lq.start_session()

        events = lq.observe(
            "Tell me more about Python decorators",
            "Decorators are functions that modify other functions..."
        )

        ach_events = [e for e in events if e.signal_type == "acetylcholine"]
        assert len(ach_events) >= 1

        clusters = lq.get_clusters()
        assert len(clusters) >= 1

    def test_cluster_creation_on_sustained_topic(self, tmp_dir):
        """A cluster is created after sustained discussion on one topic."""
        lq = Limbiq(store_path=tmp_dir, user_id="test")
        lq.start_session()

        lq.observe("What is attention in transformers?", "Attention is...")
        lq.observe("How does attention compute scores?", "The attention scores...")
        events = lq.observe("What about multi-head attention?", "Multi-head splits...")

        clusters = lq.get_clusters()
        assert len(clusters) >= 1

    def test_cluster_loaded_on_return(self, tmp_dir):
        """When a topic comes up again, its cluster memories are loaded."""
        lq = Limbiq(store_path=tmp_dir, user_id="test")

        # Session 1: build the cluster
        lq.start_session()
        lq.observe("Tell me more about decorators", "Decorators wrap functions...")
        lq.observe("Tell me more about decorator patterns", "Common patterns include...")
        lq.end_session()

        # Session 2: return to topic
        lq.start_session()
        result = lq.process("One more question about decorators")

        assert "DOMAIN KNOWLEDGE" in result.context
        assert len(result.clusters_loaded) >= 1

    def test_cluster_grows_over_sessions(self, tmp_dir):
        """Clusters accumulate knowledge across sessions."""
        lq = Limbiq(store_path=tmp_dir, user_id="test")

        # Session 1
        lq.start_session()
        lq.observe("Tell me more about Rust ownership", "Ownership is...")
        lq.end_session()

        clusters = lq.get_clusters()
        assert len(clusters) >= 1
        count_before = len(clusters[0].memory_ids)

        # Session 2: more on same topic
        lq.start_session()
        lq.observe("Tell me more about Rust borrowing", "Borrowing lets you...")
        lq.end_session()

        clusters_after = lq.get_clusters()
        matching = [c for c in clusters_after if c.id == clusters[0].id]
        assert len(matching) == 1
        assert len(matching[0].memory_ids) > count_before

    def test_cluster_memories_retrievable(self, tmp_dir):
        """Cluster memories can be retrieved by cluster ID."""
        lq = Limbiq(store_path=tmp_dir, user_id="test")
        lq.start_session()

        lq.observe("Tell me more about async Python", "Async allows...")
        lq.end_session()

        clusters = lq.get_clusters()
        assert len(clusters) >= 1

        memories = lq.get_cluster_memories(clusters[0].id)
        assert len(memories) >= 1

    def test_max_cluster_size(self, tmp_dir):
        """Clusters are capped at MAX_CLUSTER_SIZE."""
        from limbiq.store.cluster_store import ClusterStore

        lq = Limbiq(store_path=tmp_dir, user_id="test")
        cluster = lq._core.cluster_store.create_cluster("test_topic")

        # Add more than max
        for i in range(20):
            embedding = lq._core.embeddings.embed(f"fact {i}")
            mem = lq._core.store.store(
                content=f"fact {i}", embedding=embedding
            )
            lq._core.cluster_store.add_memory_to_cluster(cluster.id, mem.id)

        clusters = lq.get_clusters()
        matching = [c for c in clusters if c.id == cluster.id]
        assert len(matching[0].memory_ids) <= ClusterStore.MAX_CLUSTER_SIZE

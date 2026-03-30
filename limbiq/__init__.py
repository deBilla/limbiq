"""
Limbiq -- Neurotransmitter-inspired adaptive learning for LLMs.

Stripped to essentials: signals + graph generation + self-healing.

Usage:
    from limbiq import Limbiq

    lq = Limbiq(store_path="./data", user_id="user1")
    result = lq.process("Hello, my name is Dimuthu")
    # ... send to LLM with result.context injected ...
    lq.observe("Hello, my name is Dimuthu", llm_response)
"""

from collections.abc import Callable

from limbiq.core import LimbiqCore
from limbiq.types import (
    ProcessResult,
    SignalEvent,
    Memory,
    MemoryTier,
    SignalType,
    SuppressionReason,
    BehavioralRule,
    KnowledgeCluster,
    RetrievalConfig,
)
from limbiq.graph.propagation import ActiveGraphPropagation, PropagationResult
from limbiq.graph.entity_state import EntityState, EntityStateStore

try:
    from limbiq.graph.gnn import GNNPropagation
except ImportError:
    GNNPropagation = None

try:
    from limbiq.graph.pattern_completion import PatternCompletion
except ImportError:
    PatternCompletion = None

try:
    from limbiq.graph.reasoning import GraphReasoner, ReasoningResult
except ImportError:
    GraphReasoner = None
    ReasoningResult = None


class Limbiq:
    """Main interface for Limbiq — signals + graph focused."""

    def __init__(
        self,
        store_path: str = "./neuro_data",
        user_id: str = "default",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_fn: Callable | None = None,
    ):
        self._core = LimbiqCore(store_path, user_id, embedding_model, llm_fn)

    def process(
        self, message: str, conversation_history: list[dict] = None
    ) -> ProcessResult:
        """Process a user message and return enriched context for the LLM."""
        return self._core.process(message, conversation_history)

    def observe(
        self, message: str, response: str, feedback: str = None
    ) -> list[SignalEvent]:
        """Observe a completed exchange and fire appropriate signals."""
        return self._core.observe(message, response, feedback)

    def start_session(self) -> None:
        """Start a new conversation session."""
        self._core.start_session()

    def end_session(self) -> dict:
        """End session and run cleanup."""
        return self._core.end_session()

    # -- Explicit signals (Dopamine / GABA) --

    def dopamine(self, content: str) -> None:
        """Manually tag a piece of information as high-priority."""
        embedding = self._core.embeddings.embed(content)
        self._core.store.store(
            content=content,
            tier=MemoryTier.PRIORITY,
            confidence=1.0,
            is_priority=True,
            source="manual_dopamine",
            metadata={},
            embedding=embedding,
        )

    def gaba(self, memory_id: str) -> None:
        """Manually suppress a memory."""
        self._core.store.suppress(memory_id, SuppressionReason.MANUAL)

    def correct(self, correction: str) -> None:
        """Apply a correction -- combines dopamine (new info) + gaba (old info)."""
        self.dopamine(correction)

        embedding = self._core.embeddings.embed(correction)
        related = self._core.store.search(embedding, top_k=5, include_suppressed=False)
        for m in related:
            if m.content != correction:
                self._core.store.suppress(m.id, SuppressionReason.CONTRADICTED)

    # -- Serotonin (behavioral rules) --

    def get_active_rules(self) -> list[BehavioralRule]:
        """Return all crystallized behavioral rules."""
        return self._core.rule_store.get_active_rules()

    def deactivate_rule(self, rule_id: str) -> None:
        self._core.rule_store.deactivate_rule(rule_id)

    def reactivate_rule(self, rule_id: str) -> None:
        self._core.rule_store.reactivate_rule(rule_id)

    # -- Acetylcholine (knowledge clusters) --

    def get_clusters(self) -> list[KnowledgeCluster]:
        return self._core.cluster_store.get_all_clusters()

    def get_cluster_memories(self, cluster_id: str) -> list[Memory]:
        return self._core.cluster_store.get_cluster_memories(cluster_id)

    # -- Active Graph Propagation --

    def propagate(self) -> "PropagationResult":
        """Run Phase 1 active graph propagation cycle."""
        prop = ActiveGraphPropagation(
            store=self._core.store,
            graph=self._core.graph,
            embedding_engine=self._core.embeddings,
            user_name=self._core.user_id,
            entity_state_store=self._core.entity_state_store,
        )
        return prop.propagate()

    def propagate_gnn(self, model_dir: str = "data/gnn",
                      train_first: bool = False, epochs: int = 200) -> dict:
        """Run Phase 2 GNN-based propagation."""
        gnn = GNNPropagation(
            store=self._core.store,
            graph=self._core.graph,
            embedding_engine=self._core.embeddings,
            user_name=self._core.user_id,
            model_dir=model_dir,
        )
        if train_first:
            gnn.train_and_save(epochs=epochs)
        return gnn.propagate()

    def compute_activations_gnn(self, query: str = None,
                                model_dir: str = "data/gnn") -> list:
        gnn = GNNPropagation(
            store=self._core.store,
            graph=self._core.graph,
            embedding_engine=self._core.embeddings,
            user_name=self._core.user_id,
            model_dir=model_dir,
        )
        query_embedding = None
        if query:
            query_embedding = self._core.embeddings.embed(query)
        return gnn.compute_activations(query_embedding)

    def enable_activation_retrieval(self, gnn_model_dir: str = "data/gnn") -> bool:
        return self._core.enable_activation_retrieval(gnn_model_dir)

    def generate_graph_training_data(self, output_path: str = "data/training/graph_training.jsonl") -> int:
        from limbiq.retrieval.activation_retrieval import GraphTrainingDataGenerator
        gen = GraphTrainingDataGenerator(
            graph=self._core.graph,
            inference_engine=self._core.inference_engine,
            store_db=self._core.store.db,
            user_name=self._core.user_id,
        )
        return gen.export_jsonl(output_path)

    def run_pattern_completion(self, model_dir: str = "data/pattern",
                               train_transe: bool = True, epochs: int = 500) -> dict:
        pc = PatternCompletion(
            store=self._core.store,
            graph=self._core.graph,
            embedding_engine=self._core.embeddings,
            user_name=self._core.user_id,
            model_dir=model_dir,
        )
        return pc.run(train_transe_model=train_transe, epochs=epochs)

    def compute_activations(self, query: str = None) -> list:
        prop = ActiveGraphPropagation(
            store=self._core.store,
            graph=self._core.graph,
            embedding_engine=self._core.embeddings,
            user_name=self._core.user_id,
            entity_state_store=self._core.entity_state_store,
        )
        query_embedding = None
        if query:
            query_embedding = self._core.embeddings.embed(query)
        return prop.compute_activations(query_embedding)

    # -- Graph Reasoning (Phase 5) --

    def train_reasoner(self, model_dir: str = "data/reasoner",
                       epochs: int = 100) -> dict:
        user_name = getattr(self._core, '_graph_user_name', self._core.user_id)
        reasoner = GraphReasoner(self._core.graph, user_name=user_name,
                                 model_dir=model_dir)
        return reasoner.train(epochs=epochs)

    def reason(self, question: str, model_dir: str = "data/reasoner") -> "ReasoningResult":
        user_name = getattr(self._core, '_graph_user_name', self._core.user_id)
        reasoner = GraphReasoner(self._core.graph, user_name=user_name,
                                 model_dir=model_dir)
        return reasoner.reason(question)

    # -- Knowledge Graph --

    def heal_graph(self) -> None:
        """Self-heal: junk cleanup + inference + connectivity bridging."""
        self._core.graph.heal()
        self._core.inference_engine.run_full_inference()
        self._core._heal_graph_connectivity()
        self._core.graph_query.mark_dirty()

    def train_encoder(self) -> dict:
        """Train transformer entity encoder from existing graph data.

        The encoder learns entity type classification and relation detection
        from the graph built by regex+LLM extraction. After training, it
        produces learned entity spans and types alongside embeddings.
        """
        return self._core.entity_extractor.train_encoder()

    def get_graph_connectivity(self) -> dict:
        """Return graph connectivity statistics."""
        entities = self._core.graph.get_all_entities()
        if not entities:
            return {"components": 0, "entities": 0, "fully_connected": True}

        entity_ids = {e.id for e in entities}
        adj: dict[str, set[str]] = {eid: set() for eid in entity_ids}
        relations = self._core.graph.get_all_relations(include_inferred=True)
        for r in relations:
            if r.subject_id in adj and r.object_id in adj:
                adj[r.subject_id].add(r.object_id)
                adj[r.object_id].add(r.subject_id)

        components = self._core._find_connected_components(adj)
        return {
            "components": len(components),
            "entities": len(entities),
            "relations": len(relations),
            "fully_connected": len(components) <= 1,
            "component_sizes": sorted([len(c) for c in components], reverse=True),
        }

    def get_graph_stats(self) -> dict:
        return self._core.graph.get_stats()

    def get_entities(self) -> list:
        return self._core.graph.get_all_entities()

    def get_relations(self, include_inferred: bool = True) -> list:
        return self._core.graph.get_all_relations(include_inferred)

    def delete_relation(self, subject_name: str, predicate: str, object_name: str) -> None:
        self._core.graph.delete_relation(subject_name, predicate, object_name)
        self._core.graph.remove_inferred()

    def delete_relations_between(self, name_a: str, name_b: str) -> None:
        self._core.graph.delete_relations_between(name_a, name_b)
        self._core.graph.remove_inferred()

    def query_graph(self, question: str) -> dict:
        return self._core.graph_query.try_answer(question)

    def describe_entity(self, name: str) -> str:
        return self._core.inference_engine.describe_entity(name)

    def get_world_summary(self) -> str:
        return self._core.inference_engine.get_user_world(self._core._graph_user_name)

    # -- Unified Encoder --

    def train_encoder_bootstrap(self, num_epochs: int = 50) -> dict:
        """Bootstrap-train the unified encoder from hardcoded pattern data.

        Converts the 234+ hardcoded patterns into training examples,
        then trains the self-attention encoder to generalize beyond them.
        For better results, use train_encoder_full() which downloads
        real datasets from HuggingFace.
        """
        return self._core.encoder.train_bootstrap(num_epochs)

    def train_encoder_full(self, max_per_class: int = 2000, epochs: int = 30) -> dict:
        """Train the unified encoder on real datasets from HuggingFace.

        Downloads GoEmotions, PersonaChat, CLINC150, dair-ai/emotion,
        and generates synthetic correction examples. ~10K+ balanced
        training examples across all 7 intent categories.

        Requires: pip install datasets
        """
        from limbiq.encoder_training import download_and_train
        return download_and_train(self._core.encoder, max_per_class, epochs)

    @property
    def encoder_available(self) -> bool:
        """Whether the unified encoder is trained and ready."""
        return self._core.encoder.available

    # -- Entity State (distributed cellular memory) --

    def get_entity_state(self, entity_id: str) -> "EntityState":
        """Get the persistent state for a specific entity."""
        return self._core.entity_state_store.get_state(entity_id)

    def get_all_entity_states(self) -> list["EntityState"]:
        """Get all entity states, ordered by resting activation."""
        return self._core.entity_state_store.get_all_states()

    def get_top_activated_entities(self, limit: int = 20) -> list["EntityState"]:
        """Get entities with highest resting activation."""
        return self._core.entity_state_store.get_top_activated(limit)

    def get_sentinels(self) -> list["EntityState"]:
        """Get all entities with active sentinel patterns."""
        return self._core.entity_state_store.get_sentinels()

    # -- Inspection --

    def get_stats(self) -> dict:
        return self._core.store.get_stats()

    def get_signal_log(self, limit: int = 50) -> list[SignalEvent]:
        return self._core.signal_log.get_recent(limit)

    def get_priority_memories(self) -> list[Memory]:
        return self._core.store.get_priority_memories()

    def get_suppressed(self) -> list[Memory]:
        return self._core.store.get_suppressed()

    def restore_memory(self, memory_id: str) -> None:
        self._core.store.restore(memory_id)

    def export_state(self) -> dict:
        return self._core.store.export_all()

    def get_full_profile(self) -> dict:
        return {
            "priority_facts": [
                {"id": m.id, "content": m.content}
                for m in self.get_priority_memories()
            ],
            "behavioral_rules": [
                {"id": r.id, "rule": r.rule_text, "pattern": r.pattern_key}
                for r in self.get_active_rules()
            ],
            "knowledge_domains": [
                {"id": c.id, "topic": c.topic, "memory_count": len(c.memory_ids)}
                for c in self.get_clusters()
            ],
            "graph": {
                "stats": self.get_graph_stats(),
                "world_summary": self.get_world_summary(),
                "entities": [
                    {"id": e.id, "name": e.name, "type": e.entity_type}
                    for e in self.get_entities()
                ],
            },
            "suppressed_count": len(self.get_suppressed()),
            "stats": self.get_stats(),
        }


__all__ = [
    "Limbiq",
    "ProcessResult",
    "SignalEvent",
    "Memory",
    "MemoryTier",
    "SignalType",
    "SuppressionReason",
    "BehavioralRule",
    "KnowledgeCluster",
    "RetrievalConfig",
    # Entity state (distributed cellular memory)
    "EntityState",
    "EntityStateStore",
    # Graph propagation
    "ActiveGraphPropagation",
    "PropagationResult",
    "GNNPropagation",
    "PatternCompletion",
    # Graph reasoning
    "GraphReasoner",
    "ReasoningResult",
]

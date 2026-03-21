"""
Limbiq -- Neurotransmitter-inspired adaptive learning for LLMs.

Usage:
    from limbiq import Limbiq

    lq = Limbiq(store_path="./data", user_id="user1")
    result = lq.process("Hello, my name is Dimuthu")
    # ... send to LLM with result.context injected ...
    lq.observe("Hello, my name is Dimuthu", llm_response)
"""

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
from limbiq.graph.gnn import GNNPropagation
from limbiq.graph.pattern_completion import PatternCompletion
from limbiq.graph.reasoning import GraphReasoner, ReasoningResult
from limbiq.llm import LLMClient
from limbiq.search import SearchClient, SearchResult


class Limbiq:
    """Main interface for Limbiq."""

    def __init__(
        self,
        store_path: str = "./neuro_data",
        user_id: str = "default",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_fn=None,
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

    def start_session(self):
        """Start a new conversation session."""
        self._core.start_session()

    def end_session(self) -> dict:
        """End session and run compression/cleanup."""
        return self._core.end_session()

    # -- Explicit signals (Dopamine / GABA) --

    def dopamine(self, content: str):
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

    def gaba(self, memory_id: str):
        """Manually suppress a memory."""
        self._core.store.suppress(memory_id, SuppressionReason.MANUAL)

    def correct(self, correction: str):
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

    def deactivate_rule(self, rule_id: str):
        """Deactivate a behavioral rule (reversible)."""
        self._core.rule_store.deactivate_rule(rule_id)

    def reactivate_rule(self, rule_id: str):
        """Reactivate a previously deactivated rule."""
        self._core.rule_store.reactivate_rule(rule_id)

    # -- Acetylcholine (knowledge clusters) --

    def get_clusters(self) -> list[KnowledgeCluster]:
        """Return all knowledge clusters."""
        return self._core.cluster_store.get_all_clusters()

    def get_cluster_memories(self, cluster_id: str) -> list[Memory]:
        """Return all memories in a cluster."""
        return self._core.cluster_store.get_cluster_memories(cluster_id)

    # -- Active Graph Propagation --

    def propagate(self) -> "PropagationResult":
        """
        Run a full active graph propagation cycle.

        This is the Phase 1 implementation of the active knowledge graph:
        1. Suppress noise memories
        2. Deflate inflated priorities
        3. Merge duplicate memories
        4. Repair knowledge graph (extract entities from existing memories)
        5. Run graph inference
        6. Compute node activations

        Returns PropagationResult with statistics.
        """
        prop = ActiveGraphPropagation(
            store=self._core.store,
            graph=self._core.graph,
            embedding_engine=self._core.embeddings,
            user_name=self._core.user_id,
        )
        return prop.propagate()

    def propagate_gnn(self, model_dir: str = "data/gnn",
                      train_first: bool = False, epochs: int = 200) -> dict:
        """
        Run Phase 2 GNN-based propagation.

        If train_first=True, trains the GNN on Phase 1 labels before propagating.
        Falls back to Phase 1 if no trained model is found.
        """
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
        """Compute GNN-based activations, optionally biased by a query."""
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
        """
        Enable Phase 4 activation-weighted retrieval.
        Requires a trained GNN model from Phase 2.
        Returns True if successfully enabled.
        """
        return self._core.enable_activation_retrieval(gnn_model_dir)

    def generate_graph_training_data(self, output_path: str = "data/training/graph_training.jsonl") -> int:
        """
        Generate LoRA training data from the knowledge graph.
        Returns the number of training pairs generated.
        """
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
        """
        Run Phase 3 pattern completion: entity resolution, relation mining,
        TransE training, and learned inference.
        """
        pc = PatternCompletion(
            store=self._core.store,
            graph=self._core.graph,
            embedding_engine=self._core.embeddings,
            user_name=self._core.user_id,
            model_dir=model_dir,
        )
        return pc.run(train_transe_model=train_transe, epochs=epochs)

    def compute_activations(self, query: str = None) -> list:
        """
        Compute activation states for all memory nodes.
        Optionally bias activations toward a query.
        Returns list of ActivationState objects sorted by activation.
        """
        prop = ActiveGraphPropagation(
            store=self._core.store,
            graph=self._core.graph,
            embedding_engine=self._core.embeddings,
            user_name=self._core.user_id,
        )
        query_embedding = None
        if query:
            query_embedding = self._core.embeddings.embed(query)
        return prop.compute_activations(query_embedding)

    # -- Graph Reasoning (Phase 5) --

    def train_reasoner(self, model_dir: str = "data/reasoner",
                       epochs: int = 100) -> dict:
        """
        Train the micro-transformer graph reasoner.
        Generates synthetic QA data from the knowledge graph and trains.
        """
        user_name = getattr(self._core, '_graph_user_name', self._core.user_id)
        reasoner = GraphReasoner(self._core.graph, user_name=user_name,
                                 model_dir=model_dir)
        return reasoner.train(epochs=epochs)

    def reason(self, question: str, model_dir: str = "data/reasoner") -> "ReasoningResult":
        """
        Answer a question using the micro-transformer reasoner.
        Returns ReasoningResult with answer, confidence, and reasoning trace.
        Falls back gracefully if model not trained.
        """
        user_name = getattr(self._core, '_graph_user_name', self._core.user_id)
        reasoner = GraphReasoner(self._core.graph, user_name=user_name,
                                 model_dir=model_dir)
        return reasoner.reason(question)

    # -- Knowledge Graph --

    def get_graph_stats(self) -> dict:
        """Return knowledge graph statistics."""
        return self._core.graph.get_stats()

    def get_entities(self) -> list:
        """Return all entities in the knowledge graph."""
        return self._core.graph.get_all_entities()

    def get_relations(self, include_inferred: bool = True) -> list:
        """Return all relations in the knowledge graph."""
        return self._core.graph.get_all_relations(include_inferred)

    def query_graph(self, question: str) -> dict:
        """Query the knowledge graph with a natural language question."""
        return self._core.graph_query.try_answer(question)

    def describe_entity(self, name: str) -> str:
        """Get a natural language description of an entity."""
        return self._core.inference_engine.describe_entity(name)

    def get_world_summary(self) -> str:
        """Get a compact summary of everything known about the user."""
        return self._core.inference_engine.get_user_world(self._core._graph_user_name)

    # -- Inspection --

    def get_stats(self) -> dict:
        """Return memory statistics."""
        return self._core.store.get_stats()

    def get_signal_log(self, limit: int = 50) -> list[SignalEvent]:
        """Return recent signal events."""
        return self._core.signal_log.get_recent(limit)

    def get_priority_memories(self) -> list[Memory]:
        """Return all dopamine-tagged priority memories."""
        return self._core.store.get_priority_memories()

    def get_suppressed(self) -> list[Memory]:
        """Return all GABA-suppressed memories."""
        return self._core.store.get_suppressed()

    def restore_memory(self, memory_id: str):
        """Undo a GABA suppression."""
        self._core.store.restore(memory_id)

    def export_state(self) -> dict:
        """Export full state as JSON for debugging."""
        return self._core.store.export_all()

    def get_full_profile(self) -> dict:
        """Return a complete user profile."""
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
]

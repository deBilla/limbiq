"""
Limbiq Core -- the main orchestrator.

Coordinates detection, signals, storage, retrieval, and context building.
"""

import uuid

from limbiq.types import (
    ProcessResult,
    SignalEvent,
    SignalType,
    MemoryTier,
    SuppressionReason,
    RetrievalConfig,
)
from limbiq.store.memory_store import MemoryStore
from limbiq.store.embeddings import EmbeddingEngine
from limbiq.store.signal_log import SignalLog
from limbiq.store.rule_store import RuleStore
from limbiq.store.cluster_store import ClusterStore
from limbiq.context.builder import ContextBuilder
from limbiq.compression.compressor import MemoryCompressor
from limbiq.signals.dopamine import DopamineSignal
from limbiq.signals.gaba import GABASignal
from limbiq.signals.serotonin import SerotoninSignal
from limbiq.signals.acetylcholine import AcetylcholineSignal
from limbiq.signals.norepinephrine import NorepinephrineSignal
from limbiq.graph.store import GraphStore
from limbiq.graph.entities import EntityExtractor
from limbiq.graph.inference import InferenceEngine
from limbiq.graph.query import GraphQuery
from limbiq.retrieval.activation_retrieval import (
    ActivationRetrieval, GraphStateContextBuilder, ScoredMemory,
)


class LimbiqCore:
    def __init__(
        self,
        store_path: str,
        user_id: str,
        embedding_model: str,
        llm_fn=None,
    ):
        self.llm_fn = llm_fn
        self.user_id = user_id
        self.store = MemoryStore(store_path, user_id)
        self.embeddings = EmbeddingEngine(embedding_model)
        self.compressor = MemoryCompressor(llm_fn)
        self.context_builder = ContextBuilder()
        self.signal_log = SignalLog(self.store)
        self.rule_store = RuleStore(self.store)
        self.cluster_store = ClusterStore(self.store)

        # Knowledge graph — shares the same SQLite DB
        self.graph = GraphStore(self.store)
        self.entity_extractor = EntityExtractor(self.graph, user_id, llm_fn)
        self.inference_engine = InferenceEngine(self.graph)

        # Resolve actual user name in graph (handles "default" → "Dimuthu" merge)
        self._graph_user_name = self._resolve_graph_user_name(user_id)
        self.graph_query = GraphQuery(
            self.graph, self.inference_engine, self._graph_user_name
        )

        # Phase 4: activation-weighted retrieval + graph context builder
        self._activation_retrieval = None  # Lazy init (needs GNN model)
        self._graph_context_builder = GraphStateContextBuilder(
            self.graph, self.inference_engine
        )
        self._use_activation_retrieval = False  # Enabled after GNN training

        # v0.1 signals (detect in observe loop)
        self.signals = [
            DopamineSignal(),
            GABASignal(),
            NorepinephrineSignal(),
        ]

        # v0.2 signals (separate tracking)
        self.serotonin = SerotoninSignal(llm_fn)
        self.acetylcholine = AcetylcholineSignal(llm_fn)
        self.norepinephrine = self.signals[2]  # Same instance

        self._current_session_id = None
        self._conversation_buffer: list[dict] = []
        self._retrieval_config = RetrievalConfig()
        self._pending_ne_events: list[SignalEvent] = []

        # Embedding cache: avoids re-embedding the same message in observe()
        self._cached_query_text: str | None = None
        self._cached_query_embedding = None

    def process(self, message: str, conversation_history: list[dict] = None) -> ProcessResult:
        query_embedding = self.embeddings.embed(message)

        # Cache for reuse in observe()
        self._cached_query_text = message
        self._cached_query_embedding = query_embedding

        # FIRST: try to answer from graph (zero LLM cost, zero extra tokens)
        graph_answer = self.graph_query.try_answer(message)
        graph_context = None
        if graph_answer["answered"] and graph_answer["confidence"] > 0.8:
            graph_context = graph_answer["answer"]

        # Compact world summary from graph (replaces raw memory dump)
        world_summary = self.inference_engine.get_user_world(self._graph_user_name)

        # Reset retrieval config, then apply any pending norepinephrine effects
        self._retrieval_config.reset()
        if self._pending_ne_events:
            self.norepinephrine.apply_observe_effects(
                self._pending_ne_events, self._retrieval_config
            )
            self._pending_ne_events = []

        # Norepinephrine: check for topic shift during process
        ne_events = self.norepinephrine.detect_for_process(
            message, query_embedding, self.embeddings, self._retrieval_config
        )
        for event in ne_events:
            self.signal_log.log(event)

        # Retrieve with current config (possibly widened by norepinephrine)
        # Phase 4: use activation-weighted retrieval when available
        scored_memories = None
        if self._use_activation_retrieval and self._activation_retrieval:
            try:
                scored_memories = self._activation_retrieval.search(
                    query=message,
                    query_embedding=query_embedding,
                    top_k=self._retrieval_config.top_k,
                )
                # Convert ScoredMemory -> Memory-like objects for backward compat
                relevant = self._scored_to_memories(scored_memories)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Activation retrieval failed: {e}")
                scored_memories = None

        if scored_memories is None:
            # Fallback to standard embedding-only retrieval
            relevant = self.store.search(
                query_embedding,
                top_k=self._retrieval_config.top_k,
                include_suppressed=False,
            )

        priority = self.store.get_priority_memories()
        suppressed_ids = {m.id for m in self.store.get_suppressed()}

        for m in relevant:
            self.store.increment_access(m.id)

        # Serotonin: get active behavioral rules
        active_rules = self.rule_store.get_active_rules()

        # Acetylcholine: check for domain cluster matches
        cluster, cluster_memories = self.acetylcholine.detect_topic_for_retrieval(
            message, conversation_history or self._conversation_buffer,
            self.cluster_store, self.llm_fn,
        )
        clusters_loaded = []
        if cluster:
            clusters_loaded = [cluster.topic]

        # Phase 4: use graph-state context builder when activation retrieval is on
        if scored_memories is not None:
            context = self._graph_context_builder.build_context(
                query=message,
                scored_memories=scored_memories,
                world_summary=world_summary,
                graph_answer=graph_context,
                active_rules=active_rules,
                caution_flag=self._retrieval_config.caution_flag,
            )
        else:
            # Fallback to original context builder
            context = self.context_builder.build(
                priority, relevant, suppressed_ids,
                active_rules=active_rules,
                cluster_memories=cluster_memories,
                caution_flag=self._retrieval_config.caution_flag,
                graph_answer=graph_context,
                world_summary=world_summary,
            )

        return ProcessResult(
            context=context,
            signals_fired=ne_events,
            memories_retrieved=len(relevant),
            priority_count=len(priority),
            suppressed_count=len(suppressed_ids),
            active_rules=active_rules,
            clusters_loaded=clusters_loaded,
            norepinephrine_active=bool(ne_events) or bool(self._retrieval_config.caution_flag),
        )

    def observe(
        self, message: str, response: str, feedback: str = None
    ) -> list[SignalEvent]:
        events = []

        self._conversation_buffer.append({"role": "user", "content": message})
        self._conversation_buffer.append({"role": "assistant", "content": response})

        # Reuse cached embedding from process() if same message
        if message == self._cached_query_text and self._cached_query_embedding is not None:
            msg_embedding = self._cached_query_embedding
        else:
            msg_embedding = self.embeddings.embed(message)

        existing_memories = self.store.search(msg_embedding, top_k=5)

        # Run v0.1 signals (dopamine, gaba, norepinephrine detect)
        for signal in self.signals:
            detected = signal.detect(
                message=message,
                response=response,
                feedback=feedback,
                memories=existing_memories,
            )
            for event in detected:
                signal.apply(event, self.store, self.embeddings)
                self.signal_log.log(event)
                events.append(event)

        # Store norepinephrine events for next process() call
        ne_events = [e for e in events if e.signal_type == SignalType.NOREPINEPHRINE]
        if ne_events:
            self._pending_ne_events.extend(ne_events)

        # Serotonin: analyze patterns
        serotonin_events = self.serotonin.analyze_and_track(
            message, response, self._current_session_id or "default",
            self.rule_store, self.llm_fn,
        )
        for event in serotonin_events:
            self.signal_log.log(event)
            events.append(event)

        # Acetylcholine: analyze topic focus
        ach_events = self.acetylcholine.analyze_topic(
            message, response, self._conversation_buffer,
            self.cluster_store, self.store, self.embeddings, self.llm_fn,
        )
        for event in ach_events:
            self.signal_log.log(event)
            events.append(event)

        # When GABA or Dopamine fires (correction/denial), clean the graph too.
        # Find entity names mentioned in the correction and remove wrong relations.
        for event in events:
            if event.signal_type in (SignalType.GABA, SignalType.DOPAMINE):
                if event.trigger in ("user_correction", "user_denial", "contradicted"):
                    self._correct_graph(message)

        # Extract entities and relations from USER message only.
        # LLM responses contain general knowledge (e.g., "Machine Learning",
        # "consult your vet") that creates junk entities in the user's graph.
        self.entity_extractor.extract_from_memory(message)

        # Only store raw user message as memory when NO signals fired and
        # NO entities were extracted. When dopamine fires or the graph captures
        # entities, the structured stores are authoritative. Raw "User said: ..."
        # fragments alongside structured facts cause LLM confabulation.
        has_signals = len(events) > 0
        if len(message.strip()) > 20 and not has_signals:
            embedding = msg_embedding
            self.store.store(
                content=f"User said: {message}",
                tier=MemoryTier.SHORT,
                confidence=0.8,
                is_priority=False,
                source="conversation",
                metadata={},
                embedding=embedding,
            )

        # Clear embedding cache
        self._cached_query_text = None
        self._cached_query_embedding = None

        return events

    def end_session(self) -> dict:
        results = {"compressed": 0, "aged": 0, "suppressed": 0, "deleted": 0}

        if self._conversation_buffer:
            facts = self.compressor.compress_conversation(self._conversation_buffer)
            for fact in facts:
                embedding = self.embeddings.embed(fact)
                self.store.store(
                    content=fact,
                    tier=MemoryTier.MID,
                    confidence=0.7,
                    is_priority=False,
                    source="compression",
                    metadata={"source_turns": len(self._conversation_buffer)},
                    embedding=embedding,
                )
                results["compressed"] += 1

                # Extract entities from compressed facts (often cleaner than raw)
                self.entity_extractor.extract_from_memory(fact)

            self.store.store_conversation(
                self._conversation_buffer, self._current_session_id
            )
            self._conversation_buffer = []

        # Run graph inference after all new entities/relations are in
        inferred = self.inference_engine.run_full_inference()
        results["graph_inferred"] = inferred

        self.store.age_all()

        stale = self.store.get_stale(min_sessions=10)
        for m in stale:
            gaba_event = SignalEvent(
                signal_type=SignalType.GABA,
                trigger="never_accessed",
                details={"memory_id": m.id, "sessions_old": m.session_count},
                memory_ids_affected=[m.id],
            )
            self.store.suppress(m.id, SuppressionReason.NEVER_ACCESSED)
            self.signal_log.log(gaba_event)
            results["suppressed"] += 1

        results["deleted"] = self.store.delete_old_suppressed(min_sessions=30)

        return results

    def _correct_graph(self, message: str):
        """
        When a correction/denial is detected, find and remove wrong relations
        from the knowledge graph. Looks for entity names in the message and
        removes relations between them.
        """
        import re
        # Extract capitalized names from the correction message
        words = message.split()
        names = [w.strip(".,!?'\"") for w in words if w[0:1].isupper() and len(w) > 2]

        # Also check for "not X's Y" or "X is not Y" patterns
        denial_patterns = [
            r"(\w+)\s+is\s+not\s+(?:my\s+)?(\w+)",
            r"not\s+(\w+)(?:'s|s)\s+(\w+)",
            r"(\w+)\s+(?:isn't|isnt)\s+(?:my\s+)?(\w+)",
        ]
        for pattern in denial_patterns:
            match = re.search(pattern, message, re.I)
            if match:
                name_a = match.group(1)
                name_b = match.group(2)
                # Try to delete relations between these entities
                self.graph.delete_relations_between(name_a, name_b)
                # Also remove inferred relations that may have been built from wrong data
                self.graph.remove_inferred()
                return

        # Fallback: if we found 2+ entity names, remove relations between them
        found_entities = []
        for name in names:
            ent = self.graph.find_entity_by_name(name)
            if ent:
                found_entities.append(ent)

        if len(found_entities) >= 2:
            self.graph.delete_relations_between(
                found_entities[0].name, found_entities[1].name
            )
            self.graph.remove_inferred()

    def _resolve_graph_user_name(self, user_id: str) -> str:
        """
        Resolve the actual user entity name in the graph.
        After Phase 3 entity resolution, "default" may have been merged
        into the real user name (e.g., "Dimuthu").

        Always picks the person entity with the most outgoing explicit
        relations — that's the true user node, regardless of whether
        user_id still exists as an entity.
        """
        try:
            row = self.graph.db.execute(
                "SELECT e.name, COUNT(r.id) as rel_count "
                "FROM entities e "
                "JOIN relations r ON r.subject_id = e.id "
                "WHERE e.entity_type = 'person' AND r.is_inferred = 0 "
                "GROUP BY e.id ORDER BY rel_count DESC LIMIT 1"
            ).fetchone()
            if row and row[0] and row[1] >= 1:
                return row[0]
        except Exception:
            pass

        # Fallback: check onboarding profile for the real name
        try:
            ob_row = self.store.db.execute(
                "SELECT value FROM agent_profile WHERE key = 'user_name'"
            ).fetchone()
            if ob_row and ob_row[0] and ob_row[0] not in ("", "User", "default"):
                # Ensure entity exists
                entity = self.graph.find_entity_by_name(ob_row[0])
                if entity:
                    return ob_row[0]
        except Exception:
            pass

        # Fallback: check if user_id exists directly
        entity = self.graph.find_entity_by_name(user_id)
        if entity:
            return user_id

        return user_id

    def enable_activation_retrieval(self, gnn_model_dir: str = "data/gnn"):
        """Enable Phase 4 activation-weighted retrieval."""
        try:
            from limbiq.graph.gnn import GNNPropagation
            gnn = GNNPropagation(
                store=self.store,
                graph=self.graph,
                embedding_engine=self.embeddings,
                user_name=self._graph_user_name,
                model_dir=gnn_model_dir,
            )
            if gnn.load_model():
                self._activation_retrieval = ActivationRetrieval(
                    store=self.store,
                    graph=self.graph,
                    embedding_engine=self.embeddings,
                    gnn_propagation=gnn,
                    user_name=self._graph_user_name,
                )
                self._use_activation_retrieval = True
                return True
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to enable activation retrieval: {e}")
        return False

    def _scored_to_memories(self, scored: list[ScoredMemory]):
        """Convert ScoredMemory objects to Memory objects for backward compat."""
        from limbiq.types import Memory, MemoryTier
        memories = []
        for sm in scored:
            # Fetch full memory from store
            row = self.store.db.execute(
                "SELECT id, content, tier, confidence, created_at, session_count, "
                "access_count, is_priority, is_suppressed, suppression_reason, "
                "source, metadata, embedding FROM memories WHERE id = ?",
                (sm.memory_id,)
            ).fetchone()
            if row:
                memories.append(self.store._row_to_memory(row))
        return memories

    def start_session(self):
        self._current_session_id = str(uuid.uuid4())[:8]
        self._conversation_buffer = []
        self._pending_ne_events = []

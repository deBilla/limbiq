"""
Limbiq Core -- the main orchestrator.

Coordinates signals, graph, storage, retrieval, and context building.
Stripped to essentials: signals + graph generation + self-healing.
"""

import logging
import re
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

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
        llm_fn: Callable | None = None,
    ):
        self.llm_fn = llm_fn
        self.user_id = user_id
        self.store = MemoryStore(store_path, user_id)
        self.embeddings = EmbeddingEngine(embedding_model)
        self.context_builder = ContextBuilder()
        self.signal_log = SignalLog(self.store)
        self.rule_store = RuleStore(self.store)
        self.cluster_store = ClusterStore(self.store)

        # Knowledge graph — shares the same SQLite DB
        self.graph = GraphStore(self.store)
        self.entity_extractor = EntityExtractor(
            self.graph, user_id, llm_fn,
            embedding_engine=self.embeddings,
        )
        self.inference_engine = InferenceEngine(self.graph)

        # Resolve actual user name in graph (handles "default" → real name merge)
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

        # Run graph query, world summary, graph context, and priority memories in parallel
        with ThreadPoolExecutor(max_workers=4) as pool:
            graph_future = pool.submit(self.graph_query.try_answer, message)
            world_future = pool.submit(
                self.inference_engine.get_user_world, self._graph_user_name
            )
            graph_ctx_future = pool.submit(
                self.inference_engine.get_relevant_graph_context,
                message, self.embeddings, 5,
            )
            priority_future = pool.submit(self.store.get_priority_memories)

            graph_answer = graph_future.result()
            world_summary = world_future.result()
            graph_rel_context = graph_ctx_future.result()
            priority = priority_future.result()

        graph_fact = None
        if graph_answer["answered"] and graph_answer["confidence"] > 0.8:
            graph_fact = graph_answer["answer"]

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
        scored_memories = None
        if self._use_activation_retrieval and self._activation_retrieval:
            try:
                scored_memories = self._activation_retrieval.search(
                    query=message,
                    query_embedding=query_embedding,
                    top_k=self._retrieval_config.top_k,
                )
                relevant = self._scored_to_memories(scored_memories)
            except Exception as e:
                logger.warning(f"Activation retrieval failed: {e}")
                scored_memories = None

        if scored_memories is None:
            relevant = self.store.search(
                query_embedding,
                top_k=self._retrieval_config.top_k,
                include_suppressed=False,
            )
        suppressed_ids = {m.id for m in self.store.get_suppressed()}

        self.store.increment_access_batch([m.id for m in relevant])

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

        # Build context
        if scored_memories is not None:
            context = self._graph_context_builder.build_context(
                query=message,
                scored_memories=scored_memories,
                world_summary=world_summary,
                graph_answer=graph_fact,
                active_rules=active_rules,
                caution_flag=self._retrieval_config.caution_flag,
                graph_context=graph_rel_context,
            )
        else:
            context = self.context_builder.build(
                priority, relevant, suppressed_ids,
                active_rules=active_rules,
                cluster_memories=cluster_memories,
                caution_flag=self._retrieval_config.caution_flag,
                graph_answer=graph_fact,
                world_summary=world_summary,
                graph_context=graph_rel_context,
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

        # Detect if this is a correction
        _is_correction = any(
            e.signal_type in (SignalType.GABA, SignalType.DOPAMINE)
            and e.trigger in ("user_correction", "user_denial", "contradicted")
            for e in events
        )

        # Run serotonin, acetylcholine, and entity extraction in parallel
        with ThreadPoolExecutor(max_workers=4) as pool:
            sero_future = pool.submit(
                self.serotonin.analyze_and_track,
                message, response, self._current_session_id or "default",
                self.rule_store, self.llm_fn,
            )
            ach_future = pool.submit(
                self.acetylcholine.analyze_topic,
                message, response, self._conversation_buffer,
                self.cluster_store, self.store, self.embeddings, self.llm_fn,
            )
            ent_msg_future = pool.submit(
                self.entity_extractor.extract_from_memory, message
            )
            # Also extract from LLM response — it often confirms/clarifies
            # relationships ("your father-in-law Chandrasiri"). In response_mode,
            # only relations between existing graph entities are kept —
            # no new entities from response filler.
            ent_resp_future = pool.submit(
                self.entity_extractor.extract_from_memory,
                response, "", True,  # response_mode=True
            )

            serotonin_events = sero_future.result()
            ach_events = ach_future.result()
            ent_msg_future.result()
            ent_resp_future.result()

        # Graph correction AFTER extraction
        if _is_correction:
            self._correct_graph(message)

        # ── Graph self-healing: connectivity pass ──
        # After every observe(), heal junk entities, run inference,
        # and then bridge disconnected components to move toward
        # a fully connected graph.
        self.graph.heal()
        self.inference_engine.run_full_inference()
        self._heal_graph_connectivity()
        self.graph_query.mark_dirty()

        # Log signal events
        for event in serotonin_events:
            self.signal_log.log(event)
            events.append(event)
        for event in ach_events:
            self.signal_log.log(event)
            events.append(event)

        # Only store raw memory when NO signals fired
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
            # Extract entities from conversation buffer directly
            # (no compression module — entity extraction IS the compression)
            for turn in self._conversation_buffer:
                if turn["role"] == "user" and len(turn["content"].strip()) > 15:
                    self.entity_extractor.extract_from_memory(turn["content"])

            # Process any queued LLM extractions before inference
            self.entity_extractor.process_pending_extractions()

            self.store.store_conversation(
                self._conversation_buffer, self._current_session_id
            )
            self._conversation_buffer = []

        # Run graph inference after all new entities/relations are in
        inferred = self.inference_engine.run_full_inference()
        self._heal_graph_connectivity()
        self.graph_query.mark_dirty()
        results["graph_inferred"] = inferred

        # Phase 3: Semantic entity resolution
        try:
            from limbiq.graph.pattern_completion import PatternCompletion
            pc = PatternCompletion(
                store=self.store, graph=self.graph,
                embedding_engine=self.embeddings,
                user_name=self._graph_user_name,
            )
            er_result = pc.resolve()
            results["entities_merged"] = er_result.get("merged", 0)
            if er_result.get("merged", 0) > 0:
                self.inference_engine.run_full_inference()
                self._heal_graph_connectivity()
                self.graph_query.mark_dirty()
                logger.info(f"Entity resolution merged {er_result['merged']} entities")
        except ImportError:
            logger.debug("torch not available — skipping entity resolution in end_session")
        except Exception as e:
            logger.warning(f"Entity resolution failed in end_session: {e}")

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

        # Persist FAISS index to disk
        self.store.save_index()

        return results

    # ── Graph self-healing: connectivity bridging ──────────────

    def _heal_graph_connectivity(self):
        """Bridge disconnected components in the knowledge graph.

        After each interaction, find disconnected subgraphs and attempt
        to connect them through:
        1. Shared memory co-occurrence (entities mentioned together)
        2. Embedding similarity between entity names
        3. User-hub bridging (connect orphans to user node)

        Goal: each observe() moves the graph toward full connectivity.
        """
        try:
            all_entities = self.graph.get_all_entities()
            # Skip junk entities — don't waste effort bridging garbage
            entities = [e for e in all_entities if not self.graph._is_junk_name(e.name)]
            if len(entities) < 2:
                return

            # Build adjacency for connected component detection
            entity_ids = {e.id for e in entities}
            adj: dict[str, set[str]] = {eid: set() for eid in entity_ids}

            all_relations = self.graph.get_all_relations(include_inferred=True)
            for r in all_relations:
                if r.subject_id in adj and r.object_id in adj:
                    adj[r.subject_id].add(r.object_id)
                    adj[r.object_id].add(r.subject_id)

            # Find connected components via BFS
            components = self._find_connected_components(adj)
            if len(components) <= 1:
                return  # Already fully connected

            logger.info(
                f"Graph has {len(components)} disconnected components, "
                f"attempting to bridge"
            )

            # Strategy 1: Find the main component (largest) and user node
            entity_map = {e.id: e for e in entities}
            main_component = max(components, key=len)

            user_entity = self.graph.find_entity_by_name(self._graph_user_name)
            hub_id = user_entity.id if user_entity else None

            # If user not in main component, find the component with the user
            if hub_id and hub_id not in main_component:
                for comp in components:
                    if hub_id in comp:
                        main_component = comp
                        break

            # Strategy 2: For each orphan component, try to bridge to main
            bridged = 0
            for component in components:
                if component is main_component:
                    continue

                bridge_found = False

                # 2a: Check memory co-occurrence — if any entity in this
                # component was mentioned in the same memory as an entity
                # in the main component, create a "related_to" edge.
                for orphan_id in component:
                    orphan = entity_map.get(orphan_id)
                    if not orphan:
                        continue
                    for main_id in main_component:
                        main_ent = entity_map.get(main_id)
                        if not main_ent:
                            continue
                        if self._share_memory_context(orphan, main_ent):
                            from limbiq.graph.store import Relation
                            self.graph.add_relation(Relation(
                                subject_id=orphan_id,
                                predicate="related_to",
                                object_id=main_id,
                                confidence=0.6,
                                is_inferred=True,
                            ))
                            bridge_found = True
                            bridged += 1
                            break
                    if bridge_found:
                        break

                # 2b: Embedding similarity bridge — if entity name embeddings
                # are similar enough, create a semantic link
                if not bridge_found:
                    best_sim = 0.0
                    best_pair = None
                    for orphan_id in component:
                        orphan = entity_map.get(orphan_id)
                        if not orphan:
                            continue
                        orphan_emb = self.embeddings.embed(orphan.name)
                        for main_id in list(main_component)[:20]:  # Cap search
                            main_ent = entity_map.get(main_id)
                            if not main_ent:
                                continue
                            main_emb = self.embeddings.embed(main_ent.name)
                            sim = self.embeddings.similarity(orphan_emb, main_emb)
                            if sim > best_sim:
                                best_sim = sim
                                best_pair = (orphan_id, main_id)

                    if best_pair and best_sim > 0.5:
                        from limbiq.graph.store import Relation
                        self.graph.add_relation(Relation(
                            subject_id=best_pair[0],
                            predicate="related_to",
                            object_id=best_pair[1],
                            confidence=round(best_sim * 0.8, 3),
                            is_inferred=True,
                        ))
                        bridge_found = True
                        bridged += 1

                # 2c: Last resort — connect orphan to user hub
                if not bridge_found and hub_id and hub_id in main_component:
                    # Pick the most "important" entity in orphan component
                    # (most relations or most recently created)
                    best_orphan = max(
                        component,
                        key=lambda eid: len(adj.get(eid, set()))
                    )
                    from limbiq.graph.store import Relation
                    self.graph.add_relation(Relation(
                        subject_id=hub_id,
                        predicate="related_to",
                        object_id=best_orphan,
                        confidence=0.4,
                        is_inferred=True,
                    ))
                    bridged += 1

            if bridged:
                logger.info(f"Graph self-heal: bridged {bridged} disconnected components")

        except Exception as e:
            logger.warning(f"Graph connectivity healing failed: {e}")

    def _find_connected_components(self, adj: dict[str, set[str]]) -> list[set[str]]:
        """BFS-based connected component detection."""
        visited = set()
        components = []
        for node in adj:
            if node in visited:
                continue
            component = set()
            queue = [node]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in adj.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)
        return components

    def _share_memory_context(self, entity_a, entity_b) -> bool:
        """Check if two entities were ever mentioned in the same memory."""
        try:
            rows = self.store.db.execute(
                "SELECT content FROM memories WHERE is_suppressed = 0 "
                "AND (content LIKE ? OR content LIKE ?) "
                "LIMIT 50",
                (f"%{entity_a.name}%", f"%{entity_b.name}%"),
            ).fetchall()
            for (content,) in rows:
                if entity_a.name.lower() in content.lower() and entity_b.name.lower() in content.lower():
                    return True
        except Exception:
            pass
        return False

    def _correct_graph(self, message: str):
        """
        When a correction/denial is detected:
        1. DELETE wrong relations from the graph
        2. UPDATE entity types if the correction implies a type change
        3. RE-EXTRACT from the correction message to create correct relations
        4. Re-run inference to rebuild dependent chains
        """
        deleted_something = False

        # Strategy 1: Find denied PREDICATES and remove matching relations
        denied_pred_patterns = [
            r"not\s+(?:a\s+|an\s+|my\s+)?(\w+)",
            r"(?:isn't|isnt)\s+(?:a\s+|an\s+|my\s+)?(\w+)",
        ]
        for pattern in denied_pred_patterns:
            match = re.search(pattern, message, re.I)
            if match:
                denied_word = match.group(1).lower()
                from limbiq.graph.entities import _normalize_predicate, VALID_PREDICATES
                normalized = _normalize_predicate(denied_word)
                if normalized in VALID_PREDICATES:
                    words = message.split()
                    entity_names = [w.strip(".,!?'\"") for w in words
                                    if w[0:1].isupper() and len(w) > 2]
                    for name in entity_names:
                        ent = self.graph.find_entity_by_name(name)
                        if ent:
                            try:
                                self.graph.db.execute(
                                    "DELETE FROM relations WHERE object_id=? AND predicate=?",
                                    (ent.id, normalized),
                                )
                                self.graph.db.commit()
                                deleted_something = True
                                logger.info(f"Deleted denied predicate '{normalized}' → {name}")
                            except Exception as e:
                                logger.warning(f"Failed to delete denied predicate: {e}")
                    if deleted_something:
                        break

        # Strategy 2: "X is not Y"
        if not deleted_something:
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
                    self.graph.delete_relations_between(name_a, name_b)
                    deleted_something = True
                    break

        # Strategy 3: Fallback — extract capitalized names and delete between them
        if not deleted_something:
            words = message.split()
            names = [w.strip(".,!?'\"") for w in words if w[0:1].isupper() and len(w) > 2]
            found_entities = []
            for name in names:
                ent = self.graph.find_entity_by_name(name)
                if ent:
                    found_entities.append(ent)

            if len(found_entities) >= 2:
                self.graph.delete_relations_between(
                    found_entities[0].name, found_entities[1].name
                )
                deleted_something = True

        # Update entity types from correction context
        type_patterns = [
            (r"(\w+)\s+is\s+(?:a|an)\s+(dog|cat|animal|pet)", "animal"),
            (r"(\w+)\s+is\s+(?:a|an)\s+(person|human|man|woman|boy|girl)", "person"),
            (r"(\w+)\s+is\s+(?:a|an)\s+(place|city|country|town|village)", "place"),
            (r"(\w+)\s+is\s+(?:a|an)\s+(company|organization|org|firm)", "company"),
        ]
        for pattern, etype in type_patterns:
            match = re.search(pattern, message, re.I)
            if match:
                name = match.group(1).strip()
                ent = self.graph.find_entity_by_name(name)
                if ent and ent.entity_type != etype:
                    try:
                        self.graph.db.execute(
                            "UPDATE entities SET entity_type=? WHERE id=?",
                            (etype, ent.id),
                        )
                        self.graph.db.commit()
                        logger.info(f"Updated entity type: {name} → {etype}")
                    except Exception as e:
                        logger.warning(f"Failed to update entity type for {name}: {e}")

        # Clear inferred relations so they rebuild from correct data
        if deleted_something:
            self.graph.remove_inferred()

    def _resolve_graph_user_name(self, user_id: str) -> str:
        """Resolve the actual user entity name in the graph."""
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
        except Exception as e:
            logger.warning(f"Failed to resolve graph user name from relations: {e}")

        # Fallback: check onboarding profile
        try:
            ob_row = self.store.db.execute(
                "SELECT value FROM agent_profile WHERE key = 'user_name'"
            ).fetchone()
            if ob_row and ob_row[0] and ob_row[0] not in ("", "User", "default"):
                entity = self.graph.find_entity_by_name(ob_row[0])
                if entity:
                    return ob_row[0]
        except Exception as e:
            logger.warning(f"Failed to resolve graph user name from onboarding profile: {e}")

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
            logger.warning(f"Failed to enable activation retrieval: {e}")
        return False

    def _scored_to_memories(self, scored: list[ScoredMemory]):
        """Convert ScoredMemory objects to Memory objects for backward compat."""
        from limbiq.types import Memory, MemoryTier
        memories = []
        for sm in scored:
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
        if self._conversation_buffer:
            self.end_session()
        self._current_session_id = str(uuid.uuid4())[:8]
        self._conversation_buffer = []
        self._pending_ne_events = []
        self.norepinephrine.reset()

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
from limbiq.graph.entity_state import EntityStateStore
from limbiq.encoder import LimbiqEncoder
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

        # Distributed entity state (cellular memory)
        self.entity_state_store = EntityStateStore(self.store)

        # Knowledge graph — shares the same SQLite DB
        self.graph = GraphStore(self.store, entity_state_store=self.entity_state_store)
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

        # Unified encoder — shared self-attention for all classification tasks
        import os
        encoder_dir = os.path.join(store_path, "encoder")
        self.encoder = LimbiqEncoder(self.embeddings, model_dir=encoder_dir)

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

        # Correction-triggered retraining timestamp
        self._last_retrain_at: float = 0.0

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

        # ── Sentinel check: immune-style pattern matching ──
        # If the query mentions an entity that has a sentinel (from a past
        # correction), add a caution flag so the LLM knows this area has
        # outdated information that was previously corrected.
        sentinel_warnings = self._check_sentinels(message)
        if sentinel_warnings:
            existing_caution = self._retrieval_config.caution_flag or ""
            sentinel_msg = "; ".join(sentinel_warnings)
            if existing_caution:
                self._retrieval_config.caution_flag = f"{existing_caution}; {sentinel_msg}"
            else:
                self._retrieval_config.add_caution(sentinel_msg)

        # ── Expression masks: topic-sensitive entity property filtering ──
        # Like DNA methylation — which aspects of an entity are "expressed"
        # depends on the current conversation context.
        self._update_expression_masks(message)

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
                encoder=self.encoder,
            )
            for event in detected:
                signal.apply(event, self.store, self.embeddings,
                             graph_store=self.graph)
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

        # Skip entity extraction when correction/denial fired — extracting
        # from "Smurphy is a wrong name" creates junk entities like "wrong", "name".
        _skip_extraction = _is_correction

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

            if not _skip_extraction:
                ent_msg_future = pool.submit(
                    self.entity_extractor.extract_from_memory, message
                )
                ent_resp_future = pool.submit(
                    self.entity_extractor.extract_from_memory,
                    response, "", True,
                )
            else:
                ent_msg_future = None
                ent_resp_future = None

            serotonin_events = sero_future.result()
            ach_events = ach_future.result()
            if ent_msg_future:
                ent_msg_future.result()
            if ent_resp_future:
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

        # ── Distributed entity state: activate + flood signals ──
        # Runs AFTER entity extraction so newly extracted entities receive signals.
        # Like neurotransmitters flooding the synaptic space after signal detection.
        self._update_entity_states(message, events)

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

        # ── Correction-triggered learning ──
        # Any correction/denial signal means the user is teaching us.
        # Store as training data and retrain the unified encoder.
        _correction_events = [
            e for e in events
            if e.trigger in ("user_correction", "user_denial", "contradicted")
        ]
        if _correction_events:
            self._learn_from_correction(message, _correction_events)

        # Persist FAISS index after each observe so it survives crashes
        self.store.save_index()

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

        # Decay entity resting activations (like ion channels resetting)
        decayed = self.entity_state_store.decay_activations(decay_factor=0.95)
        results["entities_decayed"] = decayed

        # Cleanup orphaned entity states
        orphaned = self.entity_state_store.cleanup_orphaned()
        if orphaned:
            logger.info(f"Cleaned up {orphaned} orphaned entity states")

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
        """Correct the graph based on a correction/denial message.

        Simple approach: find entities mentioned in the message, then
        decide what to do based on a small set of action keywords.
        No regex pattern matching — just entity lookup + keywords.
        """
        msg_lower = message.lower()
        all_entities = self.graph.get_all_entities()
        deleted_something = False

        # Find entities mentioned in the message
        mentioned = [
            e for e in all_entities
            if len(e.name) > 1 and e.name.lower() in msg_lower
        ]

        if not mentioned:
            return

        # Action keywords — what does the user want to do?
        DELETE_KEYWORDS = {"wrong", "incorrect", "mistake", "fake", "remove",
                           "delete", "forget", "fabricated", "made up", "not real"}
        NEGATE_KEYWORDS = {"isn't", "isnt", "is not", "not", "doesn't", "doesnt",
                           "does not", "never", "no longer", "wasn't", "wasnt"}

        wants_delete = any(kw in msg_lower for kw in DELETE_KEYWORDS)
        wants_negate = any(kw in msg_lower for kw in NEGATE_KEYWORDS)

        if wants_delete and len(mentioned) == 1:
            # "Smurphy is wrong" — delete the entity entirely
            ent = mentioned[0]
            self.graph._delete_entity_and_relations(ent.id)
            self.graph.db.commit()
            # Suppress related memories
            rows = self.store.db.execute(
                "SELECT id FROM memories WHERE is_suppressed = 0 "
                "AND content LIKE ? LIMIT 10",
                (f"%{ent.name}%",),
            ).fetchall()
            for (mid,) in rows:
                self.store.suppress(mid, SuppressionReason.CONTRADICTED)
            # Clean entity state
            try:
                self.entity_state_store.db.execute(
                    "DELETE FROM entity_state WHERE entity_id = ?", (ent.id,),
                )
                self.entity_state_store.db.commit()
            except Exception:
                pass
            deleted_something = True
            logger.info(f"Deleted entity '{ent.name}' and all relations")

        elif wants_negate and len(mentioned) >= 2:
            # "Smurphy isnt Yuenshe's dog" — delete relations between them
            for i, a in enumerate(mentioned):
                for b in mentioned[i + 1:]:
                    self.graph.delete_relations_between(a.name, b.name)
                    deleted_something = True
                    logger.info(f"Deleted relations between '{a.name}' and '{b.name}'")

        elif wants_negate and len(mentioned) == 1:
            # "Smurphy is not a real name" — delete entity
            ent = mentioned[0]
            self.graph._delete_entity_and_relations(ent.id)
            self.graph.db.commit()
            deleted_something = True
            logger.info(f"Deleted negated entity '{ent.name}'")

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
                    entity_state_store=self.entity_state_store,
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

    def _update_entity_states(self, message: str, events: list[SignalEvent]):
        """Update per-entity state after signal detection.

        Biological model: neurotransmitters don't target specific neurons —
        they flood the synaptic space. Nearby receptors absorb the signal.

        1. Activate entities mentioned in the message (resting potential increases)
        2. For each signal event, flood ALL entities mentioned in the affected
           memory's text content — not just entities whose source_memory_id matches.
        3. Modulate activation by each entity's receptor density.
        4. On corrections/denials, create sentinel patterns on suppressed entities
           (immune T-cell memory — watches for stale references).
        """
        try:
            all_entities = self.graph.get_all_entities()
            if not all_entities:
                return

            # Find entities mentioned in the message
            msg_lower = message.lower()
            mentioned = [
                e for e in all_entities
                if len(e.name) > 1 and e.name.lower() in msg_lower
            ]

            # Activate mentioned entities
            for entity in mentioned:
                self.entity_state_store.activate(entity.id, delta=0.1)

            # Flood signals to entities in the neighborhood.
            # "Neighborhood" = entities mentioned in the message OR in affected memories.
            for event in events:
                # Strategy 1: Signal floods entities mentioned in the current message.
                # This is the primary pathway — the message IS the synaptic space.
                for entity in mentioned:
                    state = self.entity_state_store.get_state(entity.id)
                    receptor = state.receptor_density.get(
                        event.signal_type.value, 1.0
                    )
                    self.entity_state_store.activate(
                        entity.id, delta=0.05 * receptor
                    )
                    self.entity_state_store.record_signal(
                        entity.id, event.signal_type.value
                    )

                # Strategy 2: Also flood entities mentioned in affected memory content.
                # This catches entities that were stored earlier but relate to this signal.
                if event.memory_ids_affected:
                    for mem_id in event.memory_ids_affected:
                        try:
                            row = self.store.db.execute(
                                "SELECT content FROM memories WHERE id = ?",
                                (mem_id,),
                            ).fetchone()
                            if not row:
                                continue
                            mem_lower = row[0].lower()
                            for entity in all_entities:
                                if entity.id in {e.id for e in mentioned}:
                                    continue  # Already flooded via strategy 1
                                if len(entity.name) > 1 and entity.name.lower() in mem_lower:
                                    state = self.entity_state_store.get_state(entity.id)
                                    receptor = state.receptor_density.get(
                                        event.signal_type.value, 1.0
                                    )
                                    self.entity_state_store.activate(
                                        entity.id, delta=0.03 * receptor
                                    )
                                    self.entity_state_store.record_signal(
                                        entity.id, event.signal_type.value
                                    )
                        except Exception:
                            pass  # Memory may have been deleted

                # ── Sentinel creation on corrections/denials ──
                # Like immune T-cells: when a fact is corrected, create a sentinel
                # that watches for stale references to the OLD fact.
                if event.trigger in ("user_correction", "user_denial", "contradicted"):
                    self._create_sentinels_from_correction(event, all_entities)

        except Exception as e:
            logger.warning(f"Entity state update failed: {e}")

    def _learn_from_correction(self, message: str, events: list[SignalEvent]):
        """Learn from user corrections — store training pair + retrain encoder.

        Every correction/denial is a learning opportunity:
        1. Store the message + detected intent as a training pair
        2. If enough corrections accumulated, retrain the unified encoder
        """
        try:
            import time

            # Determine what intent this correction represents
            for event in events:
                if event.trigger == "user_correction":
                    intent_label = "correction"
                elif event.trigger == "user_denial":
                    intent_label = "denial"
                else:
                    intent_label = "correction"

                # Store as training pair for the unified encoder
                self.graph.store_relation_correction(
                    sentence=message,
                    subject_name=intent_label,  # Reusing field for intent label
                    predicate="intent",          # Marks this as intent training data
                    object_name=event.trigger,
                    is_positive=True,
                )

            # Check if enough corrections to retrain
            new_corrections = self.graph.count_corrections_since(self._last_retrain_at)
            if new_corrections < 3:
                return

            self._last_retrain_at = time.time()

            def _retrain():
                try:
                    # Collect intent corrections
                    corrections = self.graph.get_relation_corrections()
                    intent_pairs = [
                        (c["sentence"], c["subject_name"])
                        for c in corrections
                        if c["predicate"] == "intent" and c["is_positive"]
                    ]

                    if intent_pairs and self.encoder.available:
                        result = self.encoder.incremental_train(
                            intent_pairs, num_epochs=20
                        )
                        logger.info(f"Unified encoder retrained: {result}")

                    # Also retrain relation classifier if available
                    graph_encoder = self.entity_extractor._encoder
                    if graph_encoder is not None:
                        result = graph_encoder.incremental_train(
                            self.graph, num_epochs=20
                        )
                        logger.info(f"Relation classifier retrained: {result}")

                except Exception as e:
                    logger.warning(f"Incremental retraining failed: {e}")

            # Background thread to avoid blocking observe()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(_retrain)

        except Exception as e:
            logger.warning(f"Learn from correction failed: {e}")

    def _update_expression_masks(self, message: str):
        """Update expression masks based on current query topic.

        Like DNA methylation — the same entity has different aspects
        "expressed" depending on context. In a work conversation,
        work-related properties are active; in personal conversation,
        personal properties are active.

        Topic detection categories:
        - work: employment, projects, technical, meetings
        - personal: family, hobbies, feelings, social
        - location: travel, places, addresses
        """
        try:
            msg_lower = message.lower()

            # Simple topic detection via keyword presence
            work_keywords = {
                "work", "job", "office", "project", "meeting", "code",
                "deploy", "build", "architecture", "design", "sprint",
                "team", "manager", "deadline", "review", "merge",
                "production", "bug", "feature", "release", "standup",
            }
            personal_keywords = {
                "family", "wife", "husband", "father", "mother", "kid",
                "birthday", "dinner", "cook", "hobby", "paint", "music",
                "friend", "weekend", "vacation", "movie", "game", "fun",
                "love", "miss", "feel", "happy", "sad",
            }

            words = set(msg_lower.split())
            work_score = len(words & work_keywords)
            personal_score = len(words & personal_keywords)

            # Only update masks if there's a clear signal
            if work_score == 0 and personal_score == 0:
                return

            is_work = work_score > personal_score
            is_personal = personal_score > work_score

            # Update masks on entities mentioned in the message
            all_entities = self.graph.get_all_entities()
            for entity in all_entities:
                if len(entity.name) <= 1:
                    continue
                if entity.name.lower() not in msg_lower:
                    continue

                mask = {
                    "work": is_work,
                    "personal": is_personal,
                }
                self.entity_state_store.update_expression_mask(entity.id, mask)

        except Exception as e:
            logger.warning(f"Expression mask update failed: {e}")

    def _check_sentinels(self, message: str) -> list[str]:
        """Check if the message triggers any sentinel patterns.

        Like immune surveillance: sentinel entities watch for specific patterns
        and raise an alarm when detected. Returns a list of warning messages.
        """
        warnings = []
        try:
            sentinels = self.entity_state_store.get_sentinels()
            if not sentinels:
                return warnings

            msg_lower = message.lower()
            for sentinel in sentinels:
                if sentinel.sentinel_pattern and sentinel.sentinel_pattern in msg_lower:
                    # Find the entity name for a human-readable warning
                    entity = self.graph.find_entity_by_name(sentinel.sentinel_pattern)
                    entity_name = entity.name if entity else sentinel.sentinel_pattern
                    warnings.append(
                        f"Previously corrected info about '{entity_name}' — "
                        f"verify before referencing"
                    )
                    # Activate the sentinel entity (it did its job)
                    self.entity_state_store.activate(sentinel.entity_id, delta=0.05)
                    logger.info(f"Sentinel triggered: '{sentinel.sentinel_pattern}' in query")
        except Exception as e:
            logger.warning(f"Sentinel check failed: {e}")
        return warnings

    def _create_sentinels_from_correction(
        self, event: SignalEvent, all_entities: list
    ):
        """Create sentinel patterns from a correction/denial event.

        When a user says "No that's wrong, I don't live in London, I moved
        to Boston", find entities that are being NEGATED in the correction
        message and set sentinels on them.

        Strategy: parse the correction message for negation patterns
        ("don't live in X", "not at X", "no longer at X") and sentinel
        the entities found in the negated clause. These entities represent
        stale facts that should be flagged if referenced later.
        """
        try:
            msg = event.details.get("message", "")
            if not msg:
                return
            msg_lower = msg.lower()

            # Extract entities that appear in negation context within the message.
            # Patterns: "don't/doesn't/didn't [verb] [in/at] ENTITY"
            #           "not [at/in/from] ENTITY"
            #           "no longer [at/in] ENTITY"
            #           "moved from ENTITY"
            import re
            negation_patterns = [
                r"(?:don'?t|doesn'?t|didn'?t|not|no longer|never)\s+\w+\s+(?:in|at|from|to)\s+",
                r"moved\s+(?:from|away\s+from)\s+",
                r"left\s+",
                r"(?:don'?t|doesn'?t)\s+(?:live|work|stay)\s+",
            ]

            # Find entities in the negation zone vs the rest of the message.
            # Split on clause boundaries (comma, period, semicolon) to avoid
            # matching entities in adjacent clauses.
            negated_ids = set()
            affirmed_ids = set()
            entity_by_id = {e.id: e for e in all_entities}

            for entity in all_entities:
                if len(entity.name) <= 1:
                    continue
                name_lower = entity.name.lower()
                if name_lower not in msg_lower:
                    continue

                # Check if this entity appears right after a negation pattern,
                # within the SAME clause (stop at comma/period/semicolon).
                is_negated = False
                for neg_pat in negation_patterns:
                    for match in re.finditer(neg_pat, msg_lower):
                        neg_end = match.end()
                        # Find next clause boundary
                        rest = msg_lower[neg_end:]
                        clause_end = len(rest)
                        for sep in (",", ".", ";", " but ", " and "):
                            idx = rest.find(sep)
                            if idx != -1 and idx < clause_end:
                                clause_end = idx
                        search_zone = rest[:clause_end]
                        if name_lower in search_zone:
                            is_negated = True
                            break
                    if is_negated:
                        break

                if is_negated:
                    negated_ids.add(entity.id)
                else:
                    affirmed_ids.add(entity.id)

            # Create sentinels on negated entities (the stale facts)
            for eid in negated_ids:
                entity = entity_by_id[eid]
                # Don't sentinel entities that are also affirmed in the same message
                if eid in affirmed_ids:
                    continue

                existing_state = self.entity_state_store.get_state(entity.id)
                if existing_state.sentinel_pattern:
                    continue  # Already watching

                pattern = entity.name.lower()
                self.entity_state_store.set_sentinel(entity.id, pattern)
                logger.info(
                    f"Sentinel created: watching for '{entity.name}' "
                    f"(negated in correction)"
                )

            # Also check suppressed memories for additional sentinel targets
            if event.memory_ids_affected:
                for mem_id in event.memory_ids_affected:
                    row = self.store.db.execute(
                        "SELECT content, is_suppressed FROM memories WHERE id = ?",
                        (mem_id,),
                    ).fetchone()
                    if not row or not row[1]:
                        continue  # Only suppressed memories
                    content_lower = row[0].lower()
                    for entity in all_entities:
                        if len(entity.name) <= 1:
                            continue
                        if entity.name.lower() in content_lower:
                            if entity.id in affirmed_ids or entity.id in negated_ids:
                                continue
                            existing_state = self.entity_state_store.get_state(entity.id)
                            if existing_state.sentinel_pattern:
                                continue
                            pattern = entity.name.lower()
                            self.entity_state_store.set_sentinel(entity.id, pattern)
                            logger.info(
                                f"Sentinel created: watching for '{entity.name}' "
                                f"(from suppressed memory)"
                            )

        except Exception as e:
            logger.warning(f"Sentinel creation failed: {e}")

    def start_session(self):
        if self._conversation_buffer:
            self.end_session()
        self._current_session_id = str(uuid.uuid4())[:8]
        self._conversation_buffer = []
        self._pending_ne_events = []
        self.norepinephrine.reset()

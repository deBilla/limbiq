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


class LimbiqCore:
    def __init__(
        self,
        store_path: str,
        user_id: str,
        embedding_model: str,
        llm_fn=None,
    ):
        self.llm_fn = llm_fn
        self.store = MemoryStore(store_path, user_id)
        self.embeddings = EmbeddingEngine(embedding_model)
        self.compressor = MemoryCompressor(llm_fn)
        self.context_builder = ContextBuilder()
        self.signal_log = SignalLog(self.store)
        self.rule_store = RuleStore(self.store)
        self.cluster_store = ClusterStore(self.store)

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

    def process(self, message: str, conversation_history: list[dict] = None) -> ProcessResult:
        query_embedding = self.embeddings.embed(message)

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

        context = self.context_builder.build(
            priority, relevant, suppressed_ids,
            active_rules=active_rules,
            cluster_memories=cluster_memories,
            caution_flag=self._retrieval_config.caution_flag,
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

        existing_memories = self.store.search(
            self.embeddings.embed(message), top_k=5
        )

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

        # Store user message as short-term memory (if substantive)
        if len(message.strip()) > 20:
            embedding = self.embeddings.embed(message)
            self.store.store(
                content=f"User said: {message}",
                tier=MemoryTier.SHORT,
                confidence=0.8,
                is_priority=False,
                source="conversation",
                metadata={},
                embedding=embedding,
            )

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

            self.store.store_conversation(
                self._conversation_buffer, self._current_session_id
            )
            self._conversation_buffer = []

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

    def start_session(self):
        self._current_session_id = str(uuid.uuid4())[:8]
        self._conversation_buffer = []
        self._pending_ne_events = []

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
)
from limbiq.store.memory_store import MemoryStore
from limbiq.store.embeddings import EmbeddingEngine
from limbiq.store.signal_log import SignalLog
from limbiq.context.builder import ContextBuilder
from limbiq.compression.compressor import MemoryCompressor
from limbiq.signals.dopamine import DopamineSignal
from limbiq.signals.gaba import GABASignal


class LimbiqCore:
    def __init__(
        self,
        store_path: str,
        user_id: str,
        embedding_model: str,
        llm_fn=None,
    ):
        self.store = MemoryStore(store_path, user_id)
        self.embeddings = EmbeddingEngine(embedding_model)
        self.compressor = MemoryCompressor(llm_fn)
        self.context_builder = ContextBuilder()
        self.signal_log = SignalLog(self.store)

        self.signals = [
            DopamineSignal(),
            GABASignal(),
        ]

        self._current_session_id = None
        self._conversation_buffer: list[dict] = []

    def process(self, message: str, conversation_history: list[dict] = None) -> ProcessResult:
        query_embedding = self.embeddings.embed(message)

        relevant = self.store.search(
            query_embedding, top_k=10, include_suppressed=False
        )

        priority = self.store.get_priority_memories()

        suppressed_ids = {m.id for m in self.store.get_suppressed()}

        for m in relevant:
            self.store.increment_access(m.id)

        context = self.context_builder.build(priority, relevant, suppressed_ids)

        return ProcessResult(
            context=context,
            signals_fired=[],
            memories_retrieved=len(relevant),
            priority_count=len(priority),
            suppressed_count=len(suppressed_ids),
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

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
)


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

    # -- Explicit signals --

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


__all__ = [
    "Limbiq",
    "ProcessResult",
    "SignalEvent",
    "Memory",
    "MemoryTier",
    "SignalType",
    "SuppressionReason",
]

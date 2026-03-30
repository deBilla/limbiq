"""
GABA Signal -- "Suppress this. Let it fade."

Fires when a memory should be suppressed or forgotten.
GABA doesn't hard-delete -- it marks memories as suppressed
so they're excluded from retrieval but still exist for audit.
"""

from limbiq.signals.base import BaseSignal
from limbiq.types import (
    SignalEvent,
    SignalType,
    Memory,
    SuppressionReason,
)

DENIAL_PATTERNS = [
    "i never said",
    "that's not true about me",
    "i didn't tell you",
    "you're making that up",
    "that's wrong about me",
    "i don't ",
    "that's not me",
    "where did you get that",
    "i never mentioned",
    "that's fabricated",
]


class GABASignal(BaseSignal):
    @property
    def signal_type(self) -> str:
        return SignalType.GABA

    def detect(
        self,
        message: str,
        response: str = None,
        feedback: str = None,
        memories: list[Memory] = None,
        encoder=None,
    ) -> list[SignalEvent]:
        events = []
        msg_lower = message.lower() if message else ""

        # Check explicit negative feedback
        if feedback and feedback.lower() == "negative":
            events.append(
                SignalEvent(
                    signal_type=SignalType.GABA,
                    trigger="explicit_negative_feedback",
                    details={"feedback": feedback},
                )
            )
            return events

        # ── Encoder-based detection (preferred, high confidence only) ──
        if encoder and encoder.available:
            result = encoder.classify_intent(message)
            if result:
                intent, conf = result
                if intent == "denial" and conf > 0.7:
                    memory_ids = []
                    if memories:
                        memory_ids = [m.id for m in memories if not m.is_priority]
                    return [SignalEvent(
                        signal_type=SignalType.GABA,
                        trigger="user_denial",
                        details={"encoder_intent": intent, "confidence": conf,
                                 "message": message},
                        memory_ids_affected=memory_ids,
                    )]
            # Fall through to pattern matching if encoder wasn't confident

        # ── Pattern fallback ──
        for pattern in DENIAL_PATTERNS:
            if pattern in msg_lower:
                memory_ids = []
                if memories:
                    memory_ids = [m.id for m in memories if not m.is_priority]
                return [SignalEvent(
                    signal_type=SignalType.GABA,
                    trigger="user_denial",
                    details={"pattern": pattern, "message": message},
                    memory_ids_affected=memory_ids,
                )]

        return events

    def apply(self, event: SignalEvent, memory_store, embeddings=None,
              graph_store=None) -> None:
        if event.trigger == "user_denial":
            for memory_id in event.memory_ids_affected:
                memory_store.suppress(memory_id, SuppressionReason.USER_DENIED)
        elif event.trigger == "explicit_negative_feedback":
            for memory_id in event.memory_ids_affected:
                memory_store.suppress(memory_id, SuppressionReason.USER_DENIED)

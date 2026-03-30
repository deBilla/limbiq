"""
Norepinephrine Signal -- "Something changed. Be careful."

Fires when the conversation takes an unexpected turn.
Temporarily widens retrieval and adds caution flags.
Effects are transient -- they reset after each process() call.
"""

from limbiq.signals.base import BaseSignal
from limbiq.types import SignalEvent, SignalType, Memory, RetrievalConfig

FRUSTRATION_PATTERNS = [
    "i already told you", "i just said", "wrong again", "no no no",
    "listen", "pay attention", "i said", "as i mentioned",
    "for the third time", "are you even listening",
    "i already said", "i told you", "how many times",
]

CONTRADICTION_MARKERS = [
    "actually i", "actually my", "no i", "not anymore",
    "i changed", "i moved", "i switched", "i left", "i quit",
    "just moved", "just started", "just changed",
]


class NorepinephrineSignal(BaseSignal):

    def __init__(self):
        self._previous_embedding: list[float] | None = None

    def reset(self):
        """Clear state between sessions to avoid spurious topic shift on first message."""
        self._previous_embedding = None

    @property
    def signal_type(self) -> str:
        return SignalType.NOREPINEPHRINE

    def detect(
        self,
        message: str,
        response: str = None,
        feedback: str = None,
        memories: list[Memory] = None,
        encoder=None,
    ) -> list[SignalEvent]:
        """Detect in observe() context -- checks frustration and contradiction."""
        events = []
        msg_lower = message.lower() if message else ""

        # ── Encoder-based detection (preferred, high confidence only) ──
        if encoder and encoder.available:
            result = encoder.classify_intent(message)
            if result:
                intent, conf = result
                if intent == "frustration" and conf > 0.7:
                    return [SignalEvent(
                        signal_type=SignalType.NOREPINEPHRINE,
                        trigger="user_frustration",
                        details={"encoder_intent": intent, "confidence": conf,
                                 "message": message[:200]},
                    )]
                if intent == "contradiction" and conf > 0.7:
                    return [SignalEvent(
                        signal_type=SignalType.NOREPINEPHRINE,
                        trigger="potential_contradiction",
                        details={"encoder_intent": intent, "confidence": conf,
                                 "message": message[:200]},
                    )]
            # Fall through to pattern matching if encoder wasn't confident

        # ── Pattern fallback ──
        for pattern in FRUSTRATION_PATTERNS:
            if pattern in msg_lower:
                return [SignalEvent(
                    signal_type=SignalType.NOREPINEPHRINE,
                    trigger="user_frustration",
                    details={"pattern": pattern, "message": message[:200]},
                )]

        for pattern in CONTRADICTION_MARKERS:
            if pattern in msg_lower:
                return [SignalEvent(
                    signal_type=SignalType.NOREPINEPHRINE,
                    trigger="potential_contradiction",
                    details={"pattern": pattern, "message": message[:200]},
                )]

        return events

    def apply(self, event: SignalEvent, memory_store, embeddings=None,
              graph_store=None) -> None:
        # Effects are applied transiently in process(), not stored
        pass

    def detect_for_process(
        self, message: str, query_embedding: list[float], embeddings_engine, retrieval_config: RetrievalConfig
    ) -> list[SignalEvent]:
        """Called during process() to detect topic shifts and apply transient effects."""
        events = []

        # Topic shift detection
        if self._previous_embedding is not None:
            similarity = embeddings_engine.similarity(query_embedding, self._previous_embedding)
            if similarity < 0.3:
                events.append(
                    SignalEvent(
                        signal_type=SignalType.NOREPINEPHRINE,
                        trigger="topic_shift",
                        details={"similarity": similarity},
                    )
                )
                retrieval_config.widen()
                retrieval_config.add_caution(
                    "Abrupt topic change detected. Retrieving broader context."
                )

        self._previous_embedding = query_embedding

        return events

    def apply_observe_effects(self, events: list[SignalEvent], retrieval_config: RetrievalConfig) -> None:
        """Apply transient effects from observe() signals to the next process() call."""
        for event in events:
            if event.signal_type == SignalType.NOREPINEPHRINE:
                retrieval_config.widen()
                if event.trigger == "user_frustration":
                    retrieval_config.add_caution(
                        "User frustration detected. Double-check all claims carefully."
                    )
                elif event.trigger == "potential_contradiction":
                    retrieval_config.add_caution(
                        "Information may have changed. Verify against latest context."
                    )

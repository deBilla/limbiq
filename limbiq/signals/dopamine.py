"""
Dopamine Signal -- "This matters. Remember it."

Fires when something important happens that should be remembered
with high priority. Dopamine-tagged memories bypass normal retrieval
thresholds and are ALWAYS included in the context.
"""

from limbiq.signals.base import BaseSignal
from limbiq.types import (
    SignalEvent,
    SignalType,
    Memory,
    MemoryTier,
    SuppressionReason,
)

CORRECTION_PATTERNS = [
    "no that's wrong",
    "actually it's",
    "that's not right",
    "let me correct",
    "i didn't say",
    "that's incorrect",
    "no, my name is",
    "no, i",
    "wrong,",
    "not quite,",
    "close but",
    "almost but",
]

ENTHUSIASM_PATTERNS = [
    "exactly",
    "yes!",
    "that's it",
    "perfect",
    "brilliant",
    "spot on",
    "nailed it",
    "correct!",
    "you got it",
    "that's right",
    "precisely",
    "bingo",
]

PERSONAL_INFO_PATTERNS = [
    "my name is",
    "i work at",
    "i'm a ",
    "i live in",
    "my wife",
    "my husband",
    "my partner",
    "my kid",
    "my email",
    "my phone",
    "i'm from",
    "i'm based in",
    "i prefer",
    "i always",
    "i never",
    "i hate",
    "i love",
]


class DopamineSignal(BaseSignal):
    @property
    def signal_type(self) -> str:
        return SignalType.DOPAMINE

    def detect(
        self,
        message: str,
        response: str = None,
        feedback: str = None,
        memories: list[Memory] = None,
    ) -> list[SignalEvent]:
        events = []
        msg_lower = message.lower() if message else ""

        # Check explicit positive feedback
        if feedback and feedback.lower() == "positive":
            events.append(
                SignalEvent(
                    signal_type=SignalType.DOPAMINE,
                    trigger="explicit_positive_feedback",
                    details={"feedback": feedback},
                )
            )
            return events

        # Check for correction feedback
        if feedback and feedback.lower().startswith("correction:"):
            correction_content = feedback[len("correction:") :].strip()
            events.append(
                SignalEvent(
                    signal_type=SignalType.DOPAMINE,
                    trigger="user_correction",
                    details={"correction": correction_content, "original_message": message},
                )
            )
            return events

        # Check correction patterns
        for pattern in CORRECTION_PATTERNS:
            if pattern in msg_lower:
                events.append(
                    SignalEvent(
                        signal_type=SignalType.DOPAMINE,
                        trigger="user_correction",
                        details={"pattern": pattern, "message": message},
                    )
                )
                return events

        # Check enthusiasm patterns
        for pattern in ENTHUSIASM_PATTERNS:
            if pattern in msg_lower:
                events.append(
                    SignalEvent(
                        signal_type=SignalType.DOPAMINE,
                        trigger="user_enthusiasm",
                        details={"pattern": pattern, "message": message},
                    )
                )
                return events

        # Check personal info patterns
        for pattern in PERSONAL_INFO_PATTERNS:
            if pattern in msg_lower:
                events.append(
                    SignalEvent(
                        signal_type=SignalType.DOPAMINE,
                        trigger="novel_personal_info",
                        details={"pattern": pattern, "message": message},
                    )
                )
                return events

        return events

    def apply(self, event: SignalEvent, memory_store, embeddings=None) -> None:
        details = event.details
        message = details.get("message", "") or details.get("correction", "")

        if not message or not embeddings:
            return

        if event.trigger == "user_correction":
            # Store corrected info as priority
            embedding = embeddings.embed(message)
            mem = memory_store.store(
                content=message,
                tier=MemoryTier.PRIORITY,
                confidence=1.0,
                is_priority=True,
                source="correction",
                metadata={"trigger": event.trigger},
                embedding=embedding,
            )
            event.memory_ids_affected.append(mem.id)

            # Suppress contradictory memories
            related = memory_store.search(embedding, top_k=5)
            for m in related:
                if m.id != mem.id and not m.is_priority:
                    memory_store.suppress(m.id, SuppressionReason.CONTRADICTED)
                    event.memory_ids_affected.append(m.id)

        elif event.trigger == "novel_personal_info":
            embedding = embeddings.embed(message)
            mem = memory_store.store(
                content=message,
                tier=MemoryTier.PRIORITY,
                confidence=1.0,
                is_priority=True,
                source="conversation",
                metadata={"trigger": event.trigger},
                embedding=embedding,
            )
            event.memory_ids_affected.append(mem.id)

        elif event.trigger in ("user_enthusiasm", "explicit_positive_feedback"):
            # Boost confidence of recently discussed memories
            if embeddings and message:
                related = memory_store.search(embeddings.embed(message), top_k=3)
                for m in related:
                    memory_store.boost_confidence(m.id, min(m.confidence * 1.5, 1.0))
                    event.memory_ids_affected.append(m.id)

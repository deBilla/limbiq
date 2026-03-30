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
        encoder=None,
    ) -> list[SignalEvent]:
        events = []
        msg_lower = message.lower() if message else ""

        # Check explicit feedback (API-level, always first)
        if feedback and feedback.lower() == "positive":
            events.append(
                SignalEvent(
                    signal_type=SignalType.DOPAMINE,
                    trigger="explicit_positive_feedback",
                    details={"feedback": feedback},
                )
            )
            return events

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

        # ── Encoder-based detection ──
        # The unified encoder classifies intent from sentence context.
        # No hardcoded pattern fallback — if encoder isn't trained, signal doesn't fire.
        if encoder and encoder.available:
            result = encoder.classify_intent(message)
            if result:
                intent, conf = result
                if intent == "correction" and conf > 0.5:
                    return [SignalEvent(
                        signal_type=SignalType.DOPAMINE,
                        trigger="user_correction",
                        details={"encoder_intent": intent, "confidence": conf,
                                 "message": message},
                    )]
                if intent == "enthusiasm" and conf > 0.5:
                    return [SignalEvent(
                        signal_type=SignalType.DOPAMINE,
                        trigger="user_enthusiasm",
                        details={"encoder_intent": intent, "confidence": conf,
                                 "message": message},
                    )]
                if intent == "personal_info" and conf > 0.5:
                    return [SignalEvent(
                        signal_type=SignalType.DOPAMINE,
                        trigger="novel_personal_info",
                        details={"encoder_intent": intent, "confidence": conf,
                                 "message": message},
                    )]

        return events

    def apply(self, event: SignalEvent, memory_store, embeddings=None,
              graph_store=None) -> None:
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

            # Store correction as training pair for the relation classifier
            if graph_store is not None:
                self._store_correction_training_pair(message, graph_store)

        elif event.trigger == "novel_personal_info":
            embedding = embeddings.embed(message)

            # Deduplicate: skip if a similar priority memory already exists
            existing_priority = memory_store.search(embedding, top_k=3)
            for m in existing_priority:
                if m.is_priority:
                    sim = embeddings.similarity(embedding, embeddings.embed(m.content))
                    if sim > 0.92:
                        # Near-duplicate — skip storage, just boost existing
                        memory_store.boost_confidence(m.id, 1.0)
                        event.memory_ids_affected.append(m.id)
                        return

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

    @staticmethod
    def _store_correction_training_pair(message: str, graph_store):
        """Parse a correction message for entity-relation patterns and store
        as training pairs for the contextual relation classifier.

        Positive pair: the correct relation from the correction.
        Negative pair: the wrong relation being contradicted.
        """
        try:
            from limbiq.graph.entities import RELATION_PATTERNS
            import re

            for pattern_str, extractor in RELATION_PATTERNS:
                match = re.search(pattern_str, message)
                if match:
                    result = extractor(match)
                    if result is None:
                        continue
                    subj_indicator, predicate, obj_name = result
                    if subj_indicator == "user":
                        # Can't create a useful training pair for user-relations
                        # because "user" isn't an entity name in the sentence
                        continue

                    # Store as positive training pair
                    graph_store.store_relation_correction(
                        sentence=message,
                        subject_name=subj_indicator,
                        predicate=predicate,
                        object_name=obj_name,
                        is_positive=True,
                    )
                    break
        except Exception:
            pass  # Best effort — don't break the signal flow

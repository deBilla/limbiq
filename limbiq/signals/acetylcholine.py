"""
Acetylcholine Signal -- "Focus here. Build expertise."

Fires when the user shows sustained interest in a topic.
Creates and grows domain knowledge clusters.
"""

from collections import Counter

from limbiq.signals.base import BaseSignal
from limbiq.types import SignalEvent, SignalType, Memory, MemoryTier

DEPTH_PATTERNS = [
    "tell me more", "go deeper", "explain further", "more detail",
    "what about", "how exactly", "can you elaborate", "dig into",
    "let's explore", "break that down", "unpack that",
]

# Common stopwords to exclude from topic detection
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "i", "me", "my", "we", "you", "your", "he", "she", "it", "they",
    "them", "his", "her", "its", "our", "their", "this", "that", "these",
    "those", "what", "which", "who", "whom", "how", "when", "where", "why",
    "not", "no", "nor", "but", "and", "or", "if", "then", "so", "too",
    "very", "just", "about", "up", "out", "on", "off", "over", "under",
    "again", "further", "once", "here", "there", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "than", "also", "of", "in", "to", "for", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "because", "until", "while", "tell", "explain",
    "know", "think", "want", "like", "get", "make", "go", "see", "look",
    "give", "take", "find", "say", "let", "put", "keep", "still",
    "one", "two", "three", "much", "many", "new", "old", "first", "last",
    "good", "bad", "big", "small", "long", "short", "right", "wrong",
    "question", "answer", "thing", "things", "way", "time", "day",
    "something", "anything", "everything", "nothing", "another",
    "really", "always", "never", "maybe", "please", "thanks", "sure",
    "help", "work", "use", "using", "used", "try", "need", "come",
}

TOPIC_CONTINUITY_THRESHOLD = 3


class AcetylcholineSignal(BaseSignal):

    def __init__(self, llm_fn=None):
        self.llm_fn = llm_fn
        self._recent_topics: list[str] = []

    @property
    def signal_type(self) -> str:
        return SignalType.ACETYLCHOLINE

    def detect(
        self,
        message: str,
        response: str = None,
        feedback: str = None,
        memories: list[Memory] = None,
    ) -> list[SignalEvent]:
        # Detection happens in analyze_topic called from core
        return []

    def apply(self, event: SignalEvent, memory_store, embeddings=None) -> None:
        pass

    def analyze_topic(
        self,
        message: str,
        response: str,
        conversation_buffer: list[dict],
        cluster_store,
        memory_store,
        embeddings,
        llm_fn=None,
    ) -> list[SignalEvent]:
        """Detect topic focus and manage clusters."""
        events = []
        topic = self._detect_topic(message, conversation_buffer, llm_fn)
        if not topic:
            self._recent_topics.append(None)
            return events

        self._recent_topics.append(topic)
        # Keep only last 10
        if len(self._recent_topics) > 10:
            self._recent_topics = self._recent_topics[-10:]

        # Check for explicit depth request
        is_depth_request = any(p in message.lower() for p in DEPTH_PATTERNS)

        # Check topic continuity (3+ turns on same/similar topic)
        recent_non_none = [t for t in self._recent_topics[-TOPIC_CONTINUITY_THRESHOLD:] if t]
        is_sustained = False
        if len(recent_non_none) >= TOPIC_CONTINUITY_THRESHOLD:
            # Check if topics are semantically similar (not just exact match)
            if len(set(recent_non_none)) == 1:
                is_sustained = True
            else:
                # Check if all topics share common words (semantic approximation)
                topic_words = [set(t.split()) for t in recent_non_none]
                common = topic_words[0]
                for tw in topic_words[1:]:
                    common = common & tw
                if common:  # At least one word in common across all recent topics
                    is_sustained = True

        if is_sustained or is_depth_request:
            # Find or create cluster
            cluster = cluster_store.get_by_topic(topic)
            if not cluster:
                cluster = cluster_store.find_matching_cluster(topic)
            if not cluster:
                cluster = cluster_store.create_cluster(topic, f"Knowledge about {topic}")
                events.append(
                    SignalEvent(
                        signal_type=SignalType.ACETYLCHOLINE,
                        trigger="cluster_created",
                        details={"topic": topic, "cluster_id": cluster.id},
                    )
                )

            # Store the exchange as a memory and add to cluster
            content = f"[{topic}] User asked: {message[:200]}"
            embedding = embeddings.embed(content)
            mem = memory_store.store(
                content=content,
                tier=MemoryTier.MID,
                confidence=0.8,
                source="acetylcholine",
                metadata={"cluster_topic": topic, "cluster_id": cluster.id},
                embedding=embedding,
            )
            cluster_store.add_memory_to_cluster(cluster.id, mem.id)

            if response:
                resp_content = f"[{topic}] Key point: {response[:200]}"
                resp_embedding = embeddings.embed(resp_content)
                resp_mem = memory_store.store(
                    content=resp_content,
                    tier=MemoryTier.MID,
                    confidence=0.8,
                    source="acetylcholine",
                    metadata={"cluster_topic": topic, "cluster_id": cluster.id},
                    embedding=resp_embedding,
                )
                cluster_store.add_memory_to_cluster(cluster.id, resp_mem.id)

            events.append(
                SignalEvent(
                    signal_type=SignalType.ACETYLCHOLINE,
                    trigger="domain_focus",
                    details={
                        "topic": topic,
                        "cluster_id": cluster.id,
                        "is_depth_request": is_depth_request,
                        "is_sustained": is_sustained,
                    },
                )
            )

        return events

    def detect_topic_for_retrieval(
        self, message: str, conversation_history: list[dict], cluster_store, llm_fn=None
    ) -> tuple:
        """Called during process() to find matching clusters for retrieval."""
        topic = self._detect_topic(message, conversation_history, llm_fn)
        if not topic:
            return None, None

        cluster = cluster_store.get_by_topic(topic)
        if not cluster:
            cluster = cluster_store.find_matching_cluster(topic)

        if cluster:
            cluster_store.touch_cluster(cluster.id)
            memories = cluster_store.get_cluster_memories(cluster.id)
            return cluster, memories

        return None, None

    def _detect_topic(self, message: str, history: list[dict] = None, llm_fn=None) -> str | None:
        # Always try heuristic first — it's <1ms vs 500-2000ms for LLM.
        # Only fall back to LLM if heuristic finds nothing and LLM is available.
        result = self._detect_topic_heuristic(message)
        if result:
            return result
        if llm_fn:
            return self._detect_topic_llm(message, history, llm_fn)
        return None

    def _detect_topic_llm(self, message: str, history: list[dict], llm_fn) -> str | None:
        try:
            recent = ""
            if history:
                recent = "\n".join(
                    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:100]}"
                    for m in (history or [])[-6:]
                )
            prompt = (
                "What is the main topic of this conversation? "
                "Reply with just the topic name (1-3 words, lowercase). If no clear topic, reply NONE.\n\n"
            )
            if recent:
                prompt += f"Recent conversation:\n{recent}\n\n"
            prompt += f"Latest message: {message}"

            result = llm_fn(prompt).strip().lower()
            if result == "none" or len(result) > 50:
                return None
            return result
        except Exception:
            return None

    def _detect_topic_heuristic(self, message: str) -> str | None:
        """Extract the dominant topic using bigrams and word frequency."""
        words = message.lower().split()
        significant = [w.strip(".,!?;:'\"()[]{}") for w in words if len(w) > 2]
        significant = [w for w in significant if w and w not in _STOPWORDS]

        if not significant:
            return None

        # Try bigrams first (more meaningful topics)
        bigrams = []
        for i in range(len(significant) - 1):
            bigram = f"{significant[i]} {significant[i+1]}"
            bigrams.append(bigram)

        if bigrams:
            bigram_counts = Counter(bigrams)
            top_bigram = bigram_counts.most_common(1)
            if top_bigram and top_bigram[0][1] >= 1:
                return top_bigram[0][0]

        # Fallback to most common single word
        counts = Counter(significant)
        most_common = counts.most_common(1)
        if most_common and most_common[0][1] >= 1:
            return most_common[0][0]
        return None

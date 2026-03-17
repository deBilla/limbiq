"""
Context Builder -- assembles the enriched context string.

This is what gets injected into the system prompt.
"""

from limbiq.types import Memory, BehavioralRule


class ContextBuilder:
    def __init__(self, max_tokens: int = 1500):
        self.max_tokens = max_tokens

    def build(
        self,
        priority_memories: list[Memory],
        relevant_memories: list[Memory],
        suppressed_ids: set[str],
        active_rules: list[BehavioralRule] = None,
        cluster_memories: list[Memory] = None,
        caution_flag: str = None,
    ) -> str:
        sections = []

        # Caution flag comes FIRST
        if caution_flag:
            sections.append(
                f"[CAUTION: {caution_flag}]\n"
                "Double-check your claims against the provided memory context. "
                "If you're not sure about something, say so explicitly."
            )

        # Behavioral rules shape how the model uses everything else
        if active_rules:
            rules_text = "\n".join(f"  - {rule.rule_text}" for rule in active_rules)
            sections.append(
                f"[BEHAVIORAL RULES -- follow these for this user]\n{rules_text}"
            )

        if priority_memories:
            facts = "\n".join(f"  - {m.content}" for m in priority_memories)
            sections.append(
                f"[IMPORTANT -- known facts about this user]\n{facts}"
            )

        # Domain knowledge clusters
        if cluster_memories:
            # Group by cluster topic
            topics_seen = set()
            for m in cluster_memories:
                topic = m.metadata.get("cluster_topic", "this topic")
                if topic not in topics_seen:
                    topics_seen.add(topic)
            cluster_facts = "\n".join(f"  - {m.content}" for m in cluster_memories)
            topic_label = ", ".join(topics_seen) if topics_seen else "this topic"
            sections.append(
                f"[DOMAIN KNOWLEDGE -- accumulated knowledge about {topic_label}]\n{cluster_facts}"
            )

        if relevant_memories:
            priority_ids = {m.id for m in priority_memories}
            cluster_ids = {m.id for m in (cluster_memories or [])}
            filtered = [
                m
                for m in relevant_memories
                if m.id not in priority_ids
                and m.id not in suppressed_ids
                and m.id not in cluster_ids
            ]

            if filtered:
                items = "\n".join(
                    f"  - [{m.confidence:.0%} confidence] {m.content}"
                    for m in filtered
                )
                sections.append(
                    f"[Relevant memories from previous conversations]\n{items}"
                )

        if not sections:
            return ""

        context = "\n\n".join(sections)

        return (
            "<memory_context>\n"
            "The following is stored information from previous interactions with this user.\n"
            "Use it naturally when relevant. If the user asks about something listed here,\n"
            "reference it confidently. If something is NOT listed here, say you don't know\n"
            "rather than guessing.\n\n"
            f"{context}\n"
            "</memory_context>"
        )

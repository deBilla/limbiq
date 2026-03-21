"""
Context Builder -- assembles the enriched context string.

Graph-aware: when the knowledge graph provides answers or a world summary,
those are used first (compact, ~40 tokens). Raw memories are only included
for information the graph doesn't cover, avoiding duplication.
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
        graph_answer: str = None,
        world_summary: str = None,
        search_results: list = None,
    ) -> str:
        sections = []

        # Caution flag comes FIRST
        if caution_flag:
            sections.append(
                f"[CAUTION: {caution_flag}]\n"
                "Double-check your claims against the provided memory context. "
                "If you're not sure about something, say so explicitly."
            )

        # Graph answer — the most token-efficient knowledge representation
        # e.g. "Upananda is your father and Prabhashi's father-in-law." = 15 tokens
        if graph_answer:
            sections.append(f"[KNOWN FACT] {graph_answer}")

        # Compact world summary from graph — replaces raw memory dump
        # e.g. "Your father is Upananda. Your wife is Prabhashi." = 20 tokens
        if world_summary:
            sections.append(f"[ABOUT YOU] {world_summary}")

        # Web search results — fresh knowledge from the internet
        if search_results:
            from urllib.parse import urlparse
            items = []
            for sr in search_results[:3]:
                domain = urlparse(sr.url).netloc if sr.url else sr.source
                items.append(f'  - "{sr.title}" ({domain}): {sr.snippet}')
            sections.append(
                "[WEB SEARCH — recent information from the internet]\n"
                + "\n".join(items)
            )

        # Behavioral rules (serotonin)
        if active_rules:
            rules_text = "; ".join(rule.rule_text for rule in active_rules)
            sections.append(f"[STYLE] {rules_text}")

        # Priority memories that AREN'T already covered by the graph.
        # When the graph has a world summary, most priority memories are
        # redundant (they were the source of graph entities). Only include
        # ones the graph doesn't know about.
        if priority_memories:
            world_lower = (world_summary or "").lower()
            graph_lower = (graph_answer or "").lower()
            combined_lower = world_lower + " " + graph_lower
            ungraphed = [
                m for m in priority_memories
                if not self._is_covered_by_summary(m.content, combined_lower)
            ]
            if ungraphed:
                facts = "\n".join(f"  - {m.content}" for m in ungraphed[:3])
                sections.append(
                    f"[IMPORTANT -- known facts about this user]\n{facts}"
                )

        # Domain knowledge clusters
        if cluster_memories:
            topics_seen = set()
            for m in cluster_memories:
                topic = m.metadata.get("cluster_topic", "this topic")
                topics_seen.add(topic)
            cluster_facts = "\n".join(f"  - {m.content}" for m in cluster_memories)
            topic_label = ", ".join(topics_seen) if topics_seen else "this topic"
            sections.append(
                f"[DOMAIN KNOWLEDGE -- {topic_label}]\n{cluster_facts}"
            )

        # Relevant memories not already covered by graph
        if relevant_memories:
            priority_ids = {m.id for m in priority_memories}
            cluster_ids = {m.id for m in (cluster_memories or [])}
            combined_lower = ((world_summary or "") + " " + (graph_answer or "")).lower()
            filtered = [
                m
                for m in relevant_memories
                if m.id not in priority_ids
                and m.id not in suppressed_ids
                and m.id not in cluster_ids
                and not self._is_covered_by_summary(m.content, combined_lower)
            ]

            # When graph provides context, cap relevant memories aggressively
            max_relevant = 2 if (world_summary or graph_answer) else 5
            if filtered:
                items = "\n".join(
                    f"  - [{m.confidence:.0%} confidence] {m.content}"
                    for m in filtered[:max_relevant]
                )
                sections.append(
                    f"[Relevant memories from previous conversations]\n{items}"
                )

        if not sections:
            return ""

        context = "\n\n".join(sections)

        return (
            "<memory_context>\n"
            f"{context}\n"
            "</memory_context>"
        )

    @staticmethod
    def _is_covered_by_summary(memory_content: str, world_summary_lower: str) -> bool:
        """Check if a memory's key content is already in the world summary."""
        if not world_summary_lower:
            return False
        # Extract the main proper noun / value from the memory
        # Simple heuristic: check if significant words overlap
        words = memory_content.split()
        # Look for capitalized words (proper nouns) — those are the key info
        proper_nouns = [w.strip(".,!?'\"") for w in words if w[0:1].isupper() and len(w) > 2]
        if not proper_nouns:
            return False
        # If most proper nouns are in the summary, it's covered
        found = sum(1 for pn in proper_nouns if pn.lower() in world_summary_lower)
        return found >= len(proper_nouns) * 0.5

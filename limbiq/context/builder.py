"""
Context Builder -- assembles the enriched context string.

This is what gets injected into the system prompt.
"""

from limbiq.types import Memory


class ContextBuilder:
    def __init__(self, max_tokens: int = 1500):
        self.max_tokens = max_tokens

    def build(
        self,
        priority_memories: list[Memory],
        relevant_memories: list[Memory],
        suppressed_ids: set[str],
    ) -> str:
        sections = []

        if priority_memories:
            facts = "\n".join(f"  - {m.content}" for m in priority_memories)
            sections.append(
                f"[IMPORTANT -- known facts about this user]\n{facts}"
            )

        if relevant_memories:
            priority_ids = {m.id for m in priority_memories}
            filtered = [
                m
                for m in relevant_memories
                if m.id not in priority_ids and m.id not in suppressed_ids
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

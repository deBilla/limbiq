"""
Memory Compressor -- the lossy compression pipeline.

Full conversations -> gists (mid-term) -> abstract facts (long-term)
"""


class MemoryCompressor:
    def __init__(self, llm_fn=None):
        self.llm_fn = llm_fn

    def compress_conversation(self, messages: list[dict]) -> list[str]:
        if self.llm_fn:
            return self._llm_compress(messages)
        else:
            return self._extractive_compress(messages)

    def _llm_compress(self, messages: list[dict]) -> list[str]:
        conversation_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in messages
            if m.get("role") != "system"
        )

        prompt = (
            "Extract individual, self-contained facts from this conversation.\n"
            "Each fact should be on its own line. Each fact should be independently searchable.\n\n"
            "Rules:\n"
            "- One fact per line, no bullets or numbering\n"
            "- Each fact must be understandable without the others\n"
            "- Include personal details the user shared (name, job, preferences, family)\n"
            "- Include key topics discussed and conclusions reached\n"
            "- Skip greetings, filler, and meta-conversation\n"
            "- If no meaningful facts exist, respond with NONE\n\n"
            f"Conversation:\n{conversation_text}"
        )

        result = self.llm_fn(prompt)

        if "NONE" in result.upper():
            return []

        facts = [f.strip() for f in result.strip().split("\n") if len(f.strip()) > 10]
        return facts

    def _extractive_compress(self, messages: list[dict]) -> list[str]:
        """Fallback: extract key sentences from user messages."""
        personal_indicators = [
            "my name",
            "i work",
            "i'm a",
            "i live",
            "i prefer",
            "my wife",
            "my husband",
            "my partner",
            "i'm from",
            "i'm based",
            "i love",
            "i hate",
            "my kid",
            "my email",
            "my phone",
        ]

        facts = []
        for m in messages:
            if m.get("role") == "user":
                content = m["content"].strip()
                for sentence in content.replace("!", ".").replace("?", ".").split("."):
                    sentence = sentence.strip()
                    if len(sentence) > 15 and any(
                        p in sentence.lower() for p in personal_indicators
                    ):
                        facts.append(sentence)
        return facts

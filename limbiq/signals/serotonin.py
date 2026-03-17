"""
Serotonin Signal -- "This pattern is stable. Make it permanent."

Watches for recurring patterns in user behavior and preferences.
When a pattern appears 3+ times across 2+ different sessions,
crystallizes it into a permanent behavioral rule.
"""

from limbiq.signals.base import BaseSignal
from limbiq.types import SignalEvent, SignalType, Memory

CASUAL_MARKERS = [
    "yeah", "nah", "tbh", "lol", "haha", "gonna", "wanna", "kinda",
    "yep", "nope", "btw", "imo", "idk",
]

FORMAL_MARKERS = [
    "please", "kindly", "would you", "could you", "i would appreciate",
    "regarding", "furthermore", "therefore", "consequently",
]

FOLLOW_UP_MARKERS = [
    "tell me more", "go deeper", "why", "how exactly", "can you explain",
    "elaborate", "what do you mean", "expand on",
]

SIMPLIFY_MARKERS = [
    "simpler", "simplify", "too complicated", "explain like", "in plain",
    "keep it simple", "less technical", "dumb it down",
]

DETAIL_MARKERS = [
    "more detail", "go deeper", "elaborate", "be more specific",
    "full explanation", "thorough", "comprehensive",
]

RULE_TEMPLATES = {
    "prefers_concise": "Keep responses brief and to the point. This user prefers concise communication.",
    "prefers_detailed": "Provide thorough, detailed responses. This user appreciates depth.",
    "casual_tone": "Use a casual, conversational tone. This user communicates informally.",
    "formal_tone": "Maintain a professional, formal tone in responses.",
    "wants_code_examples": "Include code examples when explaining technical concepts.",
    "asks_followups": "Proactively provide deeper context -- this user likes to dig into details.",
    "wants_simplicity": "Keep explanations simple and accessible. Avoid jargon unless asked.",
    "skips_pleasantries": "Skip greetings and pleasantries. Get straight to the answer.",
    "explains_reasoning": "Proactively explain your reasoning -- this user frequently asks 'why'.",
}

CRYSTALLIZATION_THRESHOLD = 3
MIN_SESSIONS_THRESHOLD = 2


class SerotoninSignal(BaseSignal):

    def __init__(self, llm_fn=None):
        self.llm_fn = llm_fn

    @property
    def signal_type(self) -> str:
        return SignalType.SEROTONIN

    def detect(
        self,
        message: str,
        response: str = None,
        feedback: str = None,
        memories: list[Memory] = None,
    ) -> list[SignalEvent]:
        # Serotonin detection happens in analyze_and_track, not here.
        # detect() is called by the core signal loop but serotonin
        # needs the rule_store and session_id, so we return empty
        # and do real work via analyze_and_track called from core.
        return []

    def apply(self, event: SignalEvent, memory_store, embeddings=None) -> None:
        # Crystallization is handled directly in analyze_and_track
        pass

    def analyze_and_track(
        self,
        message: str,
        response: str,
        session_id: str,
        rule_store,
        llm_fn=None,
    ) -> list[SignalEvent]:
        """Analyze exchange for patterns, track observations, crystallize if threshold met."""
        events = []
        patterns = self._analyze_patterns(message, response, llm_fn)

        for pattern in patterns:
            pattern_key = pattern["pattern_key"]

            # Skip if already crystallized
            if rule_store.rule_exists(pattern_key):
                continue

            rule_store.add_observation(
                category=pattern["category"],
                pattern_key=pattern_key,
                observation=pattern["observation"],
                session_id=session_id or "default",
            )

            obs_count = rule_store.get_observation_count(pattern_key)
            session_count = rule_store.get_distinct_session_count(pattern_key)

            if obs_count >= CRYSTALLIZATION_THRESHOLD and session_count >= MIN_SESSIONS_THRESHOLD:
                rule_text = self._crystallize(pattern_key, rule_store.get_observations(pattern_key), llm_fn)
                rule = rule_store.create_rule(pattern_key, rule_text, obs_count)

                events.append(
                    SignalEvent(
                        signal_type=SignalType.SEROTONIN,
                        trigger="pattern_crystallized",
                        details={
                            "pattern_key": pattern_key,
                            "rule_text": rule_text,
                            "observation_count": obs_count,
                            "session_count": session_count,
                            "rule_id": rule.id,
                        },
                    )
                )

        return events

    def _analyze_patterns(self, message: str, response: str, llm_fn=None) -> list[dict]:
        if llm_fn:
            result = self._analyze_patterns_llm(message, response, llm_fn)
            if result:
                return result
        return self._analyze_patterns_heuristic(message, response)

    def _analyze_patterns_llm(self, message: str, response: str, llm_fn) -> list[dict]:
        prompt = (
            "Analyze this conversation exchange for recurring user patterns.\n\n"
            f"User message: {message}\n"
            f"Assistant response: {response[:200]}\n\n"
            "Look for:\n"
            "- Communication style (formal/casual, brief/verbose, technical/simple)\n"
            "- Content preferences (wants detail, wants examples, wants code)\n"
            "- Interaction habits (asks follow-ups, jumps topics, builds on previous)\n\n"
            "For each pattern detected, output one line in this format:\n"
            "PATTERN: category | pattern_key | observation\n\n"
            "Valid pattern_keys: prefers_concise, prefers_detailed, casual_tone, formal_tone, "
            "wants_code_examples, asks_followups, wants_simplicity, skips_pleasantries, explains_reasoning\n\n"
            "If no clear patterns, respond NONE."
        )
        try:
            result = llm_fn(prompt)
            if not result or "NONE" in result.strip().upper():
                return []
            patterns = []
            for line in result.strip().split("\n"):
                if line.strip().upper().startswith("PATTERN:"):
                    parts = line.split("|")
                    if len(parts) >= 3:
                        category = parts[0].replace("PATTERN:", "").strip().lower()
                        pattern_key = parts[1].strip().lower().replace(" ", "_")
                        observation = parts[2].strip()
                        if pattern_key in RULE_TEMPLATES:
                            patterns.append({
                                "category": category,
                                "pattern_key": pattern_key,
                                "observation": observation,
                            })
            return patterns
        except Exception:
            return []

    def _analyze_patterns_heuristic(self, message: str, response: str) -> list[dict]:
        patterns = []
        msg_lower = message.lower()
        word_count = len(message.split())

        # Message length as style signal
        if word_count < 10:
            patterns.append({
                "category": "style",
                "pattern_key": "prefers_concise",
                "observation": f"Short message: {word_count} words",
            })
        elif word_count > 50:
            patterns.append({
                "category": "style",
                "pattern_key": "prefers_detailed",
                "observation": f"Detailed message: {word_count} words",
            })

        # Casual language
        if any(m in msg_lower.split() for m in CASUAL_MARKERS):
            patterns.append({
                "category": "style",
                "pattern_key": "casual_tone",
                "observation": "Uses casual language markers",
            })

        # Formal language
        if any(m in msg_lower for m in FORMAL_MARKERS):
            patterns.append({
                "category": "style",
                "pattern_key": "formal_tone",
                "observation": "Uses formal language markers",
            })

        # Code interest
        if "code" in msg_lower or "example" in msg_lower or "```" in message:
            patterns.append({
                "category": "preference",
                "pattern_key": "wants_code_examples",
                "observation": "Requested or used code",
            })

        # Follow-up pattern
        if any(m in msg_lower for m in FOLLOW_UP_MARKERS):
            patterns.append({
                "category": "interaction",
                "pattern_key": "asks_followups",
                "observation": "Asked a follow-up question",
            })

        # Simplify requests
        if any(m in msg_lower for m in SIMPLIFY_MARKERS):
            patterns.append({
                "category": "preference",
                "pattern_key": "wants_simplicity",
                "observation": "Requested simpler explanation",
            })

        # Detail requests
        if any(m in msg_lower for m in DETAIL_MARKERS):
            patterns.append({
                "category": "preference",
                "pattern_key": "prefers_detailed",
                "observation": "Requested more detail",
            })

        return patterns

    def _crystallize(self, pattern_key: str, observations: list[dict], llm_fn=None) -> str:
        if llm_fn:
            try:
                obs_text = "\n".join(f"- {o['observation']}" for o in observations[:5])
                prompt = (
                    "These observations about a user have been recorded across multiple conversations:\n\n"
                    f"Pattern: {pattern_key}\n"
                    f"Observations:\n{obs_text}\n\n"
                    "Write a single, clear behavioral instruction for an AI assistant interacting with this user.\n"
                    "The instruction should be actionable and specific.\n\n"
                    "Format: Just the instruction, one sentence, no preamble."
                )
                result = llm_fn(prompt).strip().strip('"')
                if result and len(result) > 10:
                    return result
            except Exception:
                pass

        return RULE_TEMPLATES.get(pattern_key, f"Observed recurring pattern: {pattern_key}")

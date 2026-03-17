"""Tests for the Serotonin signal -- pattern crystallization."""

from limbiq import Limbiq
from limbiq.signals.serotonin import SerotoninSignal, RULE_TEMPLATES


class TestSerotoninDetection:
    def test_heuristic_detects_concise_style(self):
        signal = SerotoninSignal()
        patterns = signal._analyze_patterns_heuristic("fix bug", "I'll help...")
        keys = [p["pattern_key"] for p in patterns]
        assert "prefers_concise" in keys

    def test_heuristic_detects_casual_tone(self):
        signal = SerotoninSignal()
        patterns = signal._analyze_patterns_heuristic("yeah just do it lol", "Sure!")
        keys = [p["pattern_key"] for p in patterns]
        assert "casual_tone" in keys

    def test_heuristic_detects_code_preference(self):
        signal = SerotoninSignal()
        patterns = signal._analyze_patterns_heuristic(
            "show me a code example", "Here's the code..."
        )
        keys = [p["pattern_key"] for p in patterns]
        assert "wants_code_examples" in keys

    def test_heuristic_detects_followups(self):
        signal = SerotoninSignal()
        patterns = signal._analyze_patterns_heuristic(
            "can you explain why that works?", "Because..."
        )
        keys = [p["pattern_key"] for p in patterns]
        assert "asks_followups" in keys

    def test_heuristic_detects_formal_tone(self):
        signal = SerotoninSignal()
        patterns = signal._analyze_patterns_heuristic(
            "Could you kindly provide the information regarding the matter?",
            "Certainly..."
        )
        keys = [p["pattern_key"] for p in patterns]
        assert "formal_tone" in keys

    def test_no_patterns_on_generic_message(self):
        signal = SerotoninSignal()
        # 10+ words, no casual/formal markers, no code/followup
        patterns = signal._analyze_patterns_heuristic(
            "This message has exactly eleven words in total right now today",
            "Response..."
        )
        # Should not detect concise (11 words) or detailed (< 50)
        keys = [p["pattern_key"] for p in patterns]
        assert "prefers_concise" not in keys
        assert "casual_tone" not in keys


class TestSerotoninCrystallization:
    def test_crystallization_threshold(self, tmp_dir):
        """Rules only crystallize after 3+ observations across 2+ sessions."""
        lq = Limbiq(store_path=tmp_dir, user_id="test")

        # Session 1: 2 observations
        lq.start_session()
        lq.observe("yeah fix it", "Sure...")
        lq.observe("nah skip that", "Okay...")
        lq.end_session()

        assert len(lq.get_active_rules()) == 0

        # Session 2: 3rd observation -- should crystallize (3 obs, 2 sessions)
        lq.start_session()
        lq.observe("tbh just do it", "Done...")
        lq.end_session()

        rules = lq.get_active_rules()
        assert len(rules) >= 1
        assert any("casual" in r.rule_text.lower() for r in rules)

    def test_pattern_detection_across_sessions(self, tmp_dir):
        """Patterns are detected from conversation exchanges across sessions."""
        lq = Limbiq(store_path=tmp_dir, user_id="test")

        for _ in range(3):
            lq.start_session()
            lq.observe("fix bug", "I'll help with that bug...")
            lq.observe("how", "Here's how...")
            lq.end_session()

        rules = lq.get_active_rules()
        assert any(
            "concise" in r.rule_text.lower() or "brief" in r.rule_text.lower()
            for r in rules
        )

    def test_rules_appear_in_context(self, tmp_dir):
        """Active rules are included in process() context output."""
        lq = Limbiq(store_path=tmp_dir, user_id="test")

        # Trigger crystallization
        for _ in range(3):
            lq.start_session()
            lq.observe("yeah do it", "Done...")
            lq.end_session()

        assert len(lq.get_active_rules()) >= 1

        lq.start_session()
        result = lq.process("hello")
        assert "BEHAVIORAL RULES" in result.context

    def test_rules_are_deactivatable(self, tmp_dir):
        """Rules can be deactivated if they're wrong."""
        lq = Limbiq(store_path=tmp_dir, user_id="test")

        for _ in range(3):
            lq.start_session()
            lq.observe("yeah cool", "Great!")
            lq.end_session()

        rules = lq.get_active_rules()
        assert len(rules) >= 1

        rule_id = rules[0].id
        lq.deactivate_rule(rule_id)

        assert rule_id not in [r.id for r in lq.get_active_rules()]

    def test_rules_are_reactivatable(self, tmp_dir):
        """Deactivated rules can be reactivated."""
        lq = Limbiq(store_path=tmp_dir, user_id="test")

        for _ in range(3):
            lq.start_session()
            lq.observe("yeah cool", "Great!")
            lq.end_session()

        rules = lq.get_active_rules()
        rule_id = rules[0].id

        lq.deactivate_rule(rule_id)
        assert rule_id not in [r.id for r in lq.get_active_rules()]

        lq.reactivate_rule(rule_id)
        assert rule_id in [r.id for r in lq.get_active_rules()]

    def test_no_duplicate_crystallization(self, tmp_dir):
        """Same pattern shouldn't crystallize twice."""
        lq = Limbiq(store_path=tmp_dir, user_id="test")

        for _ in range(5):
            lq.start_session()
            lq.observe("yeah ok", "Sure...")
            lq.end_session()

        rules = lq.get_active_rules()
        casual_rules = [r for r in rules if r.pattern_key == "casual_tone"]
        assert len(casual_rules) <= 1

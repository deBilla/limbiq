from limbiq.signals.dopamine import DopamineSignal


class TestDopamineDetection:
    def setup_method(self):
        self.signal = DopamineSignal()

    def test_detects_correction(self):
        events = self.signal.detect(
            message="No that's wrong, I work at Bitsmedia not Google",
            response=None,
            feedback=None,
            memories=[],
        )
        assert len(events) > 0
        assert events[0].trigger == "user_correction"

    def test_detects_enthusiasm(self):
        events = self.signal.detect(
            message="Exactly! That's perfect!",
            response=None,
            feedback=None,
            memories=[],
        )
        assert len(events) > 0
        assert events[0].trigger == "user_enthusiasm"

    def test_detects_personal_info(self):
        events = self.signal.detect(
            message="My wife's name is Prabhashi and she works at a bank",
            response=None,
            feedback=None,
            memories=[],
        )
        assert len(events) > 0
        assert events[0].trigger == "novel_personal_info"

    def test_explicit_positive_feedback(self):
        events = self.signal.detect(
            message="",
            response="",
            feedback="positive",
            memories=[],
        )
        assert len(events) > 0
        assert events[0].trigger == "explicit_positive_feedback"

    def test_correction_feedback(self):
        events = self.signal.detect(
            message="some message",
            response="some response",
            feedback="correction:User works at Bitsmedia",
            memories=[],
        )
        assert len(events) > 0
        assert events[0].trigger == "user_correction"

    def test_no_signal_on_generic_message(self):
        events = self.signal.detect(
            message="What's the weather like today?",
            response=None,
            feedback=None,
            memories=[],
        )
        assert len(events) == 0

from limbiq.detection.patterns import (
    is_correction,
    is_enthusiasm,
    is_personal_info,
    is_denial,
    match_any,
)


class TestPatternMatching:
    def test_correction_patterns(self):
        assert is_correction("No that's wrong, it should be X")
        assert is_correction("Actually it's different")
        assert not is_correction("What is the weather?")

    def test_enthusiasm_patterns(self):
        assert is_enthusiasm("Exactly! That's what I meant")
        assert is_enthusiasm("Yes! Perfect!")
        assert not is_enthusiasm("I think so maybe")

    def test_personal_info_patterns(self):
        assert is_personal_info("My name is Dimuthu")
        assert is_personal_info("I work at Bitsmedia")
        assert is_personal_info("My wife is Prabhashi")
        assert not is_personal_info("What time is it?")

    def test_denial_patterns(self):
        assert is_denial("I never said that about me")
        assert is_denial("You're making that up")
        assert not is_denial("Tell me about Python")

    def test_match_any_returns_pattern(self):
        result = match_any("My name is Bob", ["my name is", "i work at"])
        assert result == "my name is"

    def test_match_any_returns_none(self):
        result = match_any("Hello world", ["my name is"])
        assert result is None

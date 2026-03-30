import os
import tempfile
import shutil

import pytest

from limbiq import Limbiq


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def lq(tmp_dir):
    """A fresh Limbiq instance using TF-IDF fallback (no sentence-transformers needed)."""
    return Limbiq(store_path=tmp_dir, user_id="test")


class MockEncoder:
    """Mock encoder for signal unit tests — returns pre-configured intents."""

    available = True

    def __init__(self, intent_map: dict[str, tuple[str, float]] | None = None):
        self._intent_map = intent_map or {}

    def classify_intent(self, message: str):
        msg_lower = message.lower() if message else ""
        for keyword, result in self._intent_map.items():
            if keyword in msg_lower:
                return result
        return None


@pytest.fixture
def mock_encoder():
    """Create a MockEncoder with custom intent mappings."""
    return MockEncoder


@pytest.fixture
def mock_llm():
    """A mock LLM function for compression tests."""

    def _llm(prompt: str) -> str:
        # Extract facts from the conversation in the prompt
        if "Dimuthu" in prompt or "Bitsmedia" in prompt:
            return "User's name is Dimuthu\nUser works at Bitsmedia"

        # Serotonin pattern analysis mock
        if "recurring user patterns" in prompt.lower() or "analyze this conversation" in prompt.lower():
            return "NONE"

        # Serotonin crystallization mock
        if "behavioral instruction" in prompt.lower():
            return "Keep responses brief and to the point."

        # Acetylcholine topic detection mock
        if "main topic" in prompt.lower():
            if "attention" in prompt.lower() or "transformer" in prompt.lower():
                return "attention"
            if "rust" in prompt.lower():
                return "rust"
            return "NONE"

        return "NONE"

    return _llm

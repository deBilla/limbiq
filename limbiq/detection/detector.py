"""Main signal detector -- analyzes input and determines which signals to fire."""

from limbiq.types import Memory, SignalEvent
from limbiq.signals.base import BaseSignal


class SignalDetector:
    def __init__(self, signals: list[BaseSignal]):
        self.signals = signals

    def detect_all(
        self,
        message: str,
        response: str = None,
        feedback: str = None,
        memories: list[Memory] = None,
    ) -> list[SignalEvent]:
        """Run all signal detectors and return combined events."""
        events = []
        for signal in self.signals:
            detected = signal.detect(
                message=message,
                response=response,
                feedback=feedback,
                memories=memories,
            )
            events.extend(detected)
        return events

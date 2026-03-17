from abc import ABC, abstractmethod
from limbiq.types import SignalEvent, Memory


class BaseSignal(ABC):
    """Base class for all neurotransmitter signals."""

    @property
    @abstractmethod
    def signal_type(self) -> str:
        pass

    @abstractmethod
    def detect(
        self,
        message: str,
        response: str = None,
        feedback: str = None,
        memories: list[Memory] = None,
    ) -> list[SignalEvent]:
        pass

    @abstractmethod
    def apply(self, event: SignalEvent, memory_store, embeddings=None) -> None:
        pass

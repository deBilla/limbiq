from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time
import uuid


class MemoryTier(str, Enum):
    SHORT = "short"
    MID = "mid"
    LONG = "long"
    PRIORITY = "priority"


class SignalType(str, Enum):
    DOPAMINE = "dopamine"
    GABA = "gaba"


class SuppressionReason(str, Enum):
    USER_DENIED = "user_denied"
    STALE = "stale"
    NEVER_ACCESSED = "never_accessed"
    CONTRADICTED = "contradicted"
    CONFABULATION = "confabulation"
    MANUAL = "manual"


@dataclass
class Memory:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    tier: MemoryTier = MemoryTier.SHORT
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    session_count: int = 0
    access_count: int = 0
    is_priority: bool = False
    is_suppressed: bool = False
    suppression_reason: Optional[str] = None
    source: str = "conversation"
    metadata: dict = field(default_factory=dict)


@dataclass
class SignalEvent:
    signal_type: SignalType
    trigger: str
    timestamp: float = field(default_factory=time.time)
    details: dict = field(default_factory=dict)
    memory_ids_affected: list[str] = field(default_factory=list)


@dataclass
class ProcessResult:
    context: str
    signals_fired: list[SignalEvent]
    memories_retrieved: int
    priority_count: int
    suppressed_count: int

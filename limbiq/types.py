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
    SEROTONIN = "serotonin"
    ACETYLCHOLINE = "acetylcholine"
    NOREPINEPHRINE = "norepinephrine"


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
class BehavioralRule:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_key: str = ""
    rule_text: str = ""
    confidence: float = 1.0
    observation_count: int = 0
    created_at: float = field(default_factory=time.time)
    is_active: bool = True


@dataclass
class KnowledgeCluster:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    description: str = ""
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    memory_ids: list[str] = field(default_factory=list)


@dataclass
class RetrievalConfig:
    """Dynamic retrieval parameters -- modified by norepinephrine."""
    top_k: int = 10
    relevance_threshold: float = 0.15
    caution_flag: Optional[str] = None

    def widen(self):
        self.top_k = min(self.top_k * 2, 30)
        self.relevance_threshold *= 0.5

    def add_caution(self, reason: str):
        self.caution_flag = reason

    def reset(self):
        self.top_k = 10
        self.relevance_threshold = 0.15
        self.caution_flag = None


@dataclass
class ProcessResult:
    context: str
    signals_fired: list[SignalEvent]
    memories_retrieved: int
    priority_count: int
    suppressed_count: int
    active_rules: list[BehavioralRule] = field(default_factory=list)
    clusters_loaded: list[str] = field(default_factory=list)
    norepinephrine_active: bool = False

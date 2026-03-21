"""Pydantic models for the playground API."""

from pydantic import BaseModel, Field
from typing import Optional, Any


# ─── Requests ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ObserveRequest(BaseModel):
    message: str
    response: str
    feedback: Optional[str] = None


class TrainRequest(BaseModel):
    epochs: int = 500
    model_dir: Optional[str] = None


# ─── Graph Models ─────────────────────────────────────────────

class EntityModel(BaseModel):
    id: str
    name: str
    type: str
    relation_count: int = 0


class RelationModel(BaseModel):
    id: str
    subject_id: str
    subject_name: str
    predicate: str
    object_id: str
    object_name: str
    confidence: float
    is_inferred: bool


class GraphNetworkModel(BaseModel):
    nodes: list[dict[str, Any]]
    links: list[dict[str, Any]]
    stats: dict[str, int]


# ─── Responses ────────────────────────────────────────────────

class ProcessResponse(BaseModel):
    context: str = ""
    memories_retrieved: int = 0
    world_summary: str = ""
    duration_ms: float = 0
    signals_fired: list[dict[str, Any]] = []


class GraphQueryResponse(BaseModel):
    answered: bool
    answer: Optional[str] = None
    confidence: float = 0
    source: Optional[str] = None
    duration_ms: float = 0


class ReasonResponse(BaseModel):
    answered: bool
    answer: str = ""
    confidence: float = 0
    answer_mode: str = ""
    reasoning_trace: str = ""
    duration_ms: float = 0


class StatsResponse(BaseModel):
    entities: int = 0
    relations: int = 0
    explicit_relations: int = 0
    inferred_relations: int = 0
    memories: int = 0
    signals_fired: int = 0
    uptime_seconds: float = 0


class ProfileResponse(BaseModel):
    user_name: str = ""
    world_summary: str = ""
    entities: list[EntityModel] = []
    relations: list[RelationModel] = []
    memory_count: int = 0
    priority_facts: list[str] = []


class TrainResponse(BaseModel):
    status: str = "ok"
    samples: int = 0
    best_eval_acc: float = 0
    epochs: int = 0
    duration_ms: float = 0


class SignalEvent(BaseModel):
    signal_type: str
    trigger: str
    timestamp: float
    details: dict[str, Any] = {}

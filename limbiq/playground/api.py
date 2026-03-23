"""
REST API endpoints for Limbiq Playground.

All endpoints are prefixed with /api/v1/.

Focused on: signals + graph generation + self-healing connectivity.
"""

import time
import logging
from typing import Optional

from fastapi import APIRouter, Query, Request, HTTPException

from limbiq.playground.data_models import (
    QueryRequest, ObserveRequest, TrainRequest, TrainEncoderRequest,
    ProcessResponse, GraphQueryResponse, ReasonResponse,
    StatsResponse, ProfileResponse, TrainResponse,
    EntityModel, RelationModel, GraphNetworkModel, SignalEventModel,
    ConnectivityResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_lq(request: Request):
    """Get the Limbiq instance from app state."""
    return request.app.state.lq


def _entity_map(lq) -> dict:
    """Entity ID → name lookup."""
    return {e.id: e.name for e in lq.get_entities()}


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health(request: Request):
    lq = _get_lq(request)
    connectivity = lq.get_graph_connectivity()
    return {
        "status": "ok",
        "entities": connectivity["entities"],
        "relations": connectivity["relations"],
        "fully_connected": connectivity["fully_connected"],
        "components": connectivity["components"],
    }


# ── Process + Observe (core loop) ─────────────────────────────────────────────

@router.post("/process", response_model=ProcessResponse)
async def process(req: QueryRequest, request: Request):
    """Process a user message — returns enriched context for the LLM."""
    lq = _get_lq(request)
    t0 = time.time()

    result = lq.process(req.message)

    # Get world summary for display
    world_summary = ""
    try:
        world_summary = lq.get_world_summary()
    except Exception:
        pass

    signals = []
    for s in result.signals_fired:
        signals.append({
            "signal_type": str(s.signal_type.value) if hasattr(s.signal_type, 'value') else str(s.signal_type),
            "trigger": s.trigger,
            "details": s.details,
        })

    return ProcessResponse(
        context=result.context,
        memories_retrieved=result.memories_retrieved,
        world_summary=world_summary,
        duration_ms=round((time.time() - t0) * 1000, 1),
        signals_fired=signals,
    )


@router.post("/observe")
async def observe(req: ObserveRequest, request: Request):
    """Observe a completed exchange — fires signals, updates graph."""
    lq = _get_lq(request)
    t0 = time.time()

    events = lq.observe(req.message, req.response, req.feedback)

    signal_list = []
    for e in events:
        signal_list.append({
            "signal_type": str(e.signal_type.value) if hasattr(e.signal_type, 'value') else str(e.signal_type),
            "trigger": e.trigger,
            "timestamp": e.timestamp,
            "details": e.details,
            "memory_ids_affected": e.memory_ids_affected,
        })

    # Get updated connectivity after observe
    connectivity = lq.get_graph_connectivity()

    return {
        "signals": signal_list,
        "duration_ms": round((time.time() - t0) * 1000, 1),
        "connectivity": connectivity,
    }


@router.post("/chat")
async def chat(req: QueryRequest, request: Request):
    """Full chat loop: process → LLM → observe. Returns LLM response + signals."""
    lq = _get_lq(request)
    llm = getattr(request.app.state, "llm", None)
    t0 = time.time()

    # Step 1: Process
    result = lq.process(req.message)

    # Step 2: Call LLM (or return context if no LLM)
    llm_response = ""
    if llm and callable(llm):
        try:
            prompt = req.message
            if result.context:
                prompt = f"[Context: {result.context}]\n\nUser: {req.message}"
            llm_response = llm(prompt)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            llm_response = f"[LLM unavailable] Context: {result.context}"
    else:
        llm_response = f"[No LLM configured] Context retrieved: {result.context or 'none'}"

    # Step 3: Observe
    events = lq.observe(req.message, llm_response)

    signal_list = []
    for e in events:
        signal_list.append({
            "signal_type": str(e.signal_type.value) if hasattr(e.signal_type, 'value') else str(e.signal_type),
            "trigger": e.trigger,
            "details": e.details,
        })

    world_summary = ""
    try:
        world_summary = lq.get_world_summary()
    except Exception:
        pass

    connectivity = lq.get_graph_connectivity()

    return {
        "response": llm_response,
        "context": result.context,
        "memories_retrieved": result.memories_retrieved,
        "world_summary": world_summary,
        "signals": signal_list,
        "connectivity": connectivity,
        "duration_ms": round((time.time() - t0) * 1000, 1),
    }


# ── Session management ────────────────────────────────────────────────────────

@router.post("/session/start")
async def start_session(request: Request):
    lq = _get_lq(request)
    lq.start_session()
    return {"status": "ok"}


@router.post("/session/end")
async def end_session(request: Request):
    lq = _get_lq(request)
    t0 = time.time()
    results = lq.end_session()
    results["duration_ms"] = round((time.time() - t0) * 1000, 1)
    return results


# ── Graph ──────────────────────────────────────────────────────────────────────

@router.get("/graph/entities")
async def get_entities(request: Request):
    lq = _get_lq(request)
    entities = lq.get_entities()

    # Count relations per entity
    relations = lq.get_relations(include_inferred=True)
    rel_counts = {}
    for r in relations:
        rel_counts[r.subject_id] = rel_counts.get(r.subject_id, 0) + 1
        rel_counts[r.object_id] = rel_counts.get(r.object_id, 0) + 1

    return [
        EntityModel(
            id=e.id, name=e.name, type=e.entity_type,
            relation_count=rel_counts.get(e.id, 0),
        )
        for e in entities
    ]


@router.get("/graph/relations")
async def get_relations(
    request: Request,
    include_inferred: bool = Query(True),
):
    lq = _get_lq(request)
    relations = lq.get_relations(include_inferred=include_inferred)
    emap = _entity_map(lq)

    return [
        RelationModel(
            id=r.id,
            subject_id=r.subject_id,
            subject_name=emap.get(r.subject_id, "?"),
            predicate=r.predicate,
            object_id=r.object_id,
            object_name=emap.get(r.object_id, "?"),
            confidence=r.confidence,
            is_inferred=r.is_inferred,
        )
        for r in relations
    ]


@router.get("/graph/network")
async def get_graph_network(request: Request):
    """Return graph as nodes + links for D3 visualization."""
    lq = _get_lq(request)
    entities = lq.get_entities()
    relations = lq.get_relations(include_inferred=True)
    emap = {e.id: e for e in entities}

    # Count relations per entity for sizing
    rel_counts = {}
    for r in relations:
        rel_counts[r.subject_id] = rel_counts.get(r.subject_id, 0) + 1
        rel_counts[r.object_id] = rel_counts.get(r.object_id, 0) + 1

    nodes = []
    for e in entities:
        nodes.append({
            "id": e.id,
            "name": e.name,
            "type": e.entity_type,
            "relations": rel_counts.get(e.id, 0),
        })

    links = []
    for r in relations:
        subj = emap.get(r.subject_id)
        obj = emap.get(r.object_id)
        if subj and obj:
            links.append({
                "source": r.subject_id,
                "target": r.object_id,
                "source_id": r.subject_id,
                "target_id": r.object_id,
                "predicate": r.predicate,
                "confidence": r.confidence,
                "is_inferred": r.is_inferred,
                "source_name": subj.name,
                "target_name": obj.name,
            })

    stats = lq.get_graph_stats()
    return GraphNetworkModel(nodes=nodes, links=links, stats=stats)


@router.get("/graph/query")
async def query_graph(request: Request, q: str = Query(...)):
    lq = _get_lq(request)
    t0 = time.time()
    result = lq.query_graph(q)
    return GraphQueryResponse(
        answered=result.get("answered", False),
        answer=result.get("answer"),
        confidence=result.get("confidence", 0),
        source=result.get("source"),
        duration_ms=round((time.time() - t0) * 1000, 1),
    )


@router.get("/graph/connectivity")
async def get_connectivity(request: Request):
    """Return graph connectivity stats — components, fully_connected flag."""
    lq = _get_lq(request)
    conn = lq.get_graph_connectivity()
    return ConnectivityResponse(**conn)


@router.post("/graph/heal")
async def heal_graph(request: Request):
    """Trigger graph self-healing: junk cleanup + inference + connectivity bridging."""
    lq = _get_lq(request)
    t0 = time.time()
    lq.heal_graph()
    connectivity = lq.get_graph_connectivity()
    return {
        "status": "ok",
        "connectivity": connectivity,
        "duration_ms": round((time.time() - t0) * 1000, 1),
    }


@router.get("/graph/describe/{entity_name}")
async def describe_entity(entity_name: str, request: Request):
    lq = _get_lq(request)
    description = lq.describe_entity(entity_name)
    return {"name": entity_name, "description": description}


# ── Signals ────────────────────────────────────────────────────────────────────

@router.get("/signals/log")
async def get_signal_log(request: Request, limit: int = Query(50)):
    lq = _get_lq(request)
    events = lq.get_signal_log(limit)
    return [
        SignalEventModel(
            signal_type=str(e.signal_type.value) if hasattr(e.signal_type, 'value') else str(e.signal_type),
            trigger=e.trigger,
            timestamp=e.timestamp,
            details=e.details,
        )
        for e in events
    ]


@router.get("/signals/rules")
async def get_rules(request: Request):
    lq = _get_lq(request)
    rules = lq.get_active_rules()
    return [
        {
            "id": r.id,
            "pattern_key": r.pattern_key,
            "rule_text": r.rule_text,
            "confidence": r.confidence,
            "observation_count": r.observation_count,
            "is_active": r.is_active,
        }
        for r in rules
    ]


@router.get("/signals/clusters")
async def get_clusters(request: Request):
    lq = _get_lq(request)
    clusters = lq.get_clusters()
    return [
        {
            "id": c.id,
            "topic": c.topic,
            "description": c.description,
            "access_count": c.access_count,
            "memory_count": len(c.memory_ids),
        }
        for c in clusters
    ]


# ── Memories ───────────────────────────────────────────────────────────────────

@router.get("/memories/priority")
async def get_priority_memories(request: Request):
    lq = _get_lq(request)
    memories = lq.get_priority_memories()
    return [
        {"id": m.id, "content": m.content, "confidence": m.confidence}
        for m in memories
    ]


@router.get("/memories/suppressed")
async def get_suppressed(request: Request):
    lq = _get_lq(request)
    memories = lq.get_suppressed()
    return [
        {"id": m.id, "content": m.content, "reason": m.suppression_reason}
        for m in memories
    ]


@router.post("/memories/dopamine")
async def manual_dopamine(request: Request, content: str = Query(...)):
    lq = _get_lq(request)
    lq.dopamine(content)
    return {"status": "ok"}


@router.post("/memories/suppress/{memory_id}")
async def suppress_memory(memory_id: str, request: Request):
    lq = _get_lq(request)
    lq.gaba(memory_id)
    return {"status": "ok"}


@router.post("/memories/restore/{memory_id}")
async def restore_memory(memory_id: str, request: Request):
    lq = _get_lq(request)
    lq.restore_memory(memory_id)
    return {"status": "ok"}


# ── Training ───────────────────────────────────────────────────────────────────

@router.post("/train/reasoner", response_model=TrainResponse)
async def train_reasoner(req: TrainRequest, request: Request):
    """Train the micro-transformer graph reasoner."""
    lq = _get_lq(request)
    t0 = time.time()
    try:
        result = lq.train_reasoner(
            model_dir=req.model_dir or "data/reasoner",
            epochs=req.epochs,
        )
        return TrainResponse(
            status="ok",
            samples=result.get("training_pairs", 0),
            best_eval_acc=result.get("best_eval_acc", 0),
            epochs=req.epochs,
            duration_ms=round((time.time() - t0) * 1000, 1),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/encoder")
async def train_encoder(req: TrainEncoderRequest, request: Request):
    """Train the transformer entity encoder from existing graph data."""
    lq = _get_lq(request)
    t0 = time.time()
    try:
        result = lq.train_encoder()
        result["duration_ms"] = round((time.time() - t0) * 1000, 1)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reason")
async def reason(request: Request, q: str = Query(...)):
    """Answer a question using the graph reasoner."""
    lq = _get_lq(request)
    t0 = time.time()
    try:
        result = lq.reason(q)
        return ReasonResponse(
            answered=result.confidence > 0.3,
            answer=result.answer,
            confidence=result.confidence,
            answer_mode=result.answer_mode,
            reasoning_trace=result.reasoning_trace,
            duration_ms=round((time.time() - t0) * 1000, 1),
        )
    except Exception as e:
        return ReasonResponse(answered=False, answer=str(e))


# ── Stats / Profile ───────────────────────────────────────────────────────────

@router.get("/stats", response_model=StatsResponse)
async def get_stats(request: Request):
    lq = _get_lq(request)
    entities = lq.get_entities()
    relations = lq.get_relations(include_inferred=True)
    explicit = [r for r in relations if not r.is_inferred]
    inferred = [r for r in relations if r.is_inferred]
    mem_stats = lq.get_stats()
    signals = lq.get_signal_log(limit=1000)
    uptime = time.time() - getattr(request.app.state, "start_time", time.time())

    return StatsResponse(
        entities=len(entities),
        relations=len(relations),
        explicit_relations=len(explicit),
        inferred_relations=len(inferred),
        memories=mem_stats.get("total", 0),
        signals_fired=len(signals),
        uptime_seconds=round(uptime, 1),
    )


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(request: Request):
    lq = _get_lq(request)
    entities = lq.get_entities()
    relations = lq.get_relations(include_inferred=True)
    emap = _entity_map(lq)

    rel_counts = {}
    for r in relations:
        rel_counts[r.subject_id] = rel_counts.get(r.subject_id, 0) + 1
        rel_counts[r.object_id] = rel_counts.get(r.object_id, 0) + 1

    world_summary = ""
    try:
        world_summary = lq.get_world_summary()
    except Exception:
        pass

    priority = lq.get_priority_memories()

    return ProfileResponse(
        user_name=lq._core.user_id,
        world_summary=world_summary,
        entities=[
            EntityModel(
                id=e.id, name=e.name, type=e.entity_type,
                relation_count=rel_counts.get(e.id, 0),
            )
            for e in entities
        ],
        relations=[
            RelationModel(
                id=r.id,
                subject_id=r.subject_id,
                subject_name=emap.get(r.subject_id, "?"),
                predicate=r.predicate,
                object_id=r.object_id,
                object_name=emap.get(r.object_id, "?"),
                confidence=r.confidence,
                is_inferred=r.is_inferred,
            )
            for r in relations
        ],
        memory_count=lq.get_stats().get("total", 0),
        priority_facts=[m.content for m in priority],
    )


# ── Propagation ────────────────────────────────────────────────────────────────

@router.post("/propagate")
async def propagate(request: Request):
    """Run Phase 1 active graph propagation."""
    lq = _get_lq(request)
    t0 = time.time()
    result = lq.propagate()
    return {
        "noise_suppressed": result.noise_suppressed,
        "priority_deflated": result.priority_deflated,
        "duplicates_merged": result.duplicates_merged,
        "graph_repaired": result.graph_repaired,
        "inferred_relations": result.inferred_relations,
        "duration_ms": round((time.time() - t0) * 1000, 1),
    }


@router.post("/pattern-completion")
async def pattern_completion(req: TrainRequest, request: Request):
    """Run Phase 3 pattern completion (entity resolution + TransE)."""
    lq = _get_lq(request)
    t0 = time.time()
    try:
        result = lq.run_pattern_completion(
            model_dir=req.model_dir or "data/pattern",
            epochs=req.epochs,
        )
        result["duration_ms"] = round((time.time() - t0) * 1000, 1)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

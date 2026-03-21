"""
REST API endpoints for Limbiq Playground.

All endpoints are prefixed with /api/v1/.
"""

import time
import logging
from typing import Optional

from fastapi import APIRouter, Query, Request, HTTPException

from limbiq.playground.data_models import (
    QueryRequest, ObserveRequest, TrainRequest,
    ProcessResponse, GraphQueryResponse, ReasonResponse,
    StatsResponse, ProfileResponse, TrainResponse,
    EntityModel, RelationModel, GraphNetworkModel, SignalEvent,
)
from limbiq.playground.instrumentation import get_tracer, get_metrics, get_span_exporter

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_lq(request: Request):
    """Get the Limbiq instance from app state."""
    return request.app.state.lq


def _entity_map(lq) -> dict:
    """Entity ID → name lookup."""
    return {e.id: e.name for e in lq.get_entities()}


# ─── Health ───────────────────────────────────────────────────

@router.get("/health")
async def health(request: Request):
    lq = _get_lq(request)
    return {
        "status": "ok",
        "entities": len(lq.get_entities()),
        "relations": len(lq.get_relations(True)),
    }


# ─── Process & Observe ────────────────────────────────────────

@router.post("/process", response_model=ProcessResponse)
async def process_message(body: QueryRequest, request: Request):
    """Process a user message through limbiq — returns enriched context."""
    lq = _get_lq(request)
    tracer = get_tracer()
    met = get_metrics()

    with tracer.start_as_current_span("api.process") as span:
        span.set_attribute("message_length", len(body.message))
        t0 = time.time()

        result = lq.process(message=body.message, conversation_history=[])
        duration_ms = (time.time() - t0) * 1000

        span.set_attribute("memories_retrieved", result.memories_retrieved)
        span.set_attribute("duration_ms", duration_ms)

        if met:
            met.record_process(duration_ms, result.memories_retrieved)

        return ProcessResponse(
            context=result.context,
            memories_retrieved=result.memories_retrieved,
            world_summary=getattr(result, "world_summary", ""),
            duration_ms=duration_ms,
        )


@router.post("/observe")
async def observe_exchange(body: ObserveRequest, request: Request):
    """Record a completed exchange — fires signals, stores memory."""
    lq = _get_lq(request)
    tracer = get_tracer()
    met = get_metrics()

    with tracer.start_as_current_span("api.observe") as span:
        t0 = time.time()
        events = lq.observe(body.message, body.response)
        duration_ms = (time.time() - t0) * 1000

        span.set_attribute("events_count", len(events) if events else 0)
        span.set_attribute("duration_ms", duration_ms)

        if met and events:
            for e in events:
                met.record_signal(
                    getattr(e, "signal_type", "unknown"),
                    getattr(e, "trigger", ""),
                )

        return {
            "status": "ok",
            "events": len(events) if events else 0,
            "duration_ms": duration_ms,
        }


# ─── Chat (full loop: process → answer → observe) ────────────

@router.post("/chat")
async def chat(body: QueryRequest, request: Request):
    """
    Full conversation loop — process + LLM generate + search + observe.

    If an LLM is connected, it generates a real response using limbiq's
    memory context. When the LLM indicates uncertainty (or user uses /search),
    triggers web search, re-prompts with results, and stores findings.
    """
    lq = _get_lq(request)
    llm = getattr(request.app.state, "llm", None)
    search_client = getattr(request.app.state, "search_client", None)
    logger.info(f"Chat: llm={'yes' if llm else 'no'}, search={'yes' if search_client else 'no'}")
    tracer = get_tracer()
    met = get_metrics()

    # Maintain per-session conversation history on the app state
    if not hasattr(request.app.state, "chat_history"):
        request.app.state.chat_history = []
    history = request.app.state.chat_history

    # Check for /search command prefix
    user_message = body.message
    force_search = user_message.startswith("/search ")
    if force_search:
        user_message = user_message[8:].strip()

    with tracer.start_as_current_span("api.chat") as span:
        span.set_attribute("message", user_message)
        span.set_attribute("llm_connected", llm is not None)
        span.set_attribute("search_connected", search_client is not None)
        t0 = time.time()

        # Step 1: Process (retrieve context + memories)
        process_result = lq.process(
            message=user_message,
            conversation_history=history[-12:],
        )

        # Step 2: Try graph + reasoner for instant answers
        graph_result = lq.query_graph(user_message)
        graph_answered = graph_result.get("answered", False)
        graph_answer = graph_result.get("answer", "")

        reason_answer = ""
        reason_confidence = 0
        try:
            reason_result = lq.reason(user_message)
            if reason_result.answered:
                reason_answer = reason_result.answer
                reason_confidence = reason_result.confidence
        except Exception:
            pass

        # Step 3: Generate response (first pass)
        llm_used = False
        search_connected = search_client is not None
        system_parts = [
            "You are a helpful, honest assistant. You have a memory system that "
            "stores facts from previous conversations. Those facts are provided "
            "below inside <memory_context> tags.\n\n"
            "RULES:\n"
            "- ONLY state facts that appear in the memory context or the current conversation.\n"
            "- If the user asks something NOT covered by memory or conversation, "
            + ("say you'll search for it." if search_connected else
               "say \"I don't have that information yet — tell me and I'll remember it.\"") + "\n"
            "- NEVER invent, assume, or hallucinate facts about the user.\n"
            "- NEVER describe your own architecture, memory system, or how you work internally.\n"
            "- Be concise and natural. Reference remembered facts confidently when relevant.\n"
            "- When the user tells you something new, acknowledge it warmly."
        ]
        if process_result.context:
            system_parts.append(process_result.context)
        if graph_answered:
            system_parts.append(f"\n[Graph knowledge — verified fact]: {graph_answer}")
        if reason_answer:
            system_parts.append(f"\n[Reasoner — inferred fact]: {reason_answer}")

        if llm:
            messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
            messages.extend(history[-12:])
            messages.append({"role": "user", "content": user_message})

            try:
                response_text = llm.chat(messages)
                llm_used = True
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                response_text = None

        if not llm_used:
            # Fallback: use graph/reasoner/memory heuristics
            if graph_answered:
                response_text = graph_answer
            elif reason_answer:
                response_text = reason_answer
            elif process_result.context:
                ctx = process_result.context
                lines = []
                for line in ctx.split("\n"):
                    line = line.strip()
                    if line.startswith("- ") or line.startswith("  - "):
                        clean = line.lstrip("- ").strip()
                        if clean.startswith("[") and "] " in clean:
                            clean = clean.split("] ", 1)[1]
                        lines.append(clean)
                if lines:
                    response_text = "Here's what I know:\n" + "\n".join(
                        f"• {l}" for l in lines
                    )
                else:
                    response_text = "I have some memories but couldn't find a specific answer."
            else:
                response_text = "I don't have information about that yet."

        # Step 4: Web search — if LLM is uncertain or user forced /search
        search_used = False
        search_results = []
        if search_client:
            from limbiq.search import detect_uncertainty

            # Search triggers when:
            # 1. User forced with /search prefix
            # 2. LLM response shows uncertainty
            # 3. No LLM connected AND no graph/reasoner answer (nothing useful to say)
            no_answer = not graph_answered and not reason_answer
            needs_search = force_search or (
                detect_uncertainty(response_text)
            ) or (
                not llm_used and no_answer
            )

            logger.info(f"Search check: needs_search={needs_search}, uncertainty={detect_uncertainty(response_text)}, force={force_search}, response_preview={response_text[:150]!r}")

            if needs_search:
                logger.info(f"Searching for: {user_message}")
                try:
                    search_results = search_client(user_message)
                    search_used = bool(search_results)
                    logger.info(f"Search returned {len(search_results)} results")
                except Exception as e:
                    logger.error(f"Web search failed: {e}", exc_info=True)

            # Re-prompt LLM with search results
            if search_used and llm:
                search_context = "\n".join(
                    f'- "{sr.title}" ({sr.source}): {sr.snippet}'
                    for sr in search_results[:3]
                )
                system_with_search = system_parts + [
                    f"[WEB SEARCH — recent information from the internet]\n{search_context}"
                ]
                messages_retry = [
                    {"role": "system", "content": "\n\n".join(system_with_search)}
                ]
                messages_retry.extend(history[-12:])
                messages_retry.append({"role": "user", "content": user_message})

                try:
                    response_text = llm.chat(messages_retry)
                except Exception as e:
                    logger.warning(f"LLM re-prompt with search failed: {e}")
                    # Keep original response

        # Step 5: Update conversation history
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response_text})
        if len(history) > 40:
            request.app.state.chat_history = history[-24:]

        # Step 6: Observe — signals fire, entities extracted, memory stored
        events = lq.observe(user_message, response_text)

        # Also observe search results as facts — persists for future queries
        if search_used:
            for sr in search_results[:3]:
                fact = f"{sr.title}: {sr.snippet}"
                try:
                    lq.observe(user_message, fact)
                except Exception:
                    pass

        duration_ms = (time.time() - t0) * 1000

        signal_events = []
        if events:
            for e in events:
                sig_type = str(getattr(e, "signal_type", "unknown"))
                trigger = getattr(e, "trigger", "")
                signal_events.append({"type": sig_type, "trigger": trigger})
                if met:
                    met.record_signal(sig_type, trigger)

        span.set_attribute("graph_answered", graph_answered)
        span.set_attribute("llm_used", llm_used)
        span.set_attribute("search_used", search_used)
        span.set_attribute("signals_fired", len(signal_events))
        span.set_attribute("duration_ms", duration_ms)

        return {
            "message": user_message,
            "response": response_text,
            "llm_used": llm_used,
            "search_used": search_used,
            "search_results_count": len(search_results),
            "graph_answered": graph_answered,
            "graph_answer": graph_answer,
            "graph_confidence": graph_result.get("confidence", 0),
            "reason_answer": reason_answer,
            "reason_confidence": reason_confidence,
            "context": process_result.context,
            "memories_retrieved": process_result.memories_retrieved,
            "signals_fired": signal_events,
            "stored": True,
            "duration_ms": duration_ms,
        }


# ─── Graph Queries ────────────────────────────────────────────

@router.post("/graph/query", response_model=GraphQueryResponse)
async def query_graph(body: QueryRequest, request: Request):
    """Query the knowledge graph with natural language."""
    lq = _get_lq(request)
    tracer = get_tracer()
    met = get_metrics()

    with tracer.start_as_current_span("api.graph_query") as span:
        span.set_attribute("question", body.message)
        t0 = time.time()

        result = lq.query_graph(body.message)
        duration_ms = (time.time() - t0) * 1000

        answered = result.get("answered", False)
        confidence = result.get("confidence", 0)

        span.set_attribute("answered", answered)
        span.set_attribute("confidence", confidence)

        if met:
            met.record_query(duration_ms, answered, confidence)

        return GraphQueryResponse(
            answered=answered,
            answer=result.get("answer"),
            confidence=confidence,
            source=result.get("source"),
            duration_ms=duration_ms,
        )


@router.post("/reason", response_model=ReasonResponse)
async def reason(body: QueryRequest, request: Request):
    """Answer using Phase 5 micro-transformer reasoner."""
    lq = _get_lq(request)
    tracer = get_tracer()

    with tracer.start_as_current_span("api.reason") as span:
        span.set_attribute("question", body.message)
        t0 = time.time()

        try:
            result = lq.reason(body.message)
            duration_ms = (time.time() - t0) * 1000
            span.set_attribute("answered", result.answered)
            span.set_attribute("confidence", result.confidence)

            return ReasonResponse(
                answered=result.answered,
                answer=result.answer,
                confidence=result.confidence,
                answer_mode=result.answer_mode,
                reasoning_trace=result.reasoning_trace,
                duration_ms=duration_ms,
            )
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return ReasonResponse(
                answered=False,
                answer=f"Error: {e}",
                duration_ms=(time.time() - t0) * 1000,
            )


# ─── Graph Exploration ────────────────────────────────────────

@router.get("/graph/entities")
async def get_entities(request: Request):
    """List all entities in the knowledge graph."""
    lq = _get_lq(request)
    entities = lq.get_entities()
    rels = lq.get_relations(include_inferred=True)

    # Count relations per entity
    rel_counts = {}
    for r in rels:
        rel_counts[r.subject_id] = rel_counts.get(r.subject_id, 0) + 1
        rel_counts[r.object_id] = rel_counts.get(r.object_id, 0) + 1

    return {
        "entities": [
            EntityModel(
                id=e.id, name=e.name, type=e.entity_type,
                relation_count=rel_counts.get(e.id, 0),
            ).model_dump()
            for e in sorted(entities, key=lambda e: e.name.lower())
        ],
        "count": len(entities),
    }


@router.get("/graph/relations")
async def get_relations(
    request: Request,
    include_inferred: bool = Query(True),
):
    """List all relations in the knowledge graph."""
    lq = _get_lq(request)
    rels = lq.get_relations(include_inferred=include_inferred)
    emap = _entity_map(lq)

    return {
        "relations": [
            RelationModel(
                id=r.id,
                subject_id=r.subject_id,
                subject_name=emap.get(r.subject_id, "?"),
                predicate=r.predicate,
                object_id=r.object_id,
                object_name=emap.get(r.object_id, "?"),
                confidence=r.confidence,
                is_inferred=r.is_inferred,
            ).model_dump()
            for r in rels
        ],
        "count": len(rels),
        "explicit": sum(1 for r in rels if not r.is_inferred),
        "inferred": sum(1 for r in rels if r.is_inferred),
    }


@router.get("/graph/network")
async def get_graph_network(request: Request):
    """Full graph as D3-compatible nodes + links."""
    lq = _get_lq(request)
    entities = lq.get_entities()
    rels = lq.get_relations(include_inferred=True)
    emap = _entity_map(lq)

    nodes = [
        {"id": e.id, "label": e.name, "type": e.entity_type}
        for e in entities
    ]
    links = [
        {
            "source": r.subject_id,
            "target": r.object_id,
            "label": r.predicate,
            "confidence": r.confidence,
            "inferred": r.is_inferred,
        }
        for r in rels
        # Only include links where both ends exist
        if r.subject_id in emap and r.object_id in emap
    ]

    return GraphNetworkModel(
        nodes=nodes,
        links=links,
        stats={
            "node_count": len(nodes),
            "edge_count": len(links),
            "explicit_edges": sum(1 for l in links if not l["inferred"]),
            "inferred_edges": sum(1 for l in links if l["inferred"]),
        },
    ).model_dump()


@router.get("/graph/entity/{name}")
async def describe_entity(name: str, request: Request):
    """Get detailed description of a named entity."""
    lq = _get_lq(request)
    emap = _entity_map(lq)
    rels = lq.get_relations(include_inferred=True)

    # Find entity
    entity = None
    for e in lq.get_entities():
        if e.name.lower() == name.lower():
            entity = e
            break

    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")

    # Get relations involving this entity
    outgoing = []
    incoming = []
    for r in rels:
        if r.subject_id == entity.id:
            outgoing.append(RelationModel(
                id=r.id, subject_id=r.subject_id,
                subject_name=emap.get(r.subject_id, "?"),
                predicate=r.predicate,
                object_id=r.object_id,
                object_name=emap.get(r.object_id, "?"),
                confidence=r.confidence,
                is_inferred=r.is_inferred,
            ).model_dump())
        elif r.object_id == entity.id:
            incoming.append(RelationModel(
                id=r.id, subject_id=r.subject_id,
                subject_name=emap.get(r.subject_id, "?"),
                predicate=r.predicate,
                object_id=r.object_id,
                object_name=emap.get(r.object_id, "?"),
                confidence=r.confidence,
                is_inferred=r.is_inferred,
            ).model_dump())

    return {
        "entity": EntityModel(
            id=entity.id, name=entity.name, type=entity.entity_type,
            relation_count=len(outgoing) + len(incoming),
        ).model_dump(),
        "outgoing": outgoing,
        "incoming": incoming,
    }


# ─── Stats & Profile ─────────────────────────────────────────

@router.get("/stats", response_model=StatsResponse)
async def get_stats(request: Request):
    """System statistics."""
    lq = _get_lq(request)
    entities = lq.get_entities()
    rels = lq.get_relations(include_inferred=True)
    start_time = request.app.state.start_time

    return StatsResponse(
        entities=len(entities),
        relations=len(rels),
        explicit_relations=sum(1 for r in rels if not r.is_inferred),
        inferred_relations=sum(1 for r in rels if r.is_inferred),
        uptime_seconds=time.time() - start_time,
    )


@router.get("/profile")
async def get_profile(request: Request):
    """Full user profile snapshot."""
    lq = _get_lq(request)
    entities = lq.get_entities()
    rels = lq.get_relations(include_inferred=True)
    emap = _entity_map(lq)

    # World summary
    try:
        world = lq.get_world_summary()
    except Exception:
        world = ""

    # Priority facts
    try:
        priority = lq.get_priority_memories()
        priority_facts = [m.content for m in priority[:20]] if priority else []
    except Exception:
        priority_facts = []

    return {
        "user_name": lq._core.user_name if hasattr(lq._core, "user_name") else "unknown",
        "world_summary": world,
        "entity_count": len(entities),
        "relation_count": len(rels),
        "priority_facts": priority_facts,
    }


# ─── Operations ───────────────────────────────────────────────

@router.post("/propagate")
async def run_propagate(request: Request):
    """Run inference + propagation."""
    lq = _get_lq(request)
    tracer = get_tracer()

    with tracer.start_as_current_span("api.propagate") as span:
        t0 = time.time()
        try:
            result = lq.propagate()
            duration_ms = (time.time() - t0) * 1000
            span.set_attribute("duration_ms", duration_ms)
            return {"status": "ok", "result": str(result), "duration_ms": duration_ms}
        except Exception as e:
            return {"status": "error", "error": str(e)}


@router.post("/pattern-completion")
async def run_pattern_completion(request: Request):
    """Run pattern completion."""
    lq = _get_lq(request)
    t0 = time.time()
    try:
        result = lq.run_pattern_completion()
        return {"status": "ok", "result": str(result), "duration_ms": (time.time() - t0) * 1000}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/train", response_model=TrainResponse)
async def train_reasoner(body: TrainRequest, request: Request):
    """Train the Phase 5 micro-transformer reasoner."""
    lq = _get_lq(request)
    tracer = get_tracer()

    with tracer.start_as_current_span("api.train_reasoner") as span:
        span.set_attribute("epochs", body.epochs)
        t0 = time.time()
        try:
            result = lq.train_reasoner(
                model_dir=body.model_dir or "./data/reasoner",
                epochs=body.epochs,
            )
            duration_ms = (time.time() - t0) * 1000
            return TrainResponse(
                status="ok",
                samples=result.get("samples", 0),
                best_eval_acc=result.get("best_eval_acc", 0),
                epochs=body.epochs,
                duration_ms=duration_ms,
            )
        except Exception as e:
            return TrainResponse(status=f"error: {e}")


# ─── Telemetry ────────────────────────────────────────────────

@router.get("/telemetry/traces")
async def get_traces(limit: int = Query(50)):
    """Return recent OpenTelemetry traces."""
    exporter = get_span_exporter()
    if not exporter:
        return {"traces": [], "message": "Telemetry not initialized"}
    return {"traces": exporter.get_traces(limit)}


@router.get("/telemetry/spans")
async def get_spans(limit: int = Query(100)):
    """Return recent spans (flat list)."""
    exporter = get_span_exporter()
    if not exporter:
        return {"spans": []}
    return {"spans": exporter.get_recent(limit)}


@router.get("/telemetry/metrics")
async def get_telemetry_metrics(
    since: float = Query(0),
    metric_type: Optional[str] = Query(None),
):
    """Return time-series metrics for dashboard charts."""
    met = get_metrics()
    if not met:
        return {"data": [], "message": "Metrics not initialized"}
    return {"data": met.get_time_series(since, metric_type)}

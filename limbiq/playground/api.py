"""
REST API endpoints for Limbiq Playground.

All endpoints are prefixed with /api/v1/.

Features implemented:
  1. Onboarding + agent persona
  3. Tool use (/file, /run, /calc)
  4. Auto fact-check against graph
  5. Proactive session greeting
  6. Chain-of-thought injection
  7. Multi-model orchestration
"""

import re
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


# ── CoT detection helpers ─────────────────────────────────────────────────────

_COT_RE = re.compile(
    r"\b(why|how|explain|what if|compare|because|reason|cause|differ|"
    r"analyse|analyze|understand|describe|illustrate|demonstrate|evaluate)\b",
    re.IGNORECASE,
)
_MULTI_HOP_RE = re.compile(r"\b(and|also|then|first|second|third|finally|after)\b", re.IGNORECASE)


def _needs_cot(query: str, signals: list = None) -> bool:
    """Return True when chain-of-thought guidance would help."""
    signals = signals or []
    sig_types = {str(getattr(s, "signal_type", s)).lower() for s in signals}
    if any("norepinephrine" in t for t in sig_types):
        return True
    if _COT_RE.search(query):
        return True
    if len(_MULTI_HOP_RE.findall(query)) >= 2:
        return True
    return False


# ── Fact-check helpers ────────────────────────────────────────────────────────

_CLAIM_RE = re.compile(
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:your\s+)?([a-zA-Z\s]+)",
    re.MULTILINE,
)


def _extract_claims(text: str) -> list[tuple[str, str]]:
    """Extract simple subject-predicate claims from LLM output."""
    claims = []
    for m in _CLAIM_RE.finditer(text):
        subject = m.group(1).strip()
        predicate = m.group(2).strip().rstrip(".,!?")
        if 2 < len(subject) < 50 and 1 < len(predicate) < 60:
            claims.append((subject, predicate))
    return claims[:5]  # limit to 5 claims per response


def _fact_check(lq, response_text: str) -> tuple[bool, list[str]]:
    """
    Check LLM claims against the knowledge graph.
    Returns (fact_checked, list_of_corrections).
    """
    claims = _extract_claims(response_text)
    if not claims:
        return False, []

    corrections = []
    for subject, predicate in claims:
        try:
            query = f"Who is {subject}?"
            result = lq.query_graph(query)
            if result.get("answered") and result.get("confidence", 0) > 0.8:
                graph_answer = result.get("answer", "")
                # Simple heuristic: if the predicate words aren't in the graph answer,
                # it might be contradicted.
                pred_words = set(predicate.lower().split())
                answer_words = set(graph_answer.lower().split())
                overlap = pred_words & answer_words
                # Only flag if there's near-zero overlap with something the graph knows
                if graph_answer and len(overlap) == 0 and len(pred_words) > 1:
                    corrections.append(
                        f"Note: My records say {graph_answer}"
                    )
        except Exception:
            pass  # graph check failed silently

    return bool(corrections), corrections


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health(request: Request):
    lq = _get_lq(request)
    return {
        "status": "ok",
        "entities": len(lq.get_entities()),
        "relations": len(lq.get_relations(True)),
    }


# ── Onboarding ────────────────────────────────────────────────────────────────

@router.get("/onboarding")
async def get_onboarding(request: Request):
    """Return current onboarding state and profile."""
    mgr = getattr(request.app.state, "onboarding_manager", None)
    if mgr is None:
        return {"complete": True, "agent_name": "Limbiq", "user_name": ""}
    profile = mgr.get_profile()
    return {
        "complete": profile.onboarding_complete,
        "agent_name": profile.agent_name,
        "user_name": profile.user_name,
    }


# ── Greeting ──────────────────────────────────────────────────────────────────

@router.get("/greeting")
async def get_greeting(request: Request):
    """Generate a personalised session greeting."""
    lq = _get_lq(request)
    llm = getattr(request.app.state, "llm", None)
    mgr = getattr(request.app.state, "onboarding_manager", None)
    model_router = getattr(request.app.state, "model_router", None)

    profile = mgr.get_profile() if mgr else None
    agent_name = profile.agent_name if profile else "Limbiq"
    user_name = profile.user_name if profile else ""

    # Count known facts
    try:
        world = lq.get_world_summary()
        facts_known = len([s for s in world.split(".") if s.strip()])
    except Exception:
        world = ""
        facts_known = 0

    # Build greeting from template only — never use LLM for greetings
    # because small quantized models hallucinate fabricated experiences
    greeting = _template_greeting(agent_name, user_name, world)

    return {"greeting": greeting, "facts_known": facts_known}


def _template_greeting(agent_name: str, user_name: str, world: str) -> str:
    name_part = f"Hi {user_name}!" if user_name else "Hi!"
    agent_part = f"I'm {agent_name}."
    if world:
        # Find a complete, clean fact (no "?" or broken data)
        for sentence in world.split("."):
            fact = sentence.strip()
            if fact and "?" not in fact and len(fact) > 10:
                return f"{name_part} {agent_part} I remember that {fact[0].lower()}{fact[1:]}."
    return f"{name_part} {agent_part} How can I help you today?"


# ── Process & Observe ──────────────────────────────────────────────────────────

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


# ── Chat (full loop: process → answer → observe) ──────────────────────────────

@router.post("/chat")
async def chat(body: QueryRequest, request: Request):
    """
    Full conversation loop — process + LLM generate + search + observe.

    Includes:
      - Onboarding flow (first 2 messages)
      - Tool use (/file, /run, /calc)
      - Fact-checking against graph
      - Chain-of-thought injection
      - Multi-model routing
    """
    lq = _get_lq(request)
    llm_base = getattr(request.app.state, "llm", None)
    search_client = getattr(request.app.state, "search_client", None)
    model_router = getattr(request.app.state, "model_router", None)
    onboarding_mgr = getattr(request.app.state, "onboarding_manager", None)
    tool_registry = getattr(request.app.state, "tool_registry", None)

    logger.info(f"Chat: llm={'yes' if llm_base else 'no'}, search={'yes' if search_client else 'no'}")
    tracer = get_tracer()
    met = get_metrics()

    # Maintain per-session conversation history on the app state
    if not hasattr(request.app.state, "chat_history"):
        request.app.state.chat_history = []
    history = request.app.state.chat_history

    # ── Onboarding flow ───────────────────────────────────────────────────────
    def _looks_like_name(text: str) -> bool:
        """Check if text is a plausible name (1-3 short words, no sentences)."""
        words = text.strip().split()
        return 1 <= len(words) <= 3 and len(text) <= 40

    def _onboarding_response(**overrides):
        """Build a minimal onboarding response dict."""
        base = {
            "llm_used": False, "search_used": False, "search_results_count": 0,
            "graph_answered": False, "graph_answer": "", "graph_confidence": 0,
            "reason_answer": "", "reason_confidence": 0,
            "context": "", "memories_retrieved": 0, "signals_fired": [],
            "stored": False, "duration_ms": 0,
            "tools_used": [], "fact_checked": False, "corrections": [],
            "cot_used": False, "model_used": "none", "hallucination_caught": False,
        }
        base.update(overrides)
        return base

    if onboarding_mgr and not onboarding_mgr.is_complete():
        step = onboarding_mgr.get_step()  # Persisted in DB, survives refresh
        user_message = body.message.strip()

        if step == 0:
            onboarding_mgr.set_step(1)
            return _onboarding_response(
                message=user_message,
                response="Hi! I'm Limbiq. Before we start, what's your name?",
            )

        if step == 1:
            if not _looks_like_name(user_message):
                # Not a name — re-ask nicely
                return _onboarding_response(
                    message=user_message,
                    response="I'd love to know your name! Just type your first name.",
                )
            onboarding_mgr.set_user_name(user_message)
            onboarding_mgr.set_step(2)

            # Update the graph with the real user name
            try:
                core = lq._core
                capitalized_name = user_message.strip().capitalize()
                # Create user entity with real name
                from limbiq.graph.store import Entity
                user_entity = core.graph.add_entity(
                    Entity(name=capitalized_name, entity_type="person")
                )
                # Verify entity was actually created (not silently rejected)
                found = core.graph.find_entity_by_name(capitalized_name)
                if found:
                    # Update entity extractor to use real name
                    core.entity_extractor.user_name = capitalized_name
                    core.entity_extractor.user_entity = found
                    # Update graph query and inference
                    core._graph_user_name = capitalized_name
                    core.graph_query.user_name = capitalized_name
                    # Update inference engine reference
                    core.inference_engine._user_name_cache = None
                    logger.info(f"Graph user entity updated: {capitalized_name}")
                else:
                    logger.warning(f"User entity '{capitalized_name}' was rejected by graph store")

                # Store as dopamine priority memory
                lq.dopamine(f"User's name is {capitalized_name}")
            except Exception as e:
                logger.warning(f"Failed to update graph with user name: {e}")

            return _onboarding_response(
                message=user_message,
                response=f"Nice to meet you, {user_message}! What would you like to call me?",
            )

        if step == 2:
            if not _looks_like_name(user_message):
                # Not a name — re-ask nicely
                return _onboarding_response(
                    message=user_message,
                    response="Pick a short name for me! Something like Jim, Nova, or Atlas.",
                )
            onboarding_mgr.set_agent_name(user_message)
            onboarding_mgr.complete()
            profile = onboarding_mgr.get_profile()
            return _onboarding_response(
                message=user_message,
                response=(
                    f"Perfect! I'm {profile.agent_name} and I'll remember that name. "
                    f"Now, {profile.user_name}, what's on your mind?"
                ),
            )

    # ── Normal chat flow ──────────────────────────────────────────────────────
    with tracer.start_as_current_span("api.chat") as span:
        # Check for tool command
        user_message = body.message
        force_search = user_message.startswith("/search ")
        if force_search:
            user_message = user_message[8:].strip()

        # Detect tool request before anything else
        tool_request = None
        tools_used = []
        if tool_registry:
            tool_request = tool_registry.detect_tool_request(user_message)

        span.set_attribute("message", user_message[:200])
        span.set_attribute("llm_connected", llm_base is not None)
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

        # ── Select model ──────────────────────────────────────────────────────
        signal_events_raw = []  # populated after observe below
        if model_router:
            llm = model_router.route(user_message, process_result, [])
            model_used = model_router.get_model_name(user_message, process_result, [])
        elif llm_base:
            llm = llm_base
            model_used = getattr(llm_base, "model", "default")
        else:
            llm = None
            model_used = "none"

        # ── Build system prompt ───────────────────────────────────────────────
        search_connected = search_client is not None

        # Agent persona
        persona_parts = []
        if onboarding_mgr:
            profile = onboarding_mgr.get_profile()
            if profile.agent_name:
                persona_parts.append(f"Your name is {profile.agent_name}.")
            if profile.user_name:
                persona_parts.append(f"You were named by {profile.user_name}.")
        persona_str = " ".join(persona_parts)

        no_info = ("say you don't know and offer to search for it." if search_connected
                   else "say you don't have that information yet.")

        # Build tool description for system prompt
        tool_desc = ""
        if tool_registry:
            tools = tool_registry.list_tools()
            if tools:
                tool_desc = (
                    "\nYou have access to tools. URLs shared by the user are automatically fetched. "
                    "Available commands: " + ", ".join(f"/{t}" for t in tools) + ". "
                    "When results are provided in [TOOL RESULT], use them to answer the question."
                )

        system_parts = [
            (f"{persona_str} " if persona_str else "") +
            "You are a helpful, honest assistant.\n"
            "RULES:\n"
            "- Use ONLY the facts below and the conversation history. Refer back to recent messages when relevant.\n"
            "- If a fact is NOT in the provided context or conversation history, DO NOT mention it. "
            f"Instead, {no_info}\n"
            "- NEVER make up events, trips, experiences, or stories. NEVER add fictional details.\n"
            "- Never quote tags like [KNOWN FACT] or <memory_context>. Be concise and warm."
            + tool_desc
        ]

        if process_result.context:
            system_parts.append(process_result.context)

        # Inject world summary directly if not already in context
        # (handles case where graph user name was just updated by onboarding)
        try:
            world = lq.get_world_summary()
            if world and "[ABOUT YOU]" not in (process_result.context or ""):
                system_parts.append(f"[ABOUT YOU] {world}")
        except Exception:
            pass

        if graph_answered:
            system_parts.append(f"\n[Graph knowledge — verified fact]: {graph_answer}")
        if reason_answer:
            system_parts.append(f"\n[Reasoner — inferred fact]: {reason_answer}")

        # ── CoT injection (Feature 6) ─────────────────────────────────────────
        cot_used = _needs_cot(user_message, [])
        if cot_used:
            system_parts.append(
                "Think through this step by step before answering. Show your reasoning briefly."
            )

        # ── Tool execution (Feature 3) ────────────────────────────────────────
        if tool_request and tool_registry:
            tool_name, tool_args = tool_request
            result_obj = tool_registry.execute(tool_name, tool_args)
            tools_used.append(tool_name)
            from limbiq.tools import format_tool_results
            tool_context = format_tool_results([result_obj])
            if tool_context:
                system_parts.append(f"[TOOL RESULT]\n{tool_context}")

        # ── Generate response ─────────────────────────────────────────────────
        llm_used = False
        response_text = None

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
                        # Skip raw user messages and duplicates
                        if clean.startswith("User said:") or clean.startswith("User asked:"):
                            continue
                        if clean.startswith("So my wife") or len(clean) > 150:
                            continue  # Skip long raw conversation dumps
                        lines.append(clean)
                # Deduplicate
                lines = list(dict.fromkeys(lines))
                if lines:
                    response_text = "Here's what I know:\n" + "\n".join(
                        f"• {l}" for l in lines[:5]
                    )
                else:
                    response_text = "I have some memories but couldn't find a specific answer."
            else:
                response_text = "I don't have information about that yet."

        # ── Auto-detect tool from LLM output (Feature 3 — optional) ─────────
        if llm_used and tool_registry and response_text:
            auto_tool = tool_registry.detect_auto(response_text)
            if auto_tool:
                auto_name, auto_args = auto_tool
                auto_result = tool_registry.execute(auto_name, auto_args)
                if auto_result.success:
                    tools_used.append(auto_name)
                    from limbiq.tools import format_tool_results
                    tool_ctx = format_tool_results([auto_result])
                    # Re-prompt with tool context
                    system_with_tool = system_parts + [f"[TOOL RESULT]\n{tool_ctx}"]
                    retry_msgs = [{"role": "system", "content": "\n\n".join(system_with_tool)}]
                    retry_msgs.extend(history[-12:])
                    retry_msgs.append({"role": "user", "content": user_message})
                    try:
                        response_text = llm.chat(retry_msgs)
                    except Exception:
                        pass  # keep original response

        # ── Web search ────────────────────────────────────────────────────────
        search_used = False
        search_results = []
        if search_client:
            from limbiq.search import detect_uncertainty

            no_answer = not graph_answered and not reason_answer
            needs_search = force_search or (
                detect_uncertainty(response_text)
            ) or (
                not llm_used and no_answer
            )

            logger.info(f"Search check: needs_search={needs_search}, uncertainty={detect_uncertainty(response_text)}, force={force_search}")

            if needs_search and user_message.strip():
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

        # ── Fact-check (Feature 4) ────────────────────────────────────────────
        fact_checked = False
        corrections = []
        if response_text:
            try:
                fact_checked, corrections = _fact_check(lq, response_text)
                if corrections:
                    response_text = response_text + "\n\n" + " ".join(corrections)
            except Exception as e:
                logger.warning(f"Fact-check failed: {e}")

        # ── Hallucination detection + regeneration ─────────────────────────────
        hallucination_caught = False
        if llm_used and response_text:
            try:
                detector = lq.get_hallucination_detector()
                user_name = ""
                if onboarding_mgr:
                    p = onboarding_mgr.get_profile()
                    user_name = p.user_name or ""

                # Pre-generate grounding analysis
                grounding = detector.pre_generate(
                    query=user_message,
                    graph_result=graph_result,
                    memories_retrieved=process_result.memories_retrieved,
                    memory_context=process_result.context,
                )

                # Post-generate verification (pass conversation history to avoid false positives)
                verification = detector.post_generate(
                    response=response_text,
                    user_entity_name=user_name or "user",
                    query=user_message,
                    conversation_history=history[-12:],
                )

                # Log verification results
                logger.info(
                    f"Hallucination check: score={verification.hallucination_score:.2f}, "
                    f"claims={len(verification.claims)}, verified={verification.verified_count}, "
                    f"contradicted={verification.contradicted_count}, unverified={verification.unverified_count}, "
                    f"grounding={grounding.level.value}"
                )
                for c in verification.claims:
                    logger.info(f"  Claim: [{c.status.value}] {c.predicate}: {c.text[:80]}")

                # Regenerate if hallucination detected
                if detector.should_regenerate(verification, grounding):
                    hallucination_caught = True
                    logger.warning(f"Hallucination detected (score={verification.hallucination_score:.2f}), regenerating...")

                    correction = detector.correction_prompt(
                        verification, user_message, process_result.context
                    )
                    # Include conversation history so regeneration has context
                    retry_msgs = [
                        {"role": "system", "content": "\n\n".join(system_parts) + "\n\n" + correction},
                    ]
                    retry_msgs.extend(history[-12:])
                    retry_msgs.append({"role": "user", "content": user_message})
                    try:
                        response_text = llm.chat(retry_msgs)
                    except Exception:
                        pass  # Keep original on retry failure

                    detector.record(user_message, response_text, grounding, verification, "regenerated")
                else:
                    detector.record(user_message, response_text, grounding, verification, "accepted")
            except Exception as e:
                logger.warning(f"Hallucination check failed: {e}")

        # ── Update conversation history ────────────────────────────────────────
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response_text})
        if len(history) > 40:
            request.app.state.chat_history = history[-24:]

        # ── Observe — signals fire, entities extracted, memory stored ─────────
        events = lq.observe(user_message, response_text)

        # Also observe search results as facts
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
            "tools_used": tools_used,
            "fact_checked": fact_checked,
            "corrections": corrections,
            "cot_used": cot_used,
            "model_used": model_used,
            "hallucination_caught": hallucination_caught,
        }


# ── Graph Queries ──────────────────────────────────────────────────────────────

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


# ── Graph Exploration ──────────────────────────────────────────────────────────

@router.get("/graph/entities")
async def get_entities(request: Request):
    """List all entities in the knowledge graph."""
    lq = _get_lq(request)
    entities = lq.get_entities()
    rels = lq.get_relations(include_inferred=True)

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

    entity = None
    for e in lq.get_entities():
        if e.name.lower() == name.lower():
            entity = e
            break

    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")

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


# ── Memory Views ──────────────────────────────────────────────────────────────

@router.get("/memories")
async def get_memories(request: Request, tier: str = Query(None), limit: int = Query(50)):
    """Return stored memories, optionally filtered by tier."""
    lq = _get_lq(request)
    db = lq._core.store.db
    query = "SELECT id, content, tier, confidence, created_at, access_count, is_priority, is_suppressed, source FROM memories"
    params = []
    if tier:
        query += " WHERE tier = ?"
        params.append(tier)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    rows = db.execute(query, params).fetchall()
    return {
        "memories": [
            {
                "id": r[0], "content": r[1], "tier": r[2], "confidence": r[3],
                "created_at": r[4], "access_count": r[5], "is_priority": bool(r[6]),
                "is_suppressed": bool(r[7]), "source": r[8],
            }
            for r in rows
        ],
        "count": len(rows),
    }


@router.get("/memories/priority")
async def get_priority_memories(request: Request):
    """Return all dopamine-tagged priority memories."""
    lq = _get_lq(request)
    memories = lq.get_priority_memories()
    return {
        "memories": [
            {"id": m.id, "content": m.content, "confidence": m.confidence, "source": m.source}
            for m in memories
        ],
        "count": len(memories),
    }


@router.get("/memories/suppressed")
async def get_suppressed_memories(request: Request):
    """Return all GABA-suppressed memories."""
    lq = _get_lq(request)
    memories = lq.get_suppressed()
    return {
        "memories": [
            {"id": m.id, "content": m.content, "suppression_reason": m.suppression_reason}
            for m in memories
        ],
        "count": len(memories),
    }


@router.get("/signals")
async def get_signals(request: Request, limit: int = Query(50)):
    """Return recent signal events."""
    lq = _get_lq(request)
    events = lq.get_signal_log(limit)
    return {
        "signals": [
            {
                "signal_type": str(e.signal_type),
                "trigger": e.trigger,
                "timestamp": e.timestamp,
                "details": e.details if hasattr(e, 'details') else {},
            }
            for e in events
        ],
        "count": len(events),
    }


@router.get("/rules")
async def get_rules(request: Request):
    """Return active behavioral rules (serotonin)."""
    lq = _get_lq(request)
    rules = lq.get_active_rules()
    return {
        "rules": [
            {"id": r.id, "pattern_key": r.pattern_key, "rule_text": r.rule_text,
             "confidence": r.confidence, "observation_count": r.observation_count,
             "is_active": r.is_active}
            for r in rules
        ],
        "count": len(rules),
    }


# ── Stats & Profile ────────────────────────────────────────────────────────────

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

    try:
        world = lq.get_world_summary()
    except Exception:
        world = ""

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


# ── Graph Corrections ──────────────────────────────────────────────────────────

@router.post("/graph/delete-relation")
async def delete_relation(request: Request):
    """Delete a specific relation from the knowledge graph."""
    lq = _get_lq(request)
    body = await request.json()
    subject = body.get("subject", "")
    predicate = body.get("predicate", "")
    obj = body.get("object", "")

    if subject and predicate and obj:
        lq.delete_relation(subject, predicate, obj)
        return {"status": "ok", "deleted": f"{subject} → {predicate} → {obj}"}
    elif subject and obj:
        lq.delete_relations_between(subject, obj)
        return {"status": "ok", "deleted": f"all relations between {subject} and {obj}"}
    else:
        raise HTTPException(status_code=400, detail="Provide subject + predicate + object, or subject + object")


# ── Operations ─────────────────────────────────────────────────────────────────

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


# ── Telemetry ──────────────────────────────────────────────────────────────────

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


# ── Hallucination Detection ───────────────────────────────────────────────────

@router.post("/hallucination/check")
async def check_hallucination(request: Request, body: QueryRequest):
    """
    Run hallucination detection on a query + response pair.
    Useful for testing: provide a message, get grounding + verification.
    """
    lq = request.app.state.limbiq

    # Get the hallucination detector
    detector = lq.get_hallucination_detector()
    user_name = lq._core._graph_user_name

    # Pre-generation grounding
    graph_result = lq.query_graph(body.message)
    result = lq.process(body.message)

    grounding = detector.pre_generate(
        query=body.message,
        graph_result=graph_result if graph_result.get("answered") else None,
        memories_retrieved=result.memories_retrieved,
        memory_context=result.context,
        world_summary=lq.get_world_summary(),
    )

    return {
        "grounding": {
            "level": grounding.level.value,
            "query_type": grounding.query_type,
            "graph_has_answer": grounding.graph_has_answer,
            "memory_relevance_score": grounding.memory_relevance_score,
            "relevant_fact_count": grounding.relevant_fact_count,
            "known_entities": grounding.known_entities_mentioned,
            "constraint_prompt": grounding.constraint_prompt,
            "suggested_temperature": grounding.suggested_temperature,
        },
    }


@router.post("/hallucination/verify")
async def verify_response(request: Request, body: ObserveRequest):
    """
    Verify an LLM response for hallucinated facts.
    body.message = original query, body.response = LLM response to verify.
    """
    lq = request.app.state.limbiq
    detector = lq.get_hallucination_detector()
    user_name = lq._core._graph_user_name

    verification = detector.post_generate(
        response=body.response,
        user_entity_name=user_name,
        query=body.message,
    )

    return {
        "claims": [
            {
                "text": c.text,
                "subject": c.subject,
                "predicate": c.predicate,
                "object_value": c.object_value,
                "status": c.status.value,
                "evidence": c.evidence,
                "confidence": c.confidence,
            }
            for c in verification.claims
        ],
        "verified_count": verification.verified_count,
        "unverified_count": verification.unverified_count,
        "contradicted_count": verification.contradicted_count,
        "hallucination_score": verification.hallucination_score,
        "flagged_text": verification.flagged_text,
    }


@router.get("/hallucination/stats")
async def get_hallucination_stats(request: Request):
    """Return hallucination detection statistics."""
    lq = request.app.state.limbiq
    detector = lq.get_hallucination_detector()
    return {
        "stats": detector.get_stats(),
        "recent_events": detector.get_recent_events(20),
    }

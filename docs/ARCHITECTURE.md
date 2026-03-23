# Limbiq Architecture

> Neurotransmitter-inspired adaptive learning layer for LLMs (v0.6.0)

Limbiq sits between the user and any LLM, enriching context with learned memories, behavioral rules, knowledge graph facts, and signal-driven adaptations — all without modifying model weights.

```
User Message → Limbiq.process() → Enriched Context → Any LLM → Response → Limbiq.observe() → Loop
```

---

## Table of Contents

- [System Overview](#system-overview)
- [Layer Architecture](#layer-architecture)
- [Data Flow](#data-flow)
- [Signal System](#signal-system)
- [Knowledge Graph Pipeline](#knowledge-graph-pipeline)
- [Playground](#playground)
- [Shared State & Threading](#shared-state--threading)
- [External Dependencies](#external-dependencies)
- [Dependency Graph](#dependency-graph)

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                   Playground (playground/)                            │
│           FastAPI + React SPA — REST API at /api/v1/                 │
│         Chat · Graph Viz · Memory Inspector · Query Builder          │
├──────────────────────────────────────────────────────────────────────┤
│                        PUBLIC API (__init__.py)                       │
│  Limbiq  ProcessResult  SignalEvent  Memory  RetrievalConfig  ...    │
├──────────────────────────────────────────────────────────────────────┤
│                         LimbiqCore (core.py)                         │
│              process()  →  observe()  →  end_session()               │
├──────┬──────────┬─────────┬──────────┬───────────────────────────────┤
│Store │ Signals  │  Graph  │Retrieval │  Context                      │
│Layer │  Layer   │  Layer  │  Layer   │  Layer                        │
├──────┴──────────┴─────────┴──────────┴───────────────────────────────┤
│                    Shared SQLite via MemoryStore                      │
│              (thread-local connections via self._store.db)            │
└──────────────────────────────────────────────────────────────────────┘
```

**Key design principles:**

- **LLM-agnostic** — works with any callable `llm_fn(prompt) → str`; Limbiq never calls an LLM directly
- **Zero weight modification** — all adaptation through context manipulation
- **Single SQLite database** — all stores share one file, accessed via per-thread connections
- **Lazy loading** — heavy dependencies (torch, spaCy, sentence-transformers) loaded only when needed
- **Reversible** — suppressed memories can be restored, rules deactivated, nothing permanently destructive
- **Stripped to essentials** — signals + graph generation + self-healing (compression, hallucination, steering, and other experimental layers removed in v0.5)

---

## Layer Architecture

### Directory Structure (current)

```
limbiq/
├── __init__.py              # Public API facade (Limbiq class)
├── core.py                  # Orchestrator — process(), observe(), end_session()
├── types.py                 # All dataclasses and enums
├── store/                   # SQLite persistence + embeddings
│   ├── memory_store.py      # CRUD for memories, vectorized similarity search
│   ├── embeddings.py        # Text → vector embeddings
│   ├── cluster_store.py     # Acetylcholine knowledge clusters
│   ├── rule_store.py        # Serotonin behavioral rules
│   └── signal_log.py        # Immutable log of all signal events
├── signals/                 # 5 neurotransmitter signals
│   ├── base.py              # BaseSignal ABC (detect → apply)
│   ├── dopamine.py          # "This matters. Remember it."
│   ├── gaba.py              # "Suppress this. Let it fade."
│   ├── serotonin.py         # "This pattern is stable. Make it permanent."
│   ├── acetylcholine.py     # "Focus here. Build expertise."
│   └── norepinephrine.py    # "Something changed. Be careful."
├── graph/                   # Knowledge graph + neural reasoning
│   ├── store.py             # Entity/Relation CRUD, self-healing junk cleanup
│   ├── entities.py          # Hybrid extraction: spaCy dep parse (Tier 1) + LLM (Tier 2) + regex fallback
│   ├── encoder.py           # Transformer entity encoder (type + relation classifiers)
│   ├── inference.py         # Deterministic relation inference (in-laws, multi-hop)
│   ├── query.py             # Natural language → graph lookup
│   ├── propagation.py       # Phase 1: rule-based graph maintenance
│   ├── gnn.py               # Phase 2: GNN-based activation propagation
│   ├── pattern_completion.py# Phase 3: TransE entity resolution + link prediction
│   └── reasoning.py         # Phase 5: micro-transformer graph QA
├── retrieval/               # Activation-weighted retrieval
│   └── activation_retrieval.py  # Phase 4: GNN-weighted memory search
├── context/                 # Context string assembly
│   └── builder.py           # Assembles final context from memories + rules + graph
└── playground/              # Interactive web dashboard (optional)
    ├── __init__.py          # Exports create_app
    ├── __main__.py          # CLI entry point, LLM setup
    ├── server.py            # FastAPI app + inline React SPA
    ├── api.py               # 28 REST endpoints (chat, graph, memory, signals, training)
    └── data_models.py       # 14 Pydantic request/response models
```

### 1. Store Layer (`store/`)

Persistence and embedding-based retrieval. All stores share one SQLite database.

| Module | Purpose | Key Dependencies |
|--------|---------|-----------------|
| `memory_store.py` | CRUD for memories, vectorized similarity search with numpy cache | numpy, sqlite3 |
| `embeddings.py` | Text → vector embeddings with thread-safe LRU cache | sentence-transformers (TF-IDF fallback) |
| `cluster_store.py` | Acetylcholine knowledge clusters | memory_store (shared DB) |
| `rule_store.py` | Serotonin behavioral rules & pattern observations | memory_store (shared DB) |
| `signal_log.py` | Immutable log of all signal events | memory_store (shared DB) |

**SQLite Tables (created by `memory_store.py`):**
- `memories` — main memory store (content, tier, confidence, embedding BLOB, suppression state)
- `signal_log` — signal event history
- `conversations` — raw conversation logs
- `sessions` — session metadata

**SQLite Tables (created by sub-stores):**
- `knowledge_clusters` — acetylcholine topic clusters (cluster_store.py)
- `pattern_observations` — serotonin pattern tracking (rule_store.py)
- `behavioral_rules` — crystallized behavioral rules (rule_store.py)
- `entities` — knowledge graph nodes (graph/store.py)
- `relations` — knowledge graph edges (graph/store.py)

### 2. Signals Layer (`signals/`)

Five neurotransmitter-inspired signals that detect patterns and modify memory state.

| Module | Signal | Trigger | Effect |
|--------|--------|---------|--------|
| `dopamine.py` | Dopamine | Personal info, corrections, positive feedback | Stores PRIORITY memory; suppresses contradictions |
| `gaba.py` | GABA | Denials, negative feedback | Suppresses memory (soft delete) |
| `serotonin.py` | Serotonin | Recurring patterns (3+ times, 2+ sessions) | Crystallizes behavioral rule |
| `acetylcholine.py` | Acetylcholine | Sustained topic interest (3+ turns) | Creates/grows knowledge cluster |
| `norepinephrine.py` | Norepinephrine | Topic shifts (cosine < 0.3), frustration | Widens retrieval, adds caution flag |

All signals extend `BaseSignal` (`signals/base.py`) which defines:
- `detect(message, response, feedback, memories) → list[SignalEvent]`
- `apply(event, memory_store, embeddings) → None`

**Two signal execution paths in `core.py`:**
1. **v0.1 loop** (observe): Dopamine, GABA, Norepinephrine — `detect()` + `apply()` in sequence
2. **v0.2 parallel** (observe): Serotonin and Acetylcholine — run via `ThreadPoolExecutor` alongside entity extraction

### 3. Graph Layer (`graph/`)

Knowledge graph with entity extraction, deterministic inference, and five neural phases.

| Module | Purpose | Key Dependencies |
|--------|---------|-----------------|
| `store.py` | Entity/Relation CRUD, junk entity self-healing | memory_store |
| `entities.py` | Hybrid extraction: spaCy dep parse (Tier 1), LLM tie-break (Tier 2), regex fallback | spaCy (opt), llm_fn (opt), encoder.py |
| `encoder.py` | Transformer entity/relation classifiers (BIO tagger, type clf, relation clf) | torch (opt), embeddings.py |
| `inference.py` | Deterministic transitive closure: in-laws, co-parents, multi-hop, reverse in-laws | graph/store |
| `query.py` | Regex pattern matching on questions → graph lookup | graph/store, inference |
| `propagation.py` | Phase 1: noise suppression, dedup, priority deflation, graph repair | memory_store, graph/store |
| `gnn.py` | Phase 2: GAT-based activation propagation (3-layer, 4-head, ~2-5M params) | torch, numpy |
| `pattern_completion.py` | Phase 3: TransE entity resolution + link prediction (~50K params) | torch, numpy |
| `reasoning.py` | Phase 5: micro-transformer QA (~50K params, entity/bool/count modes) | torch, numpy |

### 4. Retrieval Layer (`retrieval/`)

| Module | Purpose | Key Dependencies |
|--------|---------|-----------------|
| `activation_retrieval.py` | Phase 4: hybrid scoring `α*emb_sim + β*gnn_act + γ*graph_boost` | numpy, memory_store (lazy) |

Also provides `GraphStateContextBuilder` — builds enriched context when GNN retrieval is active, and `GraphTrainingDataGenerator` — exports graph QA pairs as JSONL.

### 5. Context Layer (`context/`)

| Module | Purpose | Key Dependencies |
|--------|---------|-----------------|
| `builder.py` | Assembles `<memory_context>` string with token budget (1500 max) | types.py (Memory, BehavioralRule) |

**Context sections (in priority order):**
1. **Mandatory:** Caution flag, graph answer (`[KNOWN FACT]`), world summary (`[ABOUT YOU]`)
2. **Skippable:** Behavioral rules (`[STYLE]`), priority memories, cluster memories, relevant memories

Graph-aware deduplication: memories whose proper nouns appear in the world summary are excluded.

### 6. Playground (`playground/`)

Interactive web dashboard for debugging and exploring limbiq. Optional — requires `pip install -e ".[playground]"`.

| Module | Purpose | Key Dependencies |
|--------|---------|-----------------|
| `__init__.py` | Package init, exports `create_app` | server.py |
| `__main__.py` | CLI entry point (`python -m limbiq.playground`), builds `llm_fn` via OpenAI SDK | argparse, uvicorn, openai |
| `server.py:22` | FastAPI app creation (v0.6.0), specific-origin CORS, lifespan (Limbiq init + session), inline React SPA | fastapi, limbiq (Limbiq class) |
| `api.py` | 28 REST endpoints under `/api/v1/` | fastapi, limbiq (Limbiq class), data_models |
| `data_models.py` | 14 Pydantic request/response models | pydantic |

**Architecture:** The playground sits ABOVE the public API — it only calls `Limbiq` methods from `__init__.py`. It never imports `core.py` or any internal module directly. The only internal access is `lq._core.user_id` in the `/profile` endpoint.

**Frontend:** Single inline HTML file embedded in `server.py` — React 18 + D3.js loaded from CDN. Zero build step required.

**CORS:** Specific origins only: `http://localhost:8765`, `http://127.0.0.1:8765` (`server.py:66`).

**Lifespan:** On startup, creates `Limbiq` instance and calls `start_session()`. On shutdown, calls `end_session()` (`server.py:30-55`).

**Pydantic Models (data_models.py):**

| Model | Type | Purpose |
|-------|------|---------|
| `QueryRequest` | Request | message + optional conversation_id |
| `ObserveRequest` | Request | message + response + optional feedback |
| `TrainRequest` | Request | epochs + optional model_dir |
| `TrainEncoderRequest` | Request | epochs |
| `EntityModel` | Graph | id, name, type, relation_count |
| `RelationModel` | Graph | full subject→predicate→object with names |
| `GraphNetworkModel` | Graph | D3-compatible nodes + links + stats |
| `ProcessResponse` | Response | context, memories, world summary, signals, duration |
| `GraphQueryResponse` | Response | answered, answer, confidence, source |
| `ReasonResponse` | Response | answer + mode + reasoning trace |
| `StatsResponse` | Response | entity/relation/memory counts + uptime |
| `ConnectivityResponse` | Response | components, fully_connected, component_sizes |
| `ProfileResponse` | Response | user profile with entities, relations, priorities |
| `TrainResponse` | Response | training status, samples, accuracy, duration |
| `SignalEventModel` | Response | signal_type, trigger, timestamp, details |

**API Endpoints (28 routes):**

| Method | Path | Purpose | Line |
|--------|------|---------|------|
| GET | `/health` | Health check (connectivity stats) | `api.py:39` |
| POST | `/process` | Run `lq.process()` — enriched context | `api.py:54` |
| POST | `/observe` | Run `lq.observe()` — fire signals, update graph | `api.py:86` |
| POST | `/chat` | Full loop: process → LLM → observe | `api.py:114` |
| POST | `/session/start` | Start new session | `api.py:170` |
| POST | `/session/end` | End session + cleanup | `api.py:177` |
| GET | `/graph/entities` | List entities with relation counts | `api.py:188` |
| GET | `/graph/relations` | List relations (filterable by inferred) | `api.py:209` |
| GET | `/graph/network` | D3-compatible nodes + links for viz | `api.py:233` |
| GET | `/graph/query` | Natural language graph query | `api.py:277` |
| GET | `/graph/connectivity` | Connected component statistics | `api.py:291` |
| POST | `/graph/heal` | Trigger self-healing + inference | `api.py:299` |
| GET | `/graph/describe/{name}` | Entity description via inference | `api.py:313` |
| GET | `/signals/log` | Recent signal event log | `api.py:322` |
| GET | `/signals/rules` | Active behavioral rules (serotonin) | `api.py:337` |
| GET | `/signals/clusters` | Knowledge clusters (acetylcholine) | `api.py:354` |
| GET | `/memories/priority` | Dopamine-tagged priority memories | `api.py:372` |
| GET | `/memories/suppressed` | GABA-suppressed memories | `api.py:382` |
| POST | `/memories/dopamine` | Manually tag content as priority | `api.py:392` |
| POST | `/memories/suppress/{id}` | Manually suppress a memory | `api.py:399` |
| POST | `/memories/restore/{id}` | Restore a suppressed memory | `api.py:406` |
| POST | `/train/reasoner` | Train micro-transformer graph reasoner | `api.py:415` |
| POST | `/train/encoder` | Train transformer entity encoder | `api.py:436` |
| POST | `/reason` | Answer question via graph reasoner | `api.py:449` |
| GET | `/stats` | Entity/relation/memory counts + uptime | `api.py:470` |
| GET | `/profile` | Full user profile snapshot | `api.py:492` |
| POST | `/propagate` | Run Phase 1 active graph propagation | `api.py:542` |
| POST | `/pattern-completion` | Run Phase 3 entity resolution + TransE | `api.py:558` |

---

## Data Flow

### `process(message)` — Before LLM Generation

```
User Message
    │
    ├─1─→ Embed query                          (embeddings.py)
    │     Cache for reuse in observe()
    │
    ├──── PARALLEL (ThreadPoolExecutor, 4 workers) ────────────────
    │     ├─2a─→ Try graph answer               (graph/query.py → inference.py)
    │     │      └─ If confidence > 0.8 → graph_context
    │     ├─2b─→ Get world summary              (inference.py → get_user_world)
    │     ├─2c─→ Get relevant graph context      (inference.py → get_relevant_graph_context)
    │     └─2d─→ Get priority memories           (memory_store.py)
    ├─────────────────────────────────────────────────────────────
    │
    ├─3─→ Apply pending NE effects             (norepinephrine.py → RetrievalConfig)
    │     └─ May widen top_k (10→20→30), add caution flag
    │
    ├─4─→ Check NE for topic shift             (norepinephrine.detect_for_process)
    │     └─ Cosine similarity < 0.3 with previous embedding
    │
    ├─5─→ Retrieve memories                    (activation_retrieval.py or memory_store.py)
    │     ├─ Phase 4 (GNN-weighted) if enabled
    │     └─ Fallback: vectorized cosine similarity search
    │
    ├─6─→ Increment access counts              (memory_store.py batch)
    │
    ├─7─→ Load behavioral rules                (rule_store.py)
    │
    ├─8─→ Match topic clusters                 (acetylcholine.detect_topic_for_retrieval)
    │
    └─9─→ Build context string                 (context/builder.py or GraphStateContextBuilder)
          └─ Combines: world summary + graph answer + rules + memories + caution
              │
              ▼
         ProcessResult { context, signals_fired, memories_retrieved, ... }
```

### `observe(message, response)` — After LLM Generation

```
User Message + LLM Response
    │
    ├─1─→ Buffer conversation                  (core.py → _conversation_buffer)
    │
    ├─2─→ Embed message (reuse cache from process if same text)
    │
    ├─3─→ Search existing memories             (memory_store.py, top_k=5)
    │
    ├─4─→ Run v0.1 signals sequentially:       (signals/)
    │     ├─ Dopamine  → detect + apply (store priority memory, suppress contradictions)
    │     ├─ GABA      → detect + apply (suppress denied memories)
    │     └─ NE        → detect (queue events for next process())
    │
    ├──── PARALLEL (ThreadPoolExecutor, 4 workers) ────────────────
    │     ├─5a─→ Serotonin: analyze patterns   (serotonin.py → rule_store.py)
    │     │      └─ May crystallize new rule (≥3 obs, ≥2 sessions)
    │     ├─5b─→ ACh: analyze topic focus       (acetylcholine.py → cluster_store.py)
    │     │      └─ May create/expand cluster (3+ turns on topic)
    │     └─5c─→ Extract entities from message  (entities.py → graph/store.py)
    │            ├─ Tier 1: spaCy dep parsing (if available)
    │            ├─ Tier 2: LLM tie-break (if uncertain + llm_fn)
    │            └─ Fallback: regex patterns (if spaCy unavailable)
    │            NOTE: entities are NOT extracted from LLM response
    │            (response echoes/rephrases pollute the graph)
    ├─────────────────────────────────────────────────────────────
    │
    ├─6─→ Graph correction on denial/correction (core._correct_graph)
    │     └─ Delete wrong relations, update entity types, re-extract
    │
    ├─7─→ Graph self-healing:
    │     ├─ graph.heal()                       (junk entity cleanup)
    │     ├─ inference_engine.run_full_inference (transitive closure)
    │     └─ _heal_graph_connectivity()         (bridge disconnected components)
    │
    ├─8─→ Store raw memory (ONLY if no signals fired AND message > 20 chars)
    │
    └─9─→ Clear embedding cache
              │
              ▼
         list[SignalEvent]
```

### `end_session()` — Session Cleanup

```
Session End
    │
    ├─1─→ Extract entities from conversation buffer  (entities.py)
    │     └─ Process any queued LLM extractions
    │
    ├─2─→ Store raw conversation log                  (memory_store.py)
    │
    ├─3─→ Run graph inference                         (inference.py)
    │     ├─ Transitive closure (in-laws, multi-hop, co-parents)
    │     └─ Heal graph connectivity (bridge disconnected components)
    │
    ├─4─→ Entity resolution (Phase 3, requires torch)
    │     ├─ Merge duplicates, resolve canonical names
    │     └─ Re-run inference after merges
    │
    ├─5─→ Age all memories                            (session_count++)
    │
    ├─6─→ Suppress stale memories (10+ sessions, never accessed)
    │     └─ Creates GABA event (never_accessed)
    │
    └─7─→ Delete old suppressed (30+ sessions)
              │
              ▼
         { compressed, aged, suppressed, deleted, graph_inferred, entities_merged }
```

---

## Signal System

### Signal Lifecycle

```
                    ┌──────────┐
                    │  detect  │  Analyze message/response/feedback
                    └────┬─────┘
                         │ list[SignalEvent]
                         ▼
                    ┌──────────┐
                    │  apply   │  Modify store/embeddings
                    └────┬─────┘
                         │
                         ▼
                    ┌──────────┐
                    │   log    │  Persist to signal_log table
                    └──────────┘
```

### Signal Interaction Map

```
Signal            Fires When                      Effect on Pipeline
──────────────────────────────────────────────────────────────────────────
Dopamine     ──→  personal info / correction  ──→  PRIORITY memory → always in context
                  enthusiasm / positive fb          Suppress contradictions
GABA         ──→  denial / negative feedback  ──→  suppress memory → excluded from search
Serotonin    ──→  pattern seen 3+ times       ──→  behavioral rule → injected as [STYLE]
ACh          ──→  sustained topic (3+ turns)  ──→  knowledge cluster → loaded as domain unit
NE           ──→  topic shift / frustration   ──→  widen retrieval + [CAUTION] flag
```

### Signal Detection Patterns

**Dopamine** triggers on:
- Correction phrases: "no that's wrong", "actually it's", "let me correct"
- Enthusiasm: "exactly", "yes!", "that's it", "nailed it"
- Personal info: "my name is", "i work at", "i live in", "i prefer"
- Explicit feedback: `feedback="positive"` or `feedback="correction:..."`

**GABA** triggers on:
- Denial phrases: "i never said", "you're making that up", "that's fabricated"
- Explicit feedback: `feedback="negative"`

**Serotonin** triggers on (via `analyze_and_track`):
- Heuristic: short messages (<10 words → prefers_concise), casual markers (yeah, lol, tbh)
- LLM analysis (optional): prompts LLM to identify communication patterns
- Crystallizes when: ≥3 observations of same pattern_key across ≥2 sessions

**Acetylcholine** triggers on (via `analyze_topic`):
- Topic continuity: 3+ consecutive turns on same/similar topic
- Depth requests: "tell me more", "go deeper", "elaborate"
- Topic detection: bigram extraction → word frequency (heuristic) or LLM (fallback)

**Norepinephrine** triggers on:
- Topic shift: cosine similarity < 0.3 between consecutive query embeddings (process-time)
- Frustration: "i already told you", "wrong again", "are you even listening" (observe-time)
- Contradiction markers: "actually i", "not anymore", "i changed" (observe-time)

### Memory Tiers

```
PRIORITY  ── Dopamine-tagged, always included in context, confidence=1.0
LONG      ── High-confidence, compressed facts (not currently used in v0.5)
MID       ── Acetylcholine cluster memories, confidence=0.8
SHORT     ── Raw conversation fragments (decay fastest)
```

---

## Knowledge Graph Pipeline

Five phases, each building on the previous:

```
Phase 1: Rule-Based Propagation     (propagation.py)
  ├─ Suppress noise memories (regex pattern matching)
  ├─ Deflate inflated priorities (demote non-genuine priority mems)
  ├─ Merge near-identical memories (cosine > 0.92)
  ├─ Repair graph (re-extract entities from all memories)
  └─ Run deterministic inference

Phase 2: GNN Propagation            (gnn.py)
  ├─ Architecture: 3-layer GAT, 4 heads, 128-dim hidden, ~2-5M params
  ├─ Three output heads: quality classifier (noise/normal/priority),
  │   activation predictor (0-1), merge embedding (64-dim)
  ├─ Trains on Phase 1 labels (self-supervised)
  └─ Replaces hand-written propagation rules

Phase 3: Pattern Completion         (pattern_completion.py)
  ├─ Entity resolution: merge "default"→user, case duplicates, semantic merges
  ├─ TransE model: score(h, r, t) = -||h + r - t||, ~50K params
  └─ Link prediction: predict missing relations from learned patterns

Phase 4: Activation Retrieval       (activation_retrieval.py)
  ├─ Hybrid scoring: α*emb_sim + β*gnn_act + γ*graph_boost
  │   (α=0.45, β=0.35, γ=0.20)
  └─ GraphStateContextBuilder: enriched context with activation scores

Phase 5: Graph Reasoning            (reasoning.py)
  ├─ Micro-transformer: 2-layer, 4-head, 64-dim, ~50K params
  ├─ Dynamic vocabulary from graph entities + predicates
  ├─ Three answer modes: entity pointer, boolean, count
  └─ 100% synthetic training data from graph triples
```

### Graph Self-Healing (runs on every `observe()`)

```
observe()
    │
    ├─ graph.heal()
    │   └─ _cleanup_junk_entities()
    │       ├─ Detect names that are predicate words ("Wife", "Dog")
    │       ├─ Resolve via graph (find real entity with that predicate)
    │       ├─ Resolve via memories (mine priority memories for proper names)
    │       ├─ Re-point relations from junk → real entity
    │       └─ Delete junk entity
    │
    ├─ inference_engine.run_full_inference()
    │   ├─ Clear all inferred relations
    │   ├─ In-law inference (same-subject pairs)
    │   ├─ Co-parent inference (father + mother → married_to)
    │   ├─ Reverse in-law inference (spouse + in-law → direct relation)
    │   └─ Multi-hop inference (A→B→C chains: grandparent, uncle, aunt)
    │
    └─ _heal_graph_connectivity()
        ├─ BFS to find disconnected components
        ├─ Strategy 1: Memory co-occurrence bridge
        ├─ Strategy 2: Embedding similarity bridge (cosine > 0.5)
        └─ Strategy 3: Connect orphan to user hub (last resort)
```

### Entity Extraction Pipeline (entities.py)

Hybrid two-tier architecture with regex fallback:

```
Input text
    │
    ├─0─→ Transformer encoder pass              (encoder.py:213)
    │     └─ Produces entity embeddings + optional learned NER/relation detection
    │
    ├─── IF spaCy available ──────────────────────────────────────
    │  │
    │  ├─1─→ TIER 1: Dependency parsing          (entities.py:660)
    │  │     │
    │  │     ├─ Pass 1: Possessive relations      (entities.py:797)
    │  │     │   ├─ Walk dep tree from "my/our" possessive tokens
    │  │     │   ├─ Simple: "my wife Prabhashi"
    │  │     │   │   poss→head→appos chain → (user, wife, Prabhashi)
    │  │     │   ├─ Chained: "my wifes father Chandrasiri"
    │  │     │   │   poss→head→compound chain → resolve_chained_predicate
    │  │     │   │   → (user, father_in_law, Chandrasiri)
    │  │     │   ├─ Copular: "my father is Upananda" / "Prabhashi is my wife"
    │  │     │   │   nsubj→copula→attr pattern → (user, father, Upananda)
    │  │     │   ├─ Naming: "my Dog's name is Dexter"
    │  │     │   │   poss→"name"→copula→attr → (user, pet, Dexter)
    │  │     │   ├─ Generic pet: "my Dog" (no name follows)
    │  │     │   │   → (user, pet, Dog) with type=animal
    │  │     │   └─ Proximity fallback: "my dad upananda" (lowercase name)
    │  │     │       → checks token after relation word even if not PROPN
    │  │     │
    │  │     ├─ Pet merge: named pet replaces generic    (entities.py:716)
    │  │     │   "Dog" + "Dexter" → merge into Dexter, transfer relations
    │  │     │
    │  │     └─ Pass 2: Subject-verb-object              (entities.py:1019)
    │  │         ├─ "[entity] has/have [condition]"
    │  │         │   nsubj→verb→dobj → (Dog, has_condition, megaesophagus)
    │  │         ├─ "[entity] works at [company]"
    │  │         │   nsubj→verb→prep→pobj → (user, works_at, Google)
    │  │         └─ "[entity] lives in [place]"
    │  │             nsubj→verb→prep→pobj → (user, lives_in, London)
    │  │
    │  └─2─→ TIER 2: LLM tie-break               (entities.py:1177)
    │        └─ Only for uncertain fragments where Tier 1 found structure
    │           but couldn't resolve (e.g., no name found for possessive)
    │
    ├─── ELSE (spaCy unavailable) ────────────────────────────────
    │  │
    │  └─1─→ FALLBACK: Regex + spaCy validation   (entities.py:1195)
    │        ├─ 20+ regex patterns for family, work, location, pets
    │        └─ Validated against spaCy NER entities (if available)
    │
    ├─────────────────────────────────────────────────────────────
    │
    ├─3─→ Register orphan spaCy NER entities     (entities.py)
    │     └─ Names found by NER but not captured by any relation pattern
    │
    ├─4─→ Merge encoder output                   (entities.py)
    │     └─ Attach embeddings to extracted entities, add encoder-detected extras
    │
    └─5─→ Queue for deep LLM extraction          (deferred batch)
          └─ Structured prompt → ENTITY/RELATION/MERGE lines
```

**Key design decisions:**
- Tier 1 handles ~90% of standard English via parse tree structure (sub-ms latency)
- Tier 2 only fires for genuinely ambiguous fragments, not every message
- Regex is retained as fallback for environments where spaCy is not installed
- The proximity fallback catches names that spaCy misclassifies (e.g., lowercase proper nouns parsed as ADV)

### Inference Rules

**Same-subject rules** (in-law detection):
```
User→father→X + User→wife→Y   ⟹  X father_in_law_of Y
User→mother→X + User→wife→Y   ⟹  X mother_in_law_of Y
User→brother→X + User→wife→Y  ⟹  X brother_in_law_of Y
```

**Co-parent rules:**
```
User→father→X + User→mother→Y ⟹  X married_to Y
```

**Multi-hop rules:**
```
A→father→B, B→father→C  ⟹  C grandfather_of A
A→father→B, B→sister→C  ⟹  C aunt_of A
A→mother→B, B→brother→C ⟹  C uncle_of A
```

---

## Shared State & Threading

The entire system shares **one SQLite database file** via `MemoryStore`.

```
                    ┌─────────────────────────┐
                    │     SQLite Database      │
                    │   ({user_id}.db)         │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │   thread-local conn     │
                    │   (self._store.db)      │
                    └────────────┬────────────┘
                                 │
           ┌────────┬────────┬───┼───┬────────┬──────────┐
           ▼        ▼        ▼   ▼   ▼        ▼          ▼
      memory_   cluster_  rule_ signal_ graph/    graph/     graph/
      store     store     store  log    store   propagation  pattern_
                                                           completion
```

**Thread safety:** `MemoryStore` uses `threading.local()` for per-thread `sqlite3.Connection` instances. All sub-stores access the database through `self._store.db`, which returns the calling thread's connection. This makes Limbiq safe for FastAPI, Gradio, and other async/threaded frameworks.

**Embedding cache:** `MemoryStore` maintains a numpy matrix cache (`_emb_matrix`) for vectorized cosine similarity. Thread-safe via `threading.Lock` (`_emb_lock`). Incremental updates on `store()`, full rebuild on invalidation.

**Concurrency in core.py:**
- `process()`: 4-worker `ThreadPoolExecutor` for graph query + world summary + graph context + priority memories
- `observe()`: 4-worker `ThreadPoolExecutor` for serotonin + acetylcholine + entity extraction (message only)

---

## External Dependencies

### Required

| Package | Used By | Purpose |
|---------|---------|---------|
| numpy | memory_store.py, propagation.py, activation_retrieval.py | Vectorized cosine similarity, matrix ops |
| sentence-transformers | embeddings.py | Text embedding (all-MiniLM-L6-v2 default, 384-dim) |

### Optional (lazy-loaded)

| Package | Used By | Purpose | Fallback |
|---------|---------|---------|----------|
| torch | gnn.py, pattern_completion.py, reasoning.py, encoder.py | Neural graph operations | Phases 2-5 skip, regex-only extraction |
| spacy | entities.py | Tier 1 dep parsing + NER (en_core_web_sm) | Regex patterns only (no dep tree) |
| fastapi | playground/server.py, playground/api.py | REST API + web dashboard | Playground unavailable |
| uvicorn | playground/__main__.py | ASGI server | Playground unavailable |
| pydantic | playground/data_models.py | Request/response validation | Playground unavailable |
| openai | playground/__main__.py | OpenAI-compatible LLM client | Heuristic-only mode (no LLM) |

### Install

```bash
pip install -e ".[dev]"           # Core + dev dependencies (pytest)
pip install -e ".[playground]"    # + FastAPI web dashboard
```

---

## Dependency Graph

Arrows point from dependent → dependency. **No circular dependencies exist.**

```
playground/ (optional — sits above public API)
    ├── server.py ──→ fastapi, limbiq (Limbiq class via lifespan)
    ├── api.py ──→ fastapi, data_models, limbiq (Limbiq class via app.state)
    ├── data_models.py ──→ pydantic
    └── __main__.py ──→ uvicorn, openai (opt), server.py

__init__.py (facade)
    │
    └──→ core.py (orchestrator — 13 internal imports)
            │
            ├──→ types.py (foundation — no limbiq imports)
            │
            ├──→ Store Layer
            │     ├── memory_store.py ──→ types, numpy
            │     ├── embeddings.py ──→ sentence-transformers (or TF-IDF fallback)
            │     ├── cluster_store.py ──→ types (receives memory_store via __init__)
            │     ├── rule_store.py ──→ types (receives memory_store via __init__)
            │     └── signal_log.py ──→ types (receives memory_store via __init__)
            │
            ├──→ Signals Layer
            │     ├── base.py ──→ types (ABC)
            │     ├── dopamine.py ──→ base, types
            │     ├── gaba.py ──→ base, types
            │     ├── serotonin.py ──→ base, types
            │     ├── acetylcholine.py ──→ base, types
            │     └── norepinephrine.py ──→ base, types
            │
            ├──→ Graph Layer
            │     ├── store.py ──→ (receives memory_store via __init__)
            │     ├── entities.py ──→ graph/store, spaCy (opt), encoder.py (opt)
            │     ├── encoder.py ──→ torch (opt), embeddings.py
            │     ├── inference.py ──→ graph/store
            │     ├── query.py ──→ graph/store, inference
            │     ├── propagation.py ──→ memory_store, graph/store, inference (lazy)
            │     ├── gnn.py ──→ torch, numpy, memory_store (lazy), propagation (lazy)
            │     ├── pattern_completion.py ──→ graph/store, torch, numpy
            │     └── reasoning.py ──→ graph/store, torch, numpy
            │
            ├──→ Retrieval Layer
            │     └── activation_retrieval.py ──→ numpy, memory_store (lazy)
            │
            └──→ context/builder.py ──→ types
```

### Layer Dependency Direction (strictly enforced)

```
types (foundation, no limbiq imports)
  ↑
store/ (imports types only)
  ↑
signals/ (imports types, base only)
  ↑
graph/ (imports store, types)
  ↑
retrieval/ (imports store — lazy)
  ↑
context/ (imports types only)
  ↑
core.py (imports all layers)
  ↑
__init__.py (imports core + re-exports)
  ↑
playground/ (imports __init__.py only — optional layer)
```

No reverse dependencies (e.g., store importing from graph) exist.
The playground is an **optional leaf** — it depends on the public API but nothing depends on it.

### Coupling Hotspots

| Module | Internal Dependencies | Notes |
|--------|----------------------|-------|
| `core.py` | 13 | Central orchestrator — expected hub |
| `__init__.py` | ~10 (with try/except) | Public facade — re-exports, optional graph imports |
| `playground/api.py` | 3 (data_models, fastapi, limbiq) | Leaf layer — 28 endpoints, no reverse deps |
| `gnn.py` | 5 (3 lazy) | Lazy imports keep startup cost low |
| `entities.py` | 3-5 (spaCy/encoder optional) | Entity extraction pipeline |

All other modules have 0-2 internal dependencies.

---

## Key Data Structures

### Core Types (types.py)

| Type | Purpose |
|------|---------|
| `Memory` | Content + tier + confidence + embedding + suppression state |
| `SignalEvent` | Signal type + trigger + timestamp + affected memory IDs |
| `ProcessResult` | Context string + signal events + retrieval stats |
| `BehavioralRule` | Pattern key + rule text + observation count + active flag |
| `KnowledgeCluster` | Topic + memory IDs + access count |
| `RetrievalConfig` | Dynamic top_k + threshold + caution flag (modified by NE) |
| `MemoryTier` | SHORT / MID / LONG / PRIORITY |
| `SignalType` | DOPAMINE / GABA / SEROTONIN / ACETYLCHOLINE / NOREPINEPHRINE |
| `SuppressionReason` | USER_DENIED / STALE / NEVER_ACCESSED / CONTRADICTED / CONFABULATION / MANUAL |

### Graph Types (graph/store.py)

| Type | Purpose |
|------|---------|
| `Entity` | id + name + entity_type + properties + source_memory_id |
| `Relation` | subject_id + predicate + object_id + confidence + is_inferred |

---

*Updated 2026-03-23. Full codebase scan of 35 `.py` files (28 modules + 7 `__init__.py`) in `limbiq/`. No circular dependencies found. Entity extraction updated to hybrid dep-parse pipeline (Tier 1: spaCy deps, Tier 2: LLM tie-break, Fallback: regex). Playground v0.6 (28 endpoints, openai SDK, specific-origin CORS).*

# Limbiq Dependency Map

> Complete import graph for every module in `limbiq/` — verified by full codebase scan.

---

## Dependency Matrix

Each row is a module. Columns show what it imports from limbiq and from external packages.

### Foundation

| Module | limbiq Imports | External Imports |
|--------|---------------|-----------------|
| `types.py` | _(none)_ | dataclasses, enum, typing, time, uuid |

### Store Layer

| Module | limbiq Imports | External Imports |
|--------|---------------|-----------------|
| `store/memory_store.py` | types (Memory, MemoryTier, SuppressionReason) | json, os, sqlite3, threading, uuid, time, **numpy** |
| `store/embeddings.py` | _(none)_ | math, threading, collections.Counter, **sentence-transformers** (or sklearn TF-IDF) |
| `store/cluster_store.py` | types (KnowledgeCluster, Memory) | json, uuid, time |
| `store/rule_store.py` | types (BehavioralRule) | json, uuid, time |
| `store/signal_log.py` | types (SignalEvent) | json, uuid |

### Signals Layer

| Module | limbiq Imports | External Imports |
|--------|---------------|-----------------|
| `signals/base.py` | types (SignalEvent, Memory) | abc |
| `signals/dopamine.py` | signals.base (BaseSignal), types (SignalEvent, SignalType, Memory, MemoryTier, SuppressionReason) | _(none)_ |
| `signals/gaba.py` | signals.base (BaseSignal), types (SignalEvent, SignalType, Memory, SuppressionReason) | _(none)_ |
| `signals/serotonin.py` | signals.base (BaseSignal), types (SignalEvent, SignalType, Memory) | _(none)_ |
| `signals/acetylcholine.py` | signals.base (BaseSignal), types (SignalEvent, SignalType, Memory, MemoryTier) | collections.Counter |
| `signals/norepinephrine.py` | signals.base (BaseSignal), types (SignalEvent, SignalType, Memory, RetrievalConfig) | _(none)_ |

### Graph Layer

| Module | limbiq Imports | External Imports |
|--------|---------------|-----------------|
| `graph/store.py` | graph.entities (lazy: VALID_PREDICATES, RELATION_ALIASES in heal) | json, logging, uuid, time, dataclasses, typing |
| `graph/entities.py` | graph.store (Entity, Relation, GraphStore) | re, logging, typing, **spaCy** (opt: dep parsing Tier 1), **encoder.py** (opt: embeddings), **llm_fn** (opt: Tier 2 tie-break) |
| `graph/encoder.py` | _(none — receives EmbeddingEngine at init)_ | logging, re, dataclasses, typing, **torch** (opt) |
| `graph/inference.py` | graph.store (GraphStore, Relation, Entity) | _(none)_ |
| `graph/query.py` | graph.store (GraphStore), graph.inference (InferenceEngine) | re |
| `graph/propagation.py` | store.memory_store (MemoryStore, _deserialize_embedding), graph.store (GraphStore, Entity, Relation) | re, time, json, struct, logging, collections.defaultdict, dataclasses, typing |
| `graph/gnn.py` | _(none — receives store/graph/embeddings at init)_ | os, json, time, math, struct, logging, dataclasses, typing, **numpy**, **torch** |
| `graph/pattern_completion.py` | graph.store (GraphStore, Entity, Relation) | os, re, json, time, math, logging, sqlite3, dataclasses, typing, collections.defaultdict, **numpy**, **torch** |
| `graph/reasoning.py` | graph.store (GraphStore, Entity, Relation) | os, re, json, math, time, random, logging, dataclasses, typing, **numpy**, **torch** |

### Retrieval Layer

| Module | limbiq Imports | External Imports |
|--------|---------------|-----------------|
| `retrieval/activation_retrieval.py` | _(none — receives store/graph/embeddings at init)_ | logging, re, dataclasses, typing, **numpy** |

### Context Layer

| Module | limbiq Imports | External Imports |
|--------|---------------|-----------------|
| `context/builder.py` | types (Memory, BehavioralRule) | _(none)_ |

### Core

| Module | limbiq Imports | External Imports |
|--------|---------------|-----------------|
| `core.py` | types (6), store.memory_store, store.embeddings, store.signal_log, store.rule_store, store.cluster_store, context.builder, signals.dopamine, signals.gaba, signals.serotonin, signals.acetylcholine, signals.norepinephrine, graph.store, graph.entities, graph.inference, graph.query, retrieval.activation_retrieval (3 symbols) | logging, re, uuid, collections.abc, concurrent.futures |

### Public API

| Module | limbiq Imports | External Imports |
|--------|---------------|-----------------|
| `__init__.py` | core (LimbiqCore), types (9 symbols), graph.propagation, graph.gnn (try), graph.pattern_completion (try), graph.reasoning (try) | collections.abc |

### Playground Layer

| Module | limbiq Imports | External Imports |
|--------|---------------|-----------------|
| `playground/__init__.py` | playground.server (create_app) | _(none)_ |
| `playground/__main__.py` | playground.server (create_app, lazy) | sys, logging, argparse, **uvicorn**, **openai** (opt) |
| `playground/server.py` | playground.api (router), limbiq (Limbiq, lazy in lifespan) | os, time, logging, contextlib, **fastapi**, **fastapi.responses**, **fastapi.middleware.cors** |
| `playground/api.py` | playground.data_models (14 symbols) | time, logging, typing, **fastapi** (APIRouter, Query, Request, HTTPException) |
| `playground/data_models.py` | _(none)_ | **pydantic** (BaseModel), typing |

---

## Import Count Summary

| Module | limbiq imports | External imports | Total |
|--------|:-------------:|:----------------:|:-----:|
| `core.py` | **18** | 5 | 23 |
| `__init__.py` | **13** (4 optional) | 1 | 14 |
| `playground/api.py` | 14 (data_models) | 4 | 18 |
| `gnn.py` | 0 (runtime) | **8** (incl. torch) | 8 |
| `reasoning.py` | 3 | **10** (incl. torch) | 13 |
| `pattern_completion.py` | 3 | **12** (incl. torch) | 15 |
| `propagation.py` | 4 | 8 | 12 |
| All others | 1-3 | 0-4 | 1-7 |

---

## Circular Dependency Check

**Result: NO circular dependencies found.**

Verified import directions:
- `types.py` → imports nothing from limbiq
- `store/` → imports only from `types`
- `signals/` → imports only from `signals.base` and `types`
- `graph/` → imports from `store` and `graph.store` (same layer, no upward)
- `retrieval/` → runtime injection only (no static limbiq imports)
- `context/` → imports only from `types`
- `core.py` → imports from all layers (downward only)
- `__init__.py` → imports from `core` and re-exports
- `playground/` → imports from `limbiq` (public API) and `playground.data_models`

---

## External Dependency Tiers

```
Tier 1 — Required (core functionality)
  numpy ............... vectorized similarity, matrix ops
  sentence-transformers  text → 384-dim embeddings

Tier 2 — Optional Neural (graph intelligence)
  torch ............... GNN, TransE, micro-transformer, encoder
  spacy ............... Tier 1 dep parsing + NER (entities.py); falls back to regex

Tier 3 — Optional Playground (web dashboard)
  fastapi ............. REST API framework
  uvicorn ............. ASGI server
  pydantic ............ request/response validation
  openai .............. OpenAI-compatible LLM client
```

---

## Lazy Import Locations

These imports happen at runtime, not at module load:

| Location | Import | Reason |
|----------|--------|--------|
| `core.py:350` | `graph.pattern_completion.PatternCompletion` | torch may not be installed |
| `core.py:458` | `graph.store.Relation` | Inside connectivity healing loop |
| `core.py:584` | `graph.entities._normalize_predicate, VALID_PREDICATES` | Only needed on corrections |
| `core.py:698` | `graph.gnn.GNNPropagation` | torch may not be installed |
| `core.py:725` | `types.Memory, MemoryTier` | Backward compat conversion |
| `__init__.py:32-45` | `graph.gnn`, `graph.pattern_completion`, `graph.reasoning` | All wrapped in try/except |
| `__init__.py:173` | `retrieval.activation_retrieval.GraphTrainingDataGenerator` | Used rarely |
| `playground/__main__.py:39` | `playground.server.create_app` | Delay until args parsed |
| `playground/__main__.py:46` | `openai.OpenAI` | May not be installed |
| `graph/store.py:119` | `graph.entities.VALID_PREDICATES, RELATION_ALIASES` | Only needed during junk cleanup/heal |
| `playground/server.py:33` | `limbiq.Limbiq` | Delay until startup event |

---

*Updated 2026-03-23. Source: full `grep` scan of all `import`/`from` statements across 35 `.py` files (28 modules + 7 `__init__.py`) in `limbiq/`. entities.py updated to hybrid dep-parse pipeline (spaCy dep Tier 1, LLM Tier 2, regex fallback).*

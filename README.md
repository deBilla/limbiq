# Limbiq

Neurotransmitter-inspired adaptive memory layer for LLMs.

Limbiq sits between the user and any LLM, enriching context with learned memories, behavioral rules, and knowledge graph facts — all without modifying model weights. Five signal types inspired by human brain chemistry control what gets remembered, suppressed, clustered, or flagged.

```
User Message → Limbiq.process() → Enriched Context → Any LLM → Response → Limbiq.observe() → Loop
```

## Installation

```bash
pip install limbiq                    # Core library
pip install limbiq[faiss]             # + FAISS vector search (recommended)
pip install limbiq[dev]               # + pytest
pip install limbiq[playground]        # + FastAPI web dashboard
```

Limbiq requires Python 3.10+ and downloads the `all-MiniLM-L6-v2` sentence-transformer model (~80MB) on first run.

For entity extraction, install the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Quick Start

```python
from limbiq import Limbiq

lq = Limbiq(
    store_path="./neuro_data",
    user_id="dimuthu",
    llm_fn=my_llm,  # Any callable(str) -> str, or None for heuristic mode
)

# Before sending to LLM — retrieve memories and build context
result = lq.process("What's my wife's name?")

# Inject result.context into your system prompt
messages = [
    {"role": "system", "content": f"You are a helpful assistant.\n\n{result.context}"},
    {"role": "user", "content": "What's my wife's name?"},
]
response = my_llm(messages)

# After getting response — let Limbiq observe and learn
lq.observe("What's my wife's name?", response)

# End session — graph inference, entity resolution, memory aging
lq.end_session()
```

`llm_fn` is optional. Without it, Limbiq runs in heuristic mode — spaCy + regex extraction, no LLM-powered compression or tie-breaking.

## The Five Signals

Signals fire automatically in `observe()` based on conversation content.

### Dopamine — "This matters, remember it"

Fires when the user shares personal info, corrections, or positive feedback. Tagged memories are **always** included in context.

```python
# Automatic — fires when patterns like "my name is..." are detected
# Manual:
lq.dopamine("User's wife is named Prabhashi")
```

### GABA — "Suppress this, let it fade"

Fires on denials, contradictions, or stale data. Suppression is soft — memories can be restored.

```python
lq.gaba(memory_id="abc123")
lq.restore_memory("abc123")  # Undo suppression
```

### Serotonin — "This pattern is stable, make it a rule"

Watches for recurring patterns across sessions. After 3+ observations across 2+ sessions, crystallizes into a permanent behavioral rule injected into every future context.

```python
rules = lq.get_active_rules()
lq.deactivate_rule(rule_id)
lq.reactivate_rule(rule_id)
```

### Acetylcholine — "Go deep here, build expertise"

Detects sustained interest in a topic (3+ turns) and creates knowledge clusters — grouped memories loaded as a unit when the topic returns.

```python
clusters = lq.get_clusters()
memories = lq.get_cluster_memories(cluster_id)
```

### Norepinephrine — "Something changed, be careful"

Fires on topic shifts or frustration. Temporarily widens memory retrieval and adds caution flags. Effects reset after each `process()` call.

## Knowledge Graph

Limbiq automatically builds a knowledge graph from conversations using a hybrid extraction pipeline:

1. **spaCy dependency parsing** — possessive relations ("my wife Prabhashi"), copular patterns, chained predicates ("my wife's father")
2. **LLM tie-break** — resolves ambiguous fragments when `llm_fn` is provided
3. **Regex fallback** — 20+ patterns for common relationships
4. **Fuzzy predicate matching** — catches typos (Levenshtein distance ≤ 1)

The graph supports transitive inference (in-laws, grandparents, co-parents) and self-heals on every `observe()` call — cleaning junk entities, bridging disconnected components, and re-running inference.

```python
lq.query_graph("Who is Prabhashi?")    # Direct graph lookup
lq.get_world_summary()                  # Compact user profile from graph
lq.get_entities()                        # All known entities
lq.get_relations()                       # All relationships (including inferred)
lq.describe_entity("Prabhashi")          # Natural language description
lq.heal_graph()                          # Manual self-heal trigger
```

### Graph Pipeline (5 Phases)

For advanced use, Limbiq includes a multi-phase graph intelligence pipeline:

```python
# Phase 1: Rule-based propagation — suppress noise, merge duplicates, repair graph
result = lq.propagate()

# Phase 2: GNN propagation — Graph Attention Network learns activation scores
#   3-layer GAT, 4 heads, 128-dim, ~2-5M params, trains on Phase 1 labels
result = lq.propagate_gnn(train_first=True, epochs=200)

# Phase 3: Pattern completion — TransE entity resolution + link prediction
#   ~50K params, merges duplicate entities, predicts missing relations
result = lq.run_pattern_completion(train_transe=True, epochs=500)

# Phase 4: Activation retrieval — hybrid scoring
#   final_score = α·embedding_sim + β·gnn_activation + γ·graph_relevance
lq.enable_activation_retrieval()

# Phase 5: Graph reasoning — micro-transformer for graph QA
#   ~50K params, 3 modes: entity pointer, boolean, count
lq.train_reasoner(epochs=100)
answer = lq.reason("Who is Dimuthu's father-in-law?")
```

Phases 2-5 require PyTorch (`pip install torch`).

## Vector Search

Memory retrieval uses **FAISS** (Facebook AI Similarity Search) when available, falling back to numpy brute-force.

- FAISS `IndexFlatIP` on normalized embeddings = exact cosine similarity
- Incremental adds/removes — no full rebuild on each memory change
- Index persisted to disk (`{user_id}.faiss`) between sessions
- Thread-safe via `threading.Lock`

```bash
pip install faiss-cpu  # ~15MB, no GPU needed
```

## Corrections

Combines Dopamine + GABA — stores new info as priority, suppresses contradicting memories.

```python
lq.correct("User works at Bitsmedia, not Google")
```

## Playground

Interactive web dashboard for exploring the knowledge graph, chatting, and inspecting signals.

```bash
pip install limbiq[playground]

# Heuristic mode (no LLM needed)
python -m limbiq.playground

# With Ollama
python -m limbiq.playground --llm-url http://localhost:11434/v1 --llm-model llama3.1

# With OpenAI
python -m limbiq.playground --llm-url https://api.openai.com/v1 --llm-model gpt-4o --llm-api-key sk-...
```

Open `http://localhost:8765` after starting. The dashboard includes:

- **Chat** — talk to Limbiq and watch it learn in real time
- **Knowledge Graph** — interactive D3 visualization of entities and relations
- **Entity Explorer** — browse entities and their relationships
- **Query Builder** — test graph queries and memory retrieval

## Inspection

```python
lq.get_stats()              # Memory counts per tier
lq.get_signal_log()         # Full history of signals fired
lq.get_priority_memories()  # All dopamine-tagged memories
lq.get_suppressed()         # All GABA-suppressed memories
lq.get_full_profile()       # Complete user profile across all signals
lq.export_state()           # Full JSON export for debugging
lq.get_graph_stats()        # Entity/relation counts
lq.get_graph_connectivity() # Connected component analysis
```

## How It Works

- **LLM-agnostic** — works with any LLM via a callable, or runs without one in heuristic mode
- **Zero weight modification** — all adaptation through context enrichment
- **Knowledge graph** — entities and relations extracted automatically, inferred transitively, self-healed continuously
- **FAISS vector search** — fast approximate nearest neighbor retrieval with numpy fallback
- **5-phase graph pipeline** — GNN, TransE, and micro-transformer for deep graph intelligence (optional, requires PyTorch)
- **SQLite persistence** — memories, graph, rules, and clusters survive across sessions
- **Semantic search** — 384-dim sentence-transformer embeddings (all-MiniLM-L6-v2) with TF-IDF fallback
- **Transparent** — every signal is logged with trigger, timestamp, and effect
- **Reversible** — suppressed memories can be restored, rules deactivated
- **Thread-safe** — per-thread SQLite connections, thread-safe FAISS index and embedding cache

## Architecture

```
┌──────────────────────────────────────────────────┐
│              Limbiq Public API                     │
├──────────────────────────────────────────────────┤
│              LimbiqCore (orchestrator)             │
│       process() → observe() → end_session()       │
├─────────┬──────────┬──────────┬──────────────────┤
│  Store  │ Signals  │  Graph   │ Retrieval+Context │
│  Layer  │  Layer   │  Layer   │     Layer         │
├─────────┴──────────┴──────────┴──────────────────┤
│      Shared SQLite DB + FAISS Vector Index        │
└──────────────────────────────────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full system topology, data flow diagrams, and module reference.

## Development

```bash
git clone https://github.com/deBilla/limbiq.git
cd limbiq
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
python -m pytest tests/ -x -q
```

CI runs tests on Python 3.10–3.13 via GitHub Actions. Releases are published to PyPI automatically when a GitHub Release is created.

## License

MIT

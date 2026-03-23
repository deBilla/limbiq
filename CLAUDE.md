# Limbiq — Claude Code Project Guide

## What is this project?

Limbiq is a neurotransmitter-inspired adaptive learning layer for LLMs. It sits between the user and any LLM, enriching context with learned memories, behavioral rules, knowledge graph facts, and signal-driven adaptations — all without modifying model weights.

## Quick start

```bash
pip install -e ".[dev]"             # Install with dev dependencies
python -m pytest tests/ -x -q      # Run tests (~255 tests, ~5 min with model loading)
python -m limbiq.playground         # Launch web dashboard on :8765
```

## Architecture

```
User Message → Limbiq.process() → Enriched Context → Any LLM → Response → Limbiq.observe() → Loop
```

Core loop: `process()` → external LLM → `observe()` → `end_session()`

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full diagrams and module reference.

## Key directories

```
limbiq/
├── core.py              # Orchestrator — process(), observe(), end_session()
├── __init__.py          # Public API facade (Limbiq class)
├── types.py             # All dataclasses and enums
├── store/               # SQLite persistence + embeddings
├── signals/             # 5 neurotransmitter signals
├── graph/               # Knowledge graph + GNN + reasoning
├── retrieval/           # Activation-weighted retrieval
├── hallucination/       # Fact checking (grounding + NLI verification)
├── compression/         # Session compression
├── context/             # Context string assembly
├── router.py            # LLM swarm routing
├── intent.py            # Intent classification
├── routing.py           # Smart query routing
├── nli.py               # NLI cross-encoder
├── tools.py             # Tool registry (file, terminal, calc, web)
├── playground/          # FastAPI web dashboard
└── steering/            # Activation steering (experimental, MLX)
tests/                   # pytest suite
docs/                    # Architecture documentation
```

## Rules

### Architecture rules
- Every feature integrates through `process()`, `observe()`, or `end_session()` — no parallel paths
- All persistent state goes through `MemoryStore` — never create separate SQLite connections
- All embeddings go through `EmbeddingEngine` — never call sentence-transformers directly
- Layer deps flow one direction: types → store → signals → graph → retrieval → hallucination → core
- No circular imports

### Code style
- Type hints on all public methods
- `logger = logging.getLogger(__name__)` at module level
- SQLite: parameterized queries (`?` placeholders), never f-strings with user data
- Imports: stdlib → third-party → limbiq (sorted within groups)
- `np.array(..., dtype=np.float32)` for all embedding operations

### Performance constraints
- `process()` must complete in <50ms (excluding LLM calls)
- `observe()` must complete in <100ms (excluding LLM calls)
- Embedding calls ~5ms each — cache aggressively
- Invalidate embedding cache (`_emb_dirty = True`) when memories change

### Testing
- Run: `python -m pytest tests/ -x -q`
- Tests must work WITHOUT a real LLM — use `llm_fn=None`
- Use `tmp_dir` fixture from conftest for test isolation
- New modules need a corresponding `tests/test_<module>.py`

### Security
- `TerminalTool`: `shell=False` + metachar rejection, never `shell=True`
- `FileReaderTool`: path allowlisting with `realpath()` normalization
- `WebFetchTool`: blocks private IP ranges (RFC 1918, loopback, link-local)
- CORS: specific origins only, no wildcard with credentials
- Never swallow exceptions silently — always `logger.warning()` at minimum

## Shared state

All stores share ONE SQLite database via `MemoryStore`. Thread safety is maintained via:
- `threading.local()` for per-thread DB connections (`self._store.db`)
- `threading.Lock` for embedding cache rebuilds (`self._emb_lock`)

## Signals

| Signal | Trigger | Effect |
|--------|---------|--------|
| Dopamine | Personal info, corrections | Boosts to PRIORITY tier |
| GABA | Denials, contradictions | Suppresses memory |
| Serotonin | Recurring patterns (3+) | Crystallizes behavioral rule |
| Acetylcholine | Sustained topic interest | Creates knowledge cluster |
| Norepinephrine | Topic shifts, frustration | Widens retrieval + caution flag |

## Graph pipeline phases

1. Rule-based propagation (`propagation.py`)
2. GNN activation scoring (`gnn.py` — requires torch)
3. TransE pattern completion (`pattern_completion.py` — requires torch)
4. Activation-weighted retrieval (`activation_retrieval.py`)
5. Micro-transformer reasoning (`reasoning.py` — requires torch)

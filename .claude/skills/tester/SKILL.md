---
name: limbiq-tester
description: "Testing agent for the limbiq codebase. Use when writing tests, running the test suite, diagnosing test failures, improving test coverage, creating test fixtures, or validating that changes don't break existing behavior. Triggers: 'test', 'write tests for', 'run tests', 'why is this test failing', 'coverage', 'test plan', 'validate', 'verify'."
---

# Limbiq Tester

You are a testing agent for the **limbiq** library. You write thorough tests, diagnose failures, and ensure code quality.

## Running Tests

```bash
cd /path/to/limbiq
pip install pytest --break-system-packages  # if not installed
python -m pytest tests/ -x -q              # fast: stop on first failure
python -m pytest tests/ -v                 # verbose: see all test names
python -m pytest tests/test_core.py -x -q  # single file
python -m pytest tests/ -k "test_dopamine" # by name pattern
```

## Test Structure

```
tests/
├── conftest.py            # Shared fixtures (tmp_dir, etc.)
├── test_core.py           # Core pipeline: process → observe → end_session
├── test_store.py          # MemoryStore: store, search, suppress, age
├── test_dopamine.py       # Dopamine signal detection + application
├── test_gaba.py           # GABA signal detection + suppression
├── test_serotonin.py      # Serotonin pattern analysis + rule creation
├── test_acetylcholine.py  # Topic detection + cluster management
├── test_norepinephrine.py # Topic shift + retrieval widening
├── test_graph.py          # Graph store: entities, relations, queries
├── test_extraction.py     # Entity/relation extraction pipeline
├── test_context.py        # Context builder output
├── test_detection.py      # Signal detection patterns
├── test_hallucination.py  # Grounding + verification + self-correction
├── test_search.py         # Search client
├── test_bridge.py         # Integration tests
└── test_steering.py       # Behavioral steering tests
```

## Writing Tests — Conventions

### 1. Always use temporary directories
```python
import tempfile
from limbiq import Limbiq

def test_something(self, tmp_dir):  # tmp_dir from conftest
    lq = Limbiq(store_path=tmp_dir, user_id="test")
    # ...

# OR without fixture:
def test_something():
    with tempfile.TemporaryDirectory() as tmp:
        lq = Limbiq(store_path=tmp, user_id="test")
```

### 2. No real LLM required
All tests must work with `llm_fn=None`. The system gracefully handles missing LLM:
- Compression falls back to regex/heuristic extraction
- Entity extraction uses spaCy + regex (no LLM tier)
- Serotonin/Acetylcholine skip LLM-dependent features

### 3. Test the signal loop
The canonical test pattern for signals:
```python
def test_signal_fires(self, tmp_dir):
    lq = Limbiq(store_path=tmp_dir, user_id="test")
    lq.start_session()

    # Trigger the signal
    events = lq.observe("trigger message", "response")

    # Verify signal fired
    signal_events = [e for e in events if e.signal_type == SignalType.EXPECTED]
    assert len(signal_events) >= 1

    # Verify side effect
    # (e.g., memory stored, rule created, cluster formed)
```

### 4. Test graph state
```python
def test_graph_extraction(self, tmp_dir):
    lq = Limbiq(store_path=tmp_dir, user_id="Dimuthu")
    lq.start_session()
    lq.observe("my wife is Prabhashi", "Nice name!")

    entities = lq.get_entities()
    entity_names = {e.name for e in entities}
    assert "Prabhashi" in entity_names

    relations = lq.get_relations(include_inferred=False)
    wife_rels = [r for r in relations if r.predicate == "wife"]
    assert len(wife_rels) >= 1
```

### 5. Test session lifecycle
```python
def test_full_session(self, tmp_dir):
    lq = Limbiq(store_path=tmp_dir, user_id="test")
    lq.start_session()
    result = lq.process("hello")
    assert result.context is not None
    lq.observe("hello", "hi there")
    stats = lq.end_session()
    assert "compressed" in stats
```

## Test Categories

### Unit Tests
- Single module, isolated behavior
- Mock external dependencies
- Fast (<1s per test)

### Integration Tests
- Multiple modules interacting
- Full pipeline: process → observe → end_session
- Graph + signals + store together
- May be slower (1-5s due to embedding model loading)

### Regression Tests
When fixing a bug, ALWAYS write a test that reproduces the bug first:
```python
def test_dog_dexter_entity_resolution():
    """Regression: generic 'Dog' entity should merge with named 'Dexter'."""
    # Setup: create the buggy state
    # Action: run entity resolution
    # Assert: Dog merged into Dexter, relations transferred
```

## Diagnosing Failures

When a test fails:

1. **Read the assertion error** — What was expected vs actual?
2. **Check if it's a data issue** — Tests using pre-existing data (like `test_hallucination.py` with `/sessions/.../test_limbiq_data/`) can fail if the data is stale or dimensions mismatch.
3. **Check for dimension mismatches** — If you see `ValueError: matmul: Input operand 1 has a mismatch`, the embedding model changed dimensions. The `memory_store.py` has a `_search_fallback()` method for this.
4. **Check for import errors** — New dependencies (spaCy models, torch) might not be installed.
5. **Check for shared state** — Tests that don't use `tmp_dir` might leak state between runs.

## Known Test Gotchas

- `test_hallucination.py` uses a pre-existing SQLite DB at a fixed path — it's an integration test, not unit test
- `sentence-transformers/all-MiniLM-L6-v2` warning about "No model found" is normal — it creates one with mean pooling
- Some tests return values (triggering `PytestReturnNotNoneWarning`) — these are informational, not failures
- The `test_acetylcholine.py::test_cluster_grows_over_sessions` test relies on topic detection matching across sessions — bigram topics must share words to match clusters

## Coverage Gaps to Fill

If asked to improve coverage, prioritize:
1. `router.py` — No tests yet for LLMRouter
2. `pattern_completion.py` — EntityResolver._semantic_entity_resolution() needs tests
3. `graph/gnn.py` — GNN training/propagation (needs torch)
4. `graph/reasoning.py` — Micro-transformer training/inference
5. Edge cases in `entities.py` — Chained possessives, LLM extraction validation

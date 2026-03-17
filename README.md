# Limbiq

Neurotransmitter-inspired adaptive learning layer for LLMs.

Limbiq makes any LLM appear to learn and adapt across conversations — without touching a single weight. It sits between the user and the LLM, modifying what the model sees through five discrete signal types inspired by human brain chemistry.

```
User → Limbiq → Modified Context → Any LLM → Response → Limbiq observes → Loop
```

## Installation

```bash
pip install limbiq
```

## Quick Start

```python
from limbiq import Limbiq

# Initialize
lq = Limbiq(
    store_path="./neuro_data",
    user_id="dimuthu",
)

# Before sending to LLM — get enriched context
result = lq.process(
    message="What's my wife's name?",
    conversation_history=[
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ],
)

# Inject result.context into your system prompt
messages = [
    {"role": "system", "content": f"You are a helpful assistant.\n\n{result.context}"},
    {"role": "user", "content": "What's my wife's name?"},
]
response = my_llm(messages)  # Any LLM

# After getting response — let Limbiq observe and learn
lq.observe("What's my wife's name?", response)

# End session — triggers memory compression
lq.end_session()
```

## Signals (v0.1)

### Dopamine — "This matters, remember it"

Fires when the user shares personal info, corrects the model, or gives positive feedback. Tagged memories are **always** included in context.

```python
lq.dopamine("User's wife is named Prabhashi")
```

### GABA — "Suppress this, let it fade"

Fires when memories are denied, contradicted, or go stale. Suppression is soft — memories can be restored.

```python
lq.gaba(memory_id="abc123")
lq.restore_memory("abc123")  # Undo suppression
```

### Corrections

Combines both signals — stores new info as priority, suppresses the old.

```python
lq.correct("User works at Bitsmedia, not Google")
```

## Inspection

```python
lq.get_stats()              # Memory counts per tier
lq.get_signal_log()         # Full history of signals fired
lq.get_priority_memories()  # All dopamine-tagged memories
lq.get_suppressed()         # All GABA-suppressed memories
lq.export_state()           # Full JSON export for debugging
```

## How It Works

- **LLM-agnostic** — works with any LLM (OpenAI, Anthropic, Ollama, llama.cpp, etc.)
- **Zero weight modification** — all adaptation through context manipulation
- **SQLite persistence** — memories survive across sessions
- **Semantic search** — uses sentence-transformers for embedding-based retrieval (falls back to TF-IDF if not installed)
- **Transparent** — every signal is logged with trigger, timestamp, and effect
- **Reversible** — suppressed memories can be restored, nothing is permanently destructive

## License

MIT

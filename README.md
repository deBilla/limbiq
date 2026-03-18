# Limbiq

Neurotransmitter-inspired adaptive learning layer for LLMs.

Limbiq makes any LLM appear to learn and adapt across conversations — without touching a single weight. It sits between the user and the LLM, modifying what the model sees through five discrete signal types inspired by human brain chemistry.

```
User → Limbiq → Modified Context → Any LLM → Response → Limbiq observes → Loop
```

## Installation

```bash
pip install limbiq                    # Core — text-based signals + knowledge graph
pip install limbiq[steering-mlx]      # + MLX activation steering (Apple Silicon)
```

## Quick Start

```python
from limbiq import Limbiq

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

# End session — triggers memory compression + graph inference
lq.end_session()
```

## The Five Signals

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

### Serotonin — "This pattern is stable, make it a rule"

Watches for recurring patterns across sessions. After 3+ observations across 2+ sessions, crystallizes into a permanent behavioral rule injected into every future context.

```python
# Automatic — fires when patterns like "user always writes short messages" repeat
rules = lq.get_active_rules()
lq.deactivate_rule(rule_id)   # Turn off a wrong rule
lq.reactivate_rule(rule_id)   # Turn it back on
```

### Acetylcholine — "Go deep here, build expertise"

Detects sustained interest in a topic and creates knowledge clusters — grouped collections of memories loaded as a unit when the topic returns.

```python
clusters = lq.get_clusters()
memories = lq.get_cluster_memories(cluster_id)
```

### Norepinephrine — "Something changed, be careful"

Fires on topic shifts, user frustration, or contradictions. Temporarily widens memory retrieval and adds caution flags. Effects reset after each `process()` call.

## Knowledge Graph

Limbiq automatically builds a knowledge graph from conversations. Entities and relationships are extracted, inferred, and used to answer questions with zero LLM cost.

```python
lq.query_graph("Who is Prabhashi?")    # Direct graph lookup
lq.get_world_summary()                  # Compact summary of everything known
lq.get_entities()                        # All known entities
lq.get_relations()                       # All relationships (including inferred)
```

## Activation Steering (Experimental)

Beyond text injection, limbiq can modify the model's internal representations at inference time using learned direction vectors.

```python
from limbiq import Limbiq
from limbiq.steering import enable_steering

lq = Limbiq(store_path="./data", user_id="test")
steered = enable_steering(lq, model_path="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")

# Signals now operate at the activation level
result = steered.generate("What's my wife's name?")
# Dopamine fires → memory_attention vector injected → model attends to memory
```

8 steering dimensions: conciseness, formality, technical depth, creativity, confidence, helpfulness, honesty, memory attention.

## Corrections

Combines Dopamine + GABA — stores new info as priority, suppresses the old.

```python
lq.correct("User works at Bitsmedia, not Google")
```

## Inspection

```python
lq.get_stats()              # Memory counts per tier
lq.get_signal_log()         # Full history of signals fired
lq.get_priority_memories()  # All dopamine-tagged memories
lq.get_suppressed()         # All GABA-suppressed memories
lq.get_full_profile()       # Complete user profile across all signals
lq.export_state()           # Full JSON export for debugging
```

## How It Works

- **LLM-agnostic** — works with any LLM (OpenAI, Anthropic, Ollama, llama.cpp, etc.)
- **Zero weight modification** — all adaptation through context manipulation and activation steering
- **Knowledge graph** — entities and relations extracted automatically, inferred transitively
- **SQLite persistence** — memories, graph, and rules survive across sessions
- **Semantic search** — sentence-transformers for embedding-based retrieval (TF-IDF fallback)
- **Transparent** — every signal is logged with trigger, timestamp, and effect
- **Reversible** — suppressed memories can be restored, rules deactivated, nothing permanently destructive
- **Thread-safe** — per-thread SQLite connections for Gradio, FastAPI, etc.

## License

MIT

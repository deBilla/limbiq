# Limbiq MCP Server

Persistent adaptive memory for [Jan](https://jan.ai), [Claude Desktop](https://claude.ai), or any MCP-compatible AI app — powered by [Limbiq](https://pypi.org/project/limbiq/).

## What it does

Gives your LLM long-term memory across conversations using neurotransmitter-inspired signals:

- **Dopamine** — stores important facts (names, preferences, relationships)
- **GABA** — suppresses contradicted or denied information
- **Serotonin** — crystallizes recurring behavioral patterns into rules
- **Acetylcholine** — builds knowledge clusters around sustained topics
- **Norepinephrine** — detects topic shifts and widens retrieval

Signal detection uses a unified self-attention encoder that learns from conversation context — no hardcoded keyword patterns. Entities carry per-entity state (activation, receptor density, signal history) inspired by biological cellular memory.

## Setup

### 1. Install dependencies

```bash
pip install limbiq mcp
```

### 2. Configure in your MCP client

#### Jan

Open Jan **Settings > MCP Servers**, click the JSON editor, and add:

```json
{
  "limbiq-memory": {
    "command": "python3",
    "args": ["/path/to/limbiq/mcp-server/server.py"],
    "env": {
      "LIMBIQ_USER_ID": "your_name",
      "LIMBIQ_STORE_PATH": "/path/to/your/neuro_data"
    }
  }
}
```

#### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "limbiq-memory": {
      "command": "python3",
      "args": ["/path/to/limbiq/mcp-server/server.py"],
      "env": {
        "LIMBIQ_USER_ID": "your_name",
        "LIMBIQ_STORE_PATH": "/path/to/your/neuro_data"
      }
    }
  }
}
```

### 3. Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIMBIQ_STORE_PATH` | `./neuro_data` | Directory for SQLite + FAISS data |
| `LIMBIQ_USER_ID` | `jan_user` | User identifier for memory isolation |

### 4. Use it

Chat normally. The LLM will call memory tools automatically when configured with an appropriate system prompt:

- `memory_recall` — before responding, to load relevant context
- `memory_observe` — after responding, to store what happened
- `memory_query` — for direct factual questions ("what's my wife's name?")
- `memory_profile` — to see everything it knows about the user

### 5. Inspect the graph (optional)

Run the Limbiq Playground pointing at the **same data directory** the MCP server uses:

```bash
pip install limbiq[playground]
python3 -m limbiq.playground \
  --store-path /path/to/your/neuro_data \
  --user-id your_name \
  --port 8765
```

Open http://localhost:8765 to see a live D3 knowledge graph, signal log, entities, and world summary — all reading from the same data that the MCP server writes to.

## Tools

| Tool | Purpose |
|------|---------|
| `memory_recall` | Retrieve relevant memories before generating a response |
| `memory_observe` | Store a completed exchange for learning |
| `memory_query` | Ask the knowledge graph a factual question |
| `memory_profile` | Get full user profile (facts, rules, clusters, graph) |
| `memory_start_session` | Begin a new conversation session |
| `memory_end_session` | End session and persist state |

## How it works

```
User message
    │
    ├─► memory_recall(message)
    │       └─► Limbiq.process() → enriched context + memories
    │
    ├─► LLM generates response (with memory context)
    │
    └─► memory_observe(message, response)
            └─► Limbiq.observe() → fires signals:
                  Dopamine  → stores important facts
                  GABA      → suppresses contradictions
                  Serotonin → crystallizes behavioral rules
                  Acetylcholine → builds topic clusters
                  Norepinephrine → detects topic shifts
```

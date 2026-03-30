"""
Limbiq MCP Server — persistent adaptive memory for any MCP-compatible AI app.

Tools:
  - memory_recall       : retrieve relevant memories + context before responding
  - memory_observe      : store what happened in an exchange
  - memory_query        : ask the knowledge graph a question
  - memory_profile      : get everything Limbiq knows about the user
  - memory_start_session: start a new session
  - memory_end_session  : end session and run cleanup

Environment variables:
  LIMBIQ_STORE_PATH : path to the data directory (default: ./neuro_data)
  LIMBIQ_USER_ID    : user identifier (default: jan_user)
"""

import json
import logging
import os
import tempfile
import warnings

# Suppress noisy library output before any imports
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Route logging to a temp file so it never pollutes MCP stdio
logging.basicConfig(
    filename=os.path.join(tempfile.gettempdir(), "limbiq_mcp.log"),
    level=logging.WARNING,
)
for _name in ["limbiq", "sentence_transformers", "transformers", "torch", "httpx"]:
    logging.getLogger(_name).setLevel(logging.ERROR)

from mcp.server.fastmcp import FastMCP
from limbiq import Limbiq

STORE_PATH = os.environ.get("LIMBIQ_STORE_PATH", os.path.join(os.path.expanduser("~"), ".limbiq", "data"))
USER_ID = os.environ.get("LIMBIQ_USER_ID", "jan_user")

lq = Limbiq(store_path=STORE_PATH, user_id=USER_ID)
lq.start_session()

mcp = FastMCP("limbiq-memory")


@mcp.tool()
def memory_recall(message: str) -> str:
    """Retrieve relevant memories and context for a user message.

    Call this BEFORE generating a response to enrich your context with
    what you know about the user from previous conversations.

    Args:
        message: The user's current message
    """
    try:
        result = lq.process(message)
        return json.dumps({
            "context": result.context,
            "memories_retrieved": result.memories_retrieved,
            "priority_count": result.priority_count,
            "active_rules": [r.rule_text for r in result.active_rules],
            "norepinephrine_active": result.norepinephrine_active,
            "signals_fired": [
                {"type": s.signal_type.value, "trigger": s.trigger}
                for s in result.signals_fired
            ],
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "context": "", "memories_retrieved": 0})


@mcp.tool()
def memory_observe(user_message: str, assistant_response: str) -> str:
    """Store a completed exchange so Limbiq can learn from it.

    Call this AFTER you have responded to the user. Limbiq will
    automatically detect important facts (dopamine), contradictions (GABA),
    recurring patterns (serotonin), topic focus (acetylcholine), and
    topic shifts (norepinephrine).

    Args:
        user_message: What the user said
        assistant_response: What you replied
    """
    try:
        signals = lq.observe(user_message, assistant_response)
        return json.dumps({
            "signals_fired": [
                {
                    "type": s.signal_type.value,
                    "trigger": s.trigger,
                    "memories_affected": len(s.memory_ids_affected),
                }
                for s in signals
            ],
            "total_signals": len(signals),
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "total_signals": 0})


@mcp.tool()
def memory_query(question: str) -> str:
    """Ask the knowledge graph a natural language question.

    Use this for direct factual questions about the user like
    "What is the user's wife's name?" or "Where does the user work?"

    Args:
        question: Natural language question about the user
    """
    try:
        graph_answer = lq.query_graph(question)
        # get_world_summary can trigger DB writes (inference), wrap separately
        try:
            world_summary = lq.get_world_summary()
        except Exception:
            world_summary = ""
        return json.dumps({
            "graph_answer": graph_answer,
            "world_summary": world_summary,
        }, indent=2, default=str)
    except Exception as e:
        logging.getLogger("limbiq").error(f"memory_query failed: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool()
def memory_profile() -> str:
    """Get everything Limbiq knows about the user.

    Returns priority facts, behavioral rules, knowledge clusters,
    knowledge graph entities, and memory statistics. Use this to
    understand the full picture of who the user is.
    """
    try:
        return json.dumps(lq.get_full_profile(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def memory_start_session() -> str:
    """Start a new Limbiq session.

    Call this at the beginning of a new conversation to reset
    session-level tracking (turn counts, topic detection, etc).
    """
    try:
        lq.start_session()
        return json.dumps({"status": "session_started"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def memory_end_session() -> str:
    """End the current Limbiq session and run cleanup.

    This triggers graph inference, memory aging, and FAISS persistence.
    Call this when the conversation is ending.
    """
    try:
        return json.dumps(lq.end_session(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

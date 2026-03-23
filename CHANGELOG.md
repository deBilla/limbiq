# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Intent classification module (`intent.py`) — rule-based query intent detection for 7 types: personal, factual, creative, meta, greeting, command, statement
- NLI contradiction detection module (`nli.py`) — cross-encoder semantic entailment/contradiction checking
- LLM Router for swarm architecture (`router.py`) — task-type routing across multiple LLM agents with confidence scoring
- Smart query routing (`routing.py`) — skips LLM when graph/templates can answer directly
- `QueryRoutingDecision` class (renamed from ambiguous `RoutingDecision` in routing.py)
- Thread-safe embedding cache with `threading.Lock` in `MemoryStore`
- Input validation in `NLIChecker.check()` — returns neutral for None/empty inputs
- `NorepinephrineSignal.reset()` method for session boundary cleanup
- Auto `end_session()` when `start_session()` called with non-empty buffer
- Exception safety in `end_session()` — raw conversation always saved even if compression fails
- Architecture documentation (`docs/ARCHITECTURE.md`)
- Test suite for intent classifier (`test_intent.py`)
- Test suite for NLI checker (`test_nli.py`)
- Test suite for LLM router (`test_router.py`)
- Test suite for query routing (`test_routing.py`)
- `CLAUDE.md` project guide for Claude Code
- `CHANGELOG.md` (this file)
- Torch-guarded imports in `__init__.py` — GNN/reasoning features gracefully unavailable without PyTorch
- Return type hints on all `Limbiq` public methods
- Type hint for `llm_fn` parameter (`Callable | None`)
- 16 missing symbols added to `__all__` in `__init__.py`

### Fixed
- **Critical:** GAT attention parameters (`a_src`, `a_dst`) were initialized to a throwaway tensor — `xavier_uniform_` wrote to a copy, not the actual parameters (`gnn.py`)
- **Critical:** Embedding cache ignored `include_suppressed` flag on subsequent searches — second call reused stale cache built with different flag (`memory_store.py`)
- **Critical:** Command injection via `shell=True` in `TerminalTool` — switched to `shell=False` with `shlex.split()` and shell metacharacter rejection (`tools.py`)
- **Critical:** Path traversal in `FileReaderTool` — added `allowed_dirs` + `realpath()` normalization + sensitive path blocking (`tools.py`)
- Double retrieval widening in `NorepinephrineSignal.detect_for_process()` — `widen()` called twice for topic shifts (`norepinephrine.py`)
- GNN `propagate()` bypassed `MemoryStore` API for direct SQL updates without invalidating embedding cache (`gnn.py`)
- `restore()` did not set `_emb_dirty = True` — restored memories missing from search results (`memory_store.py`)
- Norepinephrine `_previous_embedding` persisted across sessions causing spurious topic shift on first message (`norepinephrine.py`)
- SSRF in `WebFetchTool` — now blocks private/internal IP ranges (`tools.py`)
- Wildcard CORS with credentials in playground — restricted to specific localhost origins (`server.py`)
- 7 silent `except Exception: pass` blocks replaced with `logger.warning()` in `core.py`, `graph/store.py`, `playground/api.py`
- Broken `test_hallucination.py` — rewritten as proper pytest with fixtures and assertions

### Changed
- `RoutingDecision` in `routing.py` renamed to `QueryRoutingDecision` (backward-compat alias preserved)
- `__init__.py` now wraps torch-dependent imports (`GNNPropagation`, `PatternCompletion`, `GraphReasoner`, `ReasoningResult`) in `try/except ImportError`

### Removed
- Dead code in `embeddings.py`: unused `_transformer_embed()` and `_tfidf_embed()` methods (cached versions are used instead)

## [0.4.3] — 2025-05-01
### Added
- spaCy NER for entity extraction (falls back to regex)
- Hallucination detection: grounding analyzer, fact verifier, detector
- Memory view in playground

## [0.4.2] — 2025-04-15
### Added
- Onboarding and user profiling
- Tool registry (file reader, terminal, calculator, web fetch)
- Token budget management
- Fact-checking pipeline
- Chain-of-thought reasoning
- Multi-model routing

## [0.4.1] — 2025-04-01
### Added
- Web search with auto-trigger on LLM uncertainty
- SearXNG, Brave, and Tavily provider support

## [0.4.0] — 2025-03-15
### Added
- Knowledge graph with entity/relation storage
- Interactive playground (FastAPI + D3)
- Generic LLM client (OpenAI-compatible)
- Web search integration

## [0.3.0] — 2025-03-01
### Added
- All five neurotransmitter signals
- Knowledge graph foundation
- Activation steering (experimental)

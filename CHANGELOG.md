# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.5.0] — 2026-03-23

### Added
- **Hybrid entity extraction pipeline** — two-tier architecture replaces brittle regex-first approach
  - Tier 1: spaCy dependency parsing handles ~90% of standard English (possessive chains, copular, SVO, naming patterns)
  - Tier 2: LLM tie-break for uncertain fragments from Tier 1
  - Regex patterns retained as fallback when spaCy unavailable
- Third-person pronoun resolution — "her father Chandrasiri" resolves "her" → "wife" via gender-aware antecedent lookup
- Inline pet name detection — "My Dog dexter" creates pet entity "Dexter" via compound dep analysis
- Pet entity merging — named pet ("Dexter") auto-merges with generic pet ("Dog"), transferring relations
- "X's name is Y" naming pattern — copular verb detection in possessive tree walk
- Lowercase name proximity fallback — catches names spaCy misclassifies as ADV/NOUN
- LLM response extraction — `observe()` now extracts from both user message and LLM response, with `create_new_entities=False` for responses to prevent garbage
- Priority memory deduplication — dopamine signal checks cosine similarity (>0.92) before storing
- Transformer entity encoder — produces entity embeddings in same vector space as memories
- Architecture documentation (`docs/ARCHITECTURE.md`, `docs/DEPENDENCIES.md`)
- `CLAUDE.md` project guide

### Changed
- Entity extraction pipeline restructured: dep parsing → LLM tie-break → regex fallback (was: spaCy NER → regex → LLM)
- `extract_from_memory()` accepts `response_mode` parameter for gated entity creation
- Playground updated to v0.6: 28 endpoints, OpenAI SDK, specific-origin CORS, inline React SPA
- Encoder heuristic skips "my/our" prefixed keywords between entity spans
- Graph store protects pet entities (type=animal) from junk cleanup

### Removed
- Compression layer (`limbiq/compression/`)
- Hallucination detection (`limbiq/hallucination/`)
- Activation steering (`limbiq/steering/`)
- Web search (`limbiq/search.py`)
- Onboarding (`limbiq/onboarding.py`)
- Tool registry (`limbiq/tools.py`)
- LLM client (`limbiq/llm.py`)
- Model router (`limbiq/model_router.py`)
- Associated test files for removed modules

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

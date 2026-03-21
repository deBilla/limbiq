#!/usr/bin/env python3
"""
Entry point: python -m limbiq.playground [OPTIONS]

Starts FastAPI server with:
  - Limbiq instance
  - OpenTelemetry traces + metrics
  - React dashboard at /
  - REST API at /api/v1/
"""

import sys
import logging
import argparse

logger = logging.getLogger("limbiq.playground")


def main():
    parser = argparse.ArgumentParser(
        description="Launch Limbiq Playground dashboard"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--store-path", default="./data/limbiq", help="Limbiq store directory")
    parser.add_argument("--user-id", default="default", help="User identifier")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--llm-url", default=None, help="LLM API base URL (e.g. http://localhost:11434/v1 for Ollama)")
    parser.add_argument("--llm-model", default="llama3.1", help="LLM model name")
    parser.add_argument("--llm-api-key", default=None, help="LLM API key (for OpenAI, Claude, etc.)")
    parser.add_argument("--search-url", default=None, help="Search API URL (e.g. http://localhost:8888 for SearXNG)")
    parser.add_argument("--search-provider", default="searxng", help="Search provider: searxng, brave, tavily, google_cse")
    parser.add_argument("--search-api-key", default=None, help="Search API key (for Brave, Tavily, Google CSE)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-otel", action="store_true", help="Disable OpenTelemetry")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    )

    # Setup OpenTelemetry before anything else
    if not args.no_otel:
        from limbiq.playground.instrumentation import setup_telemetry
        setup_telemetry(service_name="limbiq-playground")
        logger.info("OpenTelemetry initialized (traces + metrics)")

    # Create and run the app
    from limbiq.playground.server import create_app
    import uvicorn

    # Setup LLM if configured
    llm_client = None
    if args.llm_url:
        from limbiq.llm import LLMClient
        llm_client = LLMClient(
            base_url=args.llm_url,
            model=args.llm_model,
            api_key=args.llm_api_key,
        )
        if llm_client.is_available():
            logger.info(f"LLM connected: {args.llm_url} (model: {args.llm_model})")
        else:
            logger.warning(f"LLM not reachable at {args.llm_url} — falling back to heuristics")
            llm_client = None

    # Setup search if configured
    search_client = None
    if args.search_url:
        from limbiq.search import SearchClient
        search_client = SearchClient(
            base_url=args.search_url,
            provider=args.search_provider,
            api_key=args.search_api_key,
        )
        if search_client.is_available():
            logger.info(f"Search connected: {args.search_url} ({args.search_provider})")
        else:
            logger.warning(f"Search not reachable at {args.search_url} — search disabled")
            search_client = None

    app = create_app(
        store_path=args.store_path,
        user_id=args.user_id,
        embedding_model=args.embedding_model,
        llm_client=llm_client,
        search_client=search_client,
    )

    logger.info(f"Limbiq Playground starting on http://{args.host}:{args.port}")
    logger.info(f"  Store: {args.store_path}")
    logger.info(f"  User: {args.user_id}")
    logger.info(f"  LLM: {llm_client or 'none (heuristic mode)'}")
    logger.info(f"  Search: {search_client or 'none'}")
    logger.info(f"  Dashboard: http://localhost:{args.port}")
    logger.info(f"  API docs: http://localhost:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

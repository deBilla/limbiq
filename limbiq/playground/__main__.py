#!/usr/bin/env python3
"""
Entry point: python -m limbiq.playground [OPTIONS]

Starts FastAPI server with:
  - Limbiq instance (signals + graph)
  - Dashboard at /
  - REST API at /api/v1/
  - API docs at /docs
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
    parser.add_argument("--llm-api-key", default=None, help="LLM API key")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    )

    from limbiq.playground.server import create_app
    import uvicorn

    # Setup LLM if configured
    llm_client = None
    if args.llm_url:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=args.llm_url,
                api_key=args.llm_api_key or "not-needed",
            )
            model_name = args.llm_model

            def llm_fn(prompt, **kwargs):
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get("max_tokens", 512),
                    temperature=kwargs.get("temperature", 0.7),
                )
                return resp.choices[0].message.content

            # Quick availability check
            try:
                llm_fn("Say OK")
                llm_client = llm_fn
                logger.info(f"LLM connected: {args.llm_url} (model: {args.llm_model})")
            except Exception as e:
                logger.warning(f"LLM not reachable at {args.llm_url}: {e}")
        except ImportError:
            logger.warning("openai package not installed — LLM disabled. pip install openai")

    app = create_app(
        store_path=args.store_path,
        user_id=args.user_id,
        embedding_model=args.embedding_model,
        llm_client=llm_client,
    )

    logger.info(f"Limbiq Playground starting on http://{args.host}:{args.port}")
    logger.info(f"  Store: {args.store_path}")
    logger.info(f"  User: {args.user_id}")
    logger.info(f"  LLM: {llm_client and 'connected' or 'none (heuristic mode)'}")
    logger.info(f"  Dashboard: http://localhost:{args.port}")
    logger.info(f"  API docs: http://localhost:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

"""
Limbiq Playground — Interactive dashboard for exploring and debugging limbiq.

Usage:
    python -m limbiq.playground [OPTIONS]

    Options:
        --port INT        Server port (default: 8765)
        --store-path STR  Limbiq store directory
        --user-id STR     User identifier (default: "default")
        --host STR        Server host (default: 0.0.0.0)
"""

from limbiq.playground.server import create_app

__all__ = ["create_app"]

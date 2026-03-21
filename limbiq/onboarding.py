"""
Onboarding Manager -- agent persona and user name setup.

Stores agent_name and user_name in a dedicated SQLite table within the
existing MemoryStore's database so no new file is created.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentProfile:
    agent_name: str = "Limbiq"
    user_name: str = ""
    onboarding_complete: bool = False


class OnboardingManager:
    """Manages first-run onboarding and agent persona persistence."""

    def __init__(self, store):
        """
        Args:
            store: MemoryStore instance (provides thread-safe .db property).
        """
        self._store = store
        self._init_table()

    # ── Table setup ─────────────────────────────────────────────────

    def _init_table(self):
        self._store.db.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_profile (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT ''
            )
            """
        )
        self._store.db.commit()

        # Seed defaults if table is brand-new
        for key, val in [
            ("agent_name", "Limbiq"),
            ("user_name", ""),
            ("onboarding_complete", "0"),
            ("onboarding_step", "0"),
        ]:
            self._store.db.execute(
                "INSERT OR IGNORE INTO agent_profile (key, value) VALUES (?, ?)",
                (key, val),
            )
        self._store.db.commit()

    # ── Internal helpers ─────────────────────────────────────────────

    def _get(self, key: str) -> str:
        row = self._store.db.execute(
            "SELECT value FROM agent_profile WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else ""

    def _set(self, key: str, value: str):
        self._store.db.execute(
            "INSERT OR REPLACE INTO agent_profile (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._store.db.commit()

    # ── Public API ───────────────────────────────────────────────────

    def is_complete(self) -> bool:
        return self._get("onboarding_complete") == "1"

    def get_profile(self) -> AgentProfile:
        return AgentProfile(
            agent_name=self._get("agent_name") or "Limbiq",
            user_name=self._get("user_name"),
            onboarding_complete=self.is_complete(),
        )

    def set_user_name(self, name: str):
        self._set("user_name", name.strip())
        logger.info(f"Onboarding: user_name set to {name!r}")

    def set_agent_name(self, name: str):
        self._set("agent_name", name.strip())
        logger.info(f"Onboarding: agent_name set to {name!r}")

    def get_step(self) -> int:
        try:
            return int(self._get("onboarding_step") or "0")
        except ValueError:
            return 0

    def set_step(self, step: int):
        self._set("onboarding_step", str(step))

    def complete(self):
        self._set("onboarding_complete", "1")
        self._set("onboarding_step", "3")
        logger.info("Onboarding: marked complete")

    def reset(self):
        """Reset onboarding to start fresh."""
        self._set("user_name", "")
        self._set("agent_name", "Limbiq")
        self._set("onboarding_complete", "0")
        self._set("onboarding_step", "0")
        logger.info("Onboarding: reset")

    def get_greeting(self) -> str:
        profile = self.get_profile()
        agent = profile.agent_name or "Limbiq"
        user = profile.user_name

        if user:
            return (
                f"Hi {user}! I'm {agent}. "
                "I learn as we talk and remember things across our conversations. "
                "What's on your mind?"
            )
        return (
            f"Hi! I'm {agent}. "
            "I learn as we talk and remember things across our conversations. "
            "What's on your mind?"
        )

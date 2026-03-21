"""
Model Router -- selects the best LLM client based on context signals.

Usage:
    router = ModelRouter(
        models={"default": llm_default, "reasoning": llm_deepseek},
        default="default",
    )
    llm = router.route(query, process_result, signals)
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Patterns that suggest creative / generative tasks
_CREATIVE_RE = re.compile(
    r"\b(write|poem|story|creative|imagine|fiction|compose|draft|generate|design)\b",
    re.IGNORECASE,
)


class ModelRouter:
    """Routes queries to the most appropriate LLM based on signals."""

    def __init__(self, models: dict, default: str = "default"):
        """
        Args:
            models: Mapping of role name → LLMClient instance.
            default: Key in `models` to use when no specialised model matches.
        """
        self._models = models
        self._default_key = default
        if default not in models and models:
            # Fall back to first available
            self._default_key = next(iter(models))

    # ── Public API ────────────────────────────────────────────────────────────

    def route(self, query: str, process_result=None, signals: list = None) -> Optional[object]:
        """
        Select the best LLMClient for this request.

        Priority:
          1. norepinephrine active → "reasoning" model
          2. creative query → "creative" model
          3. Default model

        Always falls back to default if specialised model unavailable.
        """
        signals = signals or []
        sig_types = {str(getattr(s, "signal_type", s)).lower() for s in signals}

        # Norepinephrine → reasoning model
        if any("norepinephrine" in t for t in sig_types):
            model = self._get("reasoning")
            if model:
                logger.info("ModelRouter: norepinephrine active → reasoning model")
                return model

        # Creative query → creative model
        if _CREATIVE_RE.search(query or ""):
            model = self._get("creative")
            if model:
                logger.info("ModelRouter: creative query → creative model")
                return model

        # Default
        model = self._get(self._default_key)
        if model:
            logger.debug(f"ModelRouter: using default model ({self._default_key!r})")
        return model

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def get_model_name(self, query: str, process_result=None, signals: list = None) -> str:
        """Return the *key* of the model that would be selected."""
        signals = signals or []
        sig_types = {str(getattr(s, "signal_type", s)).lower() for s in signals}
        if any("norepinephrine" in t for t in sig_types) and "reasoning" in self._models:
            return "reasoning"
        if _CREATIVE_RE.search(query or "") and "creative" in self._models:
            return "creative"
        return self._default_key

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get(self, key: str):
        return self._models.get(key)

"""Pattern matching utilities for signal detection."""

from limbiq.signals.dopamine import (
    CORRECTION_PATTERNS,
    ENTHUSIASM_PATTERNS,
    PERSONAL_INFO_PATTERNS,
)
from limbiq.signals.gaba import DENIAL_PATTERNS


def match_any(text: str, patterns: list[str]) -> str | None:
    """Return the first matching pattern found in text, or None."""
    text_lower = text.lower()
    for pattern in patterns:
        if pattern in text_lower:
            return pattern
    return None


def is_correction(text: str) -> bool:
    return match_any(text, CORRECTION_PATTERNS) is not None


def is_enthusiasm(text: str) -> bool:
    return match_any(text, ENTHUSIASM_PATTERNS) is not None


def is_personal_info(text: str) -> bool:
    return match_any(text, PERSONAL_INFO_PATTERNS) is not None


def is_denial(text: str) -> bool:
    return match_any(text, DENIAL_PATTERNS) is not None

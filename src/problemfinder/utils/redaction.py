"""Utility helpers for masking personally identifiable information."""

from __future__ import annotations

import re
from typing import Tuple

PII_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+"), "[redacted_email]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[redacted_ssn]"),
    (re.compile(r"\b\d{10}\b"), "[redacted_phone]"),
)


def redact_pii(text: str) -> str:
    """Replace well-known PII patterns with neutral tokens."""

    redacted = text
    for pattern, replacement in PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted

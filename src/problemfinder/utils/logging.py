"""Logging helpers with structured JSON payloads."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def structured_log(level: int, **payload: Any) -> None:
    """Emit a JSON-formatted log line for observability pipelines."""

    logging.getLogger().log(level, json.dumps(payload, ensure_ascii=False, sort_keys=True))

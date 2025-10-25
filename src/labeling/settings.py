"""Centralised environment loading for the ProblemFinder package."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_ENV_LOADED = False


def load_environment(dotenv_path: Optional[Path] = None) -> None:
    """Load environment variables once for the entire process."""

    global _ENV_LOADED
    if _ENV_LOADED:
        return
    load_dotenv(dotenv_path)
    _ENV_LOADED = True


def require_env(name: str) -> str:
    """Retrieve ``name`` from the environment, raising if missing."""

    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

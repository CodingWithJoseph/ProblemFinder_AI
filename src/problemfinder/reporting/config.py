"""Reporting configuration models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class ReportConfig:
    """Configuration for summary report generation."""

    path: Optional[Path] = None

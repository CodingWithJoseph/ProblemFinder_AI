"""Configuration dataclasses for the ProblemFinder pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from problemfinder.classification.ensemble import EnsembleConfig
    from problemfinder.core.cache import CacheConfig
    from problemfinder.reporting.config import ReportConfig
    from problemfinder.reporting.evaluation import EvaluationConfig


@dataclass(slots=True)
class DedupeConfig:
    """Configuration parameters for the deduplication stage."""

    enabled: bool = True
    similarity_threshold: float = 0.5
    canonical_policy: str = "earliest"
    report_path: Path | None = None
    soft_similarity_threshold: float = 0.35
    cross_subreddit: bool = True


@dataclass(slots=True)
class SplitConfig:
    """Configuration for train/validation/test split ratios."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    enabled: bool = True


@dataclass(slots=True)
class ModelConfig:
    """Model selection parameters for the OpenAI classifier."""

    name: str = "gpt-4o"
    temperature: float = 0.0
    seed: int = 42


@dataclass(slots=True)
class ParallelConfig:
    """Runtime controls for threaded batch execution."""

    max_workers: int = 4
    rate_limit: int = 30
    chunk_size: int = 50


@dataclass(slots=True)
class ResumeConfig:
    """Settings for resuming a partially completed labeling run."""

    enabled: bool = False
    checkpoint_path: Path | None = None


@dataclass(slots=True)
class RunConfig:
    """Aggregate configuration for a classification run."""

    model: "ModelConfig"
    ensemble: "EnsembleConfig"
    cache: "CacheConfig"
    parallel: "ParallelConfig"
    evaluation: "EvaluationConfig"
    report: "ReportConfig"
    dedupe: "DedupeConfig"
    split: "SplitConfig"
    resume: "ResumeConfig"

    metrics: Sequence[str] = field(default_factory=tuple)

"""Classification pipeline with Version 2 labeling rules and deduplication.

This module implements a lightweight, fully-auditable classification pipeline for
Reddit posts.  The pipeline follows the Version 2 guidance captured in
``documentation/labeling_documentation.md`` and introduces an end-to-end
deduplication stage focused on cross-subreddit reposts.

Key features implemented here:

* Canonical text normalisation (HTML/Markdown stripping, URL removal, etc.).
* Duplicate clustering that combines exact-text, URL, and similarity signals.
* Deterministic canonical selection with metadata aggregation and reporting.
* Rule-based Version 2 label assignment with explicit reasoning strings.
* Optional train/validation/test splitting that respects duplicate clusters.

The module can be executed as a script (``python scripts/classify.py``) or
imported from tests.  When run as a script it accepts several CLI flags to
control deduplication behaviour and output destinations.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import html
import json
import logging
import math
import os
import queue
import random
import re
import signal
import threading
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlsplit

import pandas as pd
import yaml
from jsonschema import ValidationError, validate
from openai import OpenAI

try:  # pragma: no cover - import flexibility for script execution
    from scripts.cache import CacheConfig, ResponseCache, cached_api_call
    from scripts.ensemble import (
        PROMPT_VERSION,
        EnsembleConfig,
        EnsembleMemberResult,
        ensemble_classify,
        summarise_member_agreement,
    )
    from scripts.evaluation import EvaluationConfig, compare_against_history, evaluate_against_gold
    from scripts.report import ReportConfig, generate_summary_report
except ModuleNotFoundError:  # pragma: no cover - fallback when running as package
    from cache import CacheConfig, ResponseCache, cached_api_call
    from ensemble import (
        PROMPT_VERSION,
        EnsembleConfig,
        EnsembleMemberResult,
        ensemble_classify,
        summarise_member_agreement,
    )
    from evaluation import EvaluationConfig, compare_against_history, evaluate_against_gold
    from report import ReportConfig, generate_summary_report


logger = logging.getLogger(__name__)


_OPENAI_CLIENT = OpenAI()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


@dataclass
class DedupeConfig:
    """Configuration parameters for the deduplication stage."""

    enabled: bool = True
    similarity_threshold: float = 0.5
    canonical_policy: str = "earliest"
    report_path: Optional[Path] = None
    soft_similarity_threshold: float = 0.35
    cross_subreddit: bool = True


@dataclass
class SplitConfig:
    """Configuration for train/validation/test split ratios."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    enabled: bool = True


@dataclass
class ModelConfig:
    name: str = "gpt-4o"
    temperature: float = 0.0
    seed: int = 42


@dataclass
class ParallelConfig:
    max_workers: int = 4
    rate_limit: int = 30
    chunk_size: int = 50


@dataclass
class ResumeConfig:
    enabled: bool = False
    checkpoint_path: Optional[Path] = None


@dataclass
class RunConfig:
    model: ModelConfig
    ensemble: EnsembleConfig
    cache: CacheConfig
    parallel: ParallelConfig
    evaluation: EvaluationConfig
    report: ReportConfig
    dedupe: DedupeConfig
    split: SplitConfig
    resume: ResumeConfig


class RateLimiter:
    """Simple rate limiter expressed in requests per minute."""

    def __init__(self, rate_limit: int) -> None:
        self._rate_limit = max(rate_limit, 0)
        self._lock = threading.Lock()
        self._interval = 60.0 / self._rate_limit if self._rate_limit else 0.0
        self._next_allowed = time.time()

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.time()
            wait_time = max(0.0, self._next_allowed - now)
            self._next_allowed = max(self._next_allowed, now) + self._interval
        if wait_time:
            time.sleep(wait_time)

    @contextlib.contextmanager
    def slot(self) -> Iterator[None]:
        self.acquire()
        yield


def _load_yaml_config(path: Optional[Path]) -> Dict[str, Any]:
    if path and path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid YAML config structure at {path}")
        return payload
    return {}


def build_run_config(args: argparse.Namespace) -> RunConfig:
    config_path = Path(args.config) if getattr(args, "config", None) else None
    yaml_payload = _load_yaml_config(config_path)

    model_cfg = ModelConfig(
        name=str(getattr(args, "model", yaml_payload.get("model", "gpt-4o"))),
        temperature=float(yaml_payload.get("temperature", 0.0)),
        seed=int(yaml_payload.get("seed", 42)),
    )
    if getattr(args, "temperature", None) is not None:
        model_cfg.temperature = float(args.temperature)
    if getattr(args, "seed", None) is not None:
        model_cfg.seed = int(args.seed)

    ensemble_section = yaml_payload.get("ensemble", {}) if isinstance(yaml_payload, dict) else {}
    ensemble_cfg = EnsembleConfig(
        enabled=bool(ensemble_section.get("enabled", False)),
        members=ensemble_section.get("members", ["direct", "reasoning", "rules"]),
        disagreement_threshold=float(ensemble_section.get("disagreement_threshold", 0.3)),
    )
    if getattr(args, "ensemble", None) is not None:
        ensemble_cfg.enabled = args.ensemble == "on"
    if getattr(args, "ensemble_members", None):
        ensemble_cfg.members = [member.strip() for member in args.ensemble_members.split(",") if member.strip()]
    if getattr(args, "ensemble_disagreement_threshold", None) is not None:
        ensemble_cfg.disagreement_threshold = float(args.ensemble_disagreement_threshold)

    cache_section = yaml_payload.get("cache", {}) if isinstance(yaml_payload, dict) else {}
    cache_cfg = CacheConfig(
        enabled=bool(cache_section.get("enabled", False)),
        path=Path(cache_section.get("path", "data/.cache/responses.json")),
        ttl_hours=int(cache_section.get("ttl_hours", 24)),
    )
    if getattr(args, "cache", None) is not None:
        cache_cfg.enabled = args.cache == "on"
    if getattr(args, "cache_ttl", None) is not None:
        cache_cfg.ttl_hours = int(args.cache_ttl)
    if getattr(args, "cache_path", None):
        cache_cfg.path = Path(args.cache_path)

    parallel_section = yaml_payload.get("parallel", {}) if isinstance(yaml_payload, dict) else {}
    parallel_cfg = ParallelConfig(
        max_workers=int(parallel_section.get("max_workers", 4)),
        rate_limit=int(parallel_section.get("rate_limit", 30)),
        chunk_size=int(parallel_section.get("chunk_size", 50)),
    )
    if getattr(args, "max_workers", None) is not None:
        parallel_cfg.max_workers = max(1, int(args.max_workers))
    if getattr(args, "rate_limit", None) is not None:
        parallel_cfg.rate_limit = max(0, int(args.rate_limit))
    if getattr(args, "chunk_size", None) is not None:
        parallel_cfg.chunk_size = max(1, int(args.chunk_size))

    evaluation_section = yaml_payload.get("evaluation", {}) if isinstance(yaml_payload, dict) else {}
    evaluation_cfg = EvaluationConfig(
        enabled=bool(evaluation_section.get("enabled", False)),
        gold_set_path=Path(evaluation_section.get("gold_set_path")) if evaluation_section.get("gold_set_path") else None,
        metrics=evaluation_section.get("metrics", ["accuracy", "macro_f1", "confusion_matrix"]),
    )
    if getattr(args, "evaluation", None) is not None:
        evaluation_cfg.enabled = args.evaluation == "on"
    if getattr(args, "evaluation_gold_set", None):
        evaluation_cfg.gold_set_path = Path(args.evaluation_gold_set)

    report_section = yaml_payload.get("report", {}) if isinstance(yaml_payload, dict) else {}
    report_cfg = ReportConfig(path=Path(report_section.get("path")) if report_section.get("path") else None)
    if getattr(args, "report_path", None):
        report_cfg.path = Path(args.report_path)

    cross_subreddit_setting = yaml_payload.get("cross_subreddit", True)
    if getattr(args, "cross_subreddit_dedupe", False):
        cross_subreddit_setting = True

    dedupe_cfg = DedupeConfig(
        enabled=(args.dedupe == "on"),
        similarity_threshold=float(args.similarity_threshold),
        canonical_policy=args.canonical_policy,
        report_path=args.dedupe_report,
        soft_similarity_threshold=float(
            getattr(args, "soft_similarity_threshold", yaml_payload.get("soft_similarity_threshold", 0.35))
        ),
        cross_subreddit=bool(cross_subreddit_setting),
    )
    if isinstance(dedupe_cfg.report_path, str):
        dedupe_cfg.report_path = Path(dedupe_cfg.report_path)

    split_cfg = SplitConfig(
        enabled=not args.no_split,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )

    resume_cfg = ResumeConfig(
        enabled=bool(getattr(args, "resume", False)),
        checkpoint_path=Path(args.resume_from) if getattr(args, "resume_from", None) else None,
    )

    return RunConfig(
        model=model_cfg,
        ensemble=ensemble_cfg,
        cache=cache_cfg,
        parallel=parallel_cfg,
        evaluation=evaluation_cfg,
        report=report_cfg,
        dedupe=dedupe_cfg,
        split=split_cfg,
        resume=resume_cfg,
    )


def _create_member_callables(
    *,
    engine: "Version2RuleEngine",
    cache: ResponseCache,
    cache_stats: MutableMapping[str, int],
    rate_limiter: RateLimiter,
    model_cfg: ModelConfig,
) -> Dict[str, Any]:
    def _coerce_label(value: Any) -> str:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped in {"0", "1"}:
                return stripped
            if stripped.lower() in {"true", "false"}:
                return "1" if stripped.lower() == "true" else "0"
            if stripped.isdigit():
                return "1" if int(stripped) >= 1 else "0"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (int, float)):
            return "1" if float(value) >= 0.5 else "0"
        return ""

    def _llm_member(member_name: str) -> Any:
        if member_name == "direct":
            system_prompt = (
                "You are a classifier for Reddit posts. "
                "Decide whether the post describes a problem, if that problem is solvable with software alone, and whether external help is needed. "
                "Respond ONLY with minified JSON in this form: {\"is_problem\": \"0 or 1\", \"is_software_solvable\": \"0 or 1\", "
                "\"is_external\": \"0 or 1\", \"confidence\": number between 0 and 1, \"rationale\": short factual string}. "
                "Keep the rationale concise and free of Markdown."
            )
            default_confidence = 0.8
        else:
            system_prompt = (
                "You are a careful classifier for Reddit posts. Think step by step about whether the post describes a problem, "
                "if software alone can solve it, and whether outside coordination is required. After reasoning internally, respond ONLY "
                "with JSON matching {\"is_problem\": \"0 or 1\", \"is_software_solvable\": \"0 or 1\", \"is_external\": \"0 or 1\", "
                "\"confidence\": number between 0 and 1, \"rationale\": short factual string}."
            )
            default_confidence = 0.9

        def _call(text: str) -> EnsembleMemberResult:
            prompt_version = f"{PROMPT_VERSION}:{member_name}"

            def _fetch() -> Dict[str, Any]:
                with rate_limiter.slot():
                    response = _OPENAI_CLIENT.chat.completions.create(
                        model=model_cfg.name,
                        temperature=model_cfg.temperature,
                        seed=model_cfg.seed,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Post:\n{text.strip()}"},
                        ],
                    )
                content = ""
                if response.choices:
                    content = response.choices[0].message.content or ""
                cleaned = content.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("`")
                    parts = cleaned.split("\n", 1)
                    if parts and parts[0].lower().startswith("json"):
                        cleaned = parts[1] if len(parts) > 1 else ""
                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                    raise ValueError(f"Failed to parse JSON from LLM response: {cleaned}") from exc

                rationale = str(parsed.get("rationale", "")).strip()
                confidence_value = parsed.get("confidence", default_confidence)
                try:
                    confidence_float = float(confidence_value)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    confidence_float = default_confidence
                confidence_float = max(0.0, min(1.0, confidence_float))

                payload = {
                    "intent": str(parsed.get("intent", "")) if parsed.get("intent") is not None else "",
                    "is_problem": _coerce_label(parsed.get("is_problem")),
                    "is_software_solvable": _coerce_label(parsed.get("is_software_solvable")),
                    "is_external": _coerce_label(parsed.get("is_external")),
                    "problem_reason": rationale,
                    "software_reason": rationale,
                    "external_reason": rationale,
                    "detected_patterns": "",
                    "confidence": confidence_float,
                }
                return {"payload": payload, "confidence": confidence_float, "rationale": rationale}

            response = cached_api_call(
                model=model_cfg.name,
                prompt_version=prompt_version,
                text=text,
                cache=cache,
                fetch_fn=_fetch,
                cache_stats=cache_stats,
            )
            payload = response.get("payload", {})
            confidence = response.get("confidence", default_confidence)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                confidence = default_confidence
            rationale = response.get("rationale", "")
            return EnsembleMemberResult(member=member_name, payload=payload, confidence=confidence, rationale=rationale)

        return _call

    members: Dict[str, Any] = {"rules": lambda text: None}
    members["direct"] = _llm_member("direct")
    members["reasoning"] = _llm_member("reasoning")
    return members


PII_PATTERNS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+"), "[redacted_email]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[redacted_ssn]"),
    (re.compile(r"\b\d{10}\b"), "[redacted_phone]"),
]


def redact_pii(text: str) -> str:
    redacted = text
    for pattern, replacement in PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def structured_log(level: int, **payload: Any) -> None:
    logger.log(level, json.dumps(payload, ensure_ascii=False, sort_keys=True))


def parallel_process_batch(
    tasks: Sequence[Tuple[int, pd.Series]],
    worker_fn,
    *,
    parallel_cfg: ParallelConfig,
) -> List[Any]:
    """Run ``worker_fn`` over ``tasks`` in parallel while preserving order."""

    results: List[Any] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=parallel_cfg.max_workers) as executor:
        future_items: List[Tuple[int, Any]] = []
        for position, (index, row) in enumerate(tasks):
            future_items.append((position, executor.submit(worker_fn, index, row)))
        for position, future in future_items:
            try:
                results[position] = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                results[position] = exc
    return results


def _normalise_config(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _normalise_config(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_normalise_config(item) for item in payload]
    if isinstance(payload, Path):
        return str(payload)
    return payload


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_markdown(text: str) -> str:
    """Remove common Markdown constructs from text."""

    # Remove code blocks and inline code
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", " ", text)

    # Remove images/links while keeping the anchor text if present
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", lambda m: m.group(0).split("]")[0][1:], text)

    # Remove headings/blockquote markers/lists
    text = re.sub(r"^>+\s?", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^#{1,6}\s*", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^(?:\s*[-*+]\s+|\d+\.\s+)", " ", text, flags=re.MULTILINE)
    return text


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def is_valid_ipv6_url(url):
    """Check if a URL contains a valid IPv6 address format."""
    # IPv6 URLs should have the format: scheme://[IPv6]:port/path
    ipv6_pattern = r'://\[([0-9a-fA-F:]+)\]'
    return re.search(ipv6_pattern, url) is not None


def safe_urlsplit(url):
    """Safely split a URL with validation for IPv6 format."""
    try:
        # Basic validation - check if it looks like a valid URL
        if not url or not isinstance(url, str):
            return None

        # Handle potential IPv6 URL issues
        if '[' in url and ']' in url:
            if not is_valid_ipv6_url(url):
                logger.warning(f"Skipping invalid IPv6 URL: {url[:50]}...")
                return None

        return urlsplit(url)
    except ValueError as e:
        logger.warning(f"ValueError parsing URL: {e} for URL: {url[:50]}...")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error parsing URL: {e} for URL: {url[:50]}...")
        return None


def _remove_urls(text: str) -> Tuple[str, List[str]]:
    """Remove URLs from ``text`` while collecting them with robust error handling."""

    urls: List[str] = []

    def _collect(match: re.Match[str]) -> str:
        url = match.group(0)
        # Validate URL before adding to collection
        try:
            parsed = safe_urlsplit(url)
            if parsed is not None:
                urls.append(url)
            else:
                # Log the problematic URL for debugging
                logger.debug(f"Skipped malformed URL: {url[:50]}...")
        except Exception as e:
            logger.debug(f"Error processing URL {url[:50]}...: {e}")

        return " "

    cleaned = re.sub(r"https?://\S+", _collect, text)
    return cleaned, urls


def _remove_crosspost_boilerplate(text: str) -> str:
    return re.sub(r"cross\s*post(ed)?\s+from\s+r/\w+", " ", text, flags=re.IGNORECASE)


def _remove_mentions(text: str) -> str:
    return re.sub(r"/?u/[A-Za-z0-9_-]+", " ", text)


def _strip_emoji(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch) != "So")


def _tokenise(text: str) -> List[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]


def _char_ngrams(text: str, n: int = 5) -> List[str]:
    if len(text) < n:
        return [text]
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def _combined_similarity(text_a: str, text_b: str) -> float:
    tokens_a = _tokenise(text_a)
    tokens_b = _tokenise(text_b)
    token_score = _jaccard(tokens_a, tokens_b)

    chars_a = _char_ngrams(text_a)
    chars_b = _char_ngrams(text_b)
    char_score = _jaccard(chars_a, chars_b)

    # Weighted average gives stronger emphasis to token overlap while still
    # rewarding similar phrasing.
    return 0.7 * token_score + 0.3 * char_score


@dataclass
class NormalisedPost:
    """Container for normalised text and metadata used during deduping."""

    post_id: str
    index: int
    combined_text: str
    normalised_text: str
    urls: List[str]
    subreddit: str
    created_utc: float
    body_length: int


def normalise_post(row: pd.Series, *, post_id: str) -> NormalisedPost:
    """Create a :class:`NormalisedPost` by applying canonical cleaning steps."""

    title = (row.get("title", "") or "")
    body = (row.get("body", "") or "")
    combined = f"{title}\n\n{body}".strip()

    lowered = combined.lower()
    lowered = _remove_crosspost_boilerplate(lowered)
    lowered = _remove_mentions(lowered)

    no_html = _strip_html(lowered)
    cleaned_urls_text, urls = _remove_urls(no_html)
    no_markdown = _strip_markdown(cleaned_urls_text)
    no_emoji = _strip_emoji(no_markdown)
    no_symbols = re.sub(r"[^a-z0-9\s]", " ", no_emoji)

    normalised = _normalise_whitespace(no_symbols)

    subreddit = str(row.get("subreddit", "")).strip()
    created_utc_raw = row.get("created_utc", math.inf)
    created_utc = float(created_utc_raw) if pd.notna(created_utc_raw) else math.inf

    cleaned_urls = [re.sub(r"[)\]\.,;:!?]+$", "", url.lower()) for url in urls]

    return NormalisedPost(
        post_id=post_id,
        index=int(row.name),
        combined_text=combined,
        normalised_text=normalised,
        urls=[_normalise_whitespace(url) for url in cleaned_urls],
        subreddit=subreddit,
        created_utc=created_utc,
        body_length=len(body or ""),
    )


class UnionFind:
    """Union-Find/Disjoint-set implementation for clustering duplicates."""

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def find(self, item: int) -> int:
        if self.parent.get(item, item) != item:
            self.parent[item] = self.find(self.parent[item])
        else:
            self.parent.setdefault(item, item)
            self.rank.setdefault(item, 0)
        return self.parent[item]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        rank_a = self.rank.get(root_a, 0)
        rank_b = self.rank.get(root_b, 0)
        if rank_a < rank_b:
            self.parent[root_a] = root_b
        elif rank_a > rank_b:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] = rank_a + 1


def _iter_candidate_pairs(posts: Sequence[NormalisedPost]) -> Iterator[Tuple[int, int]]:
    """Yield pairs of indices that are likely duplicates using blocking heuristics."""

    token_index: Dict[str, List[int]] = defaultdict(list)
    url_index: Dict[str, List[int]] = defaultdict(list)
    domain_index: Dict[str, List[int]] = defaultdict(list)
    text_index: Dict[str, List[int]] = defaultdict(list)

    for idx, post in enumerate(posts):
        text_index[post.normalised_text].append(idx)
        for url in post.urls:
            url_index[url].append(idx)
            domain = urlparse(url).netloc
            if domain:
                domain_index[domain].append(idx)

        tokens = _tokenise(post.normalised_text)
        unique_tokens = {tok for tok in tokens if len(tok) > 3}
        for token in unique_tokens:
            token_index[token].append(idx)

    # Exact text duplicates
    for indices in text_index.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    yield indices[i], indices[j]

    # Same URL duplicates
    for indices in url_index.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    yield indices[i], indices[j]

    # Same domain duplicates (link aggregation)
    for indices in domain_index.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    yield indices[i], indices[j]

    # Candidate pairs based on shared tokens
    seen_pairs: set[Tuple[int, int]] = set()
    for indices in token_index.values():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                pair = (min(a, b), max(a, b))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    yield pair


def cluster_duplicates(posts: Sequence[NormalisedPost], config: DedupeConfig) -> Dict[int, List[int]]:
    """Cluster posts using similarity and URL signals."""

    if not config.enabled or len(posts) <= 1:
        return {idx: [idx] for idx in range(len(posts))}

    uf = UnionFind()

    for a_idx, b_idx in _iter_candidate_pairs(posts):
        post_a = posts[a_idx]
        post_b = posts[b_idx]

        similarity = _combined_similarity(post_a.normalised_text, post_b.normalised_text)
        shared_url = bool(set(post_a.urls) & set(post_b.urls))
        soft_match = similarity >= config.soft_similarity_threshold and (post_a.subreddit != post_b.subreddit)
        if (
            post_a.normalised_text == post_b.normalised_text
            or shared_url
            or similarity >= config.similarity_threshold
            or (config.cross_subreddit and soft_match)
        ):
            uf.union(a_idx, b_idx)

    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(posts)):
        root = uf.find(idx)
        clusters[root].append(idx)

    return dict(clusters)


def _canonical_index(indices: List[int], posts: Sequence[NormalisedPost], policy: str) -> int:
    """Select the canonical representative index for a cluster."""

    candidates = [posts[i] for i in indices]

    if policy == "longest":
        candidates.sort(key=lambda p: (-p.body_length, p.created_utc, p.post_id))
    else:  # default 'earliest'
        candidates.sort(key=lambda p: (p.created_utc, -p.body_length, p.post_id))

    return posts.index(candidates[0])


def deduplicate_dataframe(df: pd.DataFrame, config: DedupeConfig) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, List[str]]]:
    """Deduplicate dataframe rows and return canonical dataframe and mapping.

    Returns
    -------
    canonical_df : pd.DataFrame
        Dataframe containing only canonical rows.  Additional columns include
        ``canonical_post_id`` (self-referential) and aggregated subreddit list.
    id_mapping : Dict[str, str]
        Mapping from original post_id -> canonical post_id.
    cluster_members : Dict[str, List[str]]
        Mapping from canonical post_id -> list of member post_ids.
    """

    df = df.copy().reset_index(drop=True)

    if "post_id" in df.columns:
        post_ids = df["post_id"].astype(str).tolist()
    elif "url" in df.columns:
        post_ids = df["url"].fillna("").astype(str).tolist()
    else:
        post_ids = [f"row_{i}" for i in range(len(df))]

    posts = [normalise_post(df.iloc[i], post_id=post_ids[i]) for i in range(len(df))]
    clusters = cluster_duplicates(posts, config)

    canonical_rows: List[pd.Series] = []
    id_mapping: Dict[str, str] = {}
    cluster_members: Dict[str, List[str]] = {}

    for cluster in clusters.values():
        if len(cluster) == 1:
            idx = cluster[0]
            post = posts[idx]
            row = df.iloc[[idx]].copy()
            row.loc[:, "canonical_post_id"] = post.post_id
            row.loc[:, "duplicate_post_ids"] = json.dumps([post.post_id])
            row.loc[:, "duplicate_subreddits"] = json.dumps(sorted({post.subreddit} if post.subreddit else []))
            row.loc[:, "normalized_text"] = post.normalised_text
            canonical_rows.append(row)
            id_mapping[post.post_id] = post.post_id
            cluster_members[post.post_id] = [post.post_id]
            continue

        canonical_idx = _canonical_index(cluster, posts, config.canonical_policy)
        canonical_post = posts[canonical_idx]
        canonical_row = df.iloc[[canonical_idx]].copy()

        member_ids = [posts[i].post_id for i in cluster]
        member_subreddits = sorted({posts[i].subreddit for i in cluster if posts[i].subreddit})

        canonical_row.loc[:, "canonical_post_id"] = canonical_post.post_id
        canonical_row.loc[:, "duplicate_post_ids"] = json.dumps(member_ids)
        canonical_row.loc[:, "duplicate_subreddits"] = json.dumps(member_subreddits)
        canonical_row.loc[:, "normalized_text"] = canonical_post.normalised_text

        canonical_rows.append(canonical_row)

        for member_id in member_ids:
            id_mapping[member_id] = canonical_post.post_id

        cluster_members[canonical_post.post_id] = member_ids

    canonical_df = pd.concat(canonical_rows, ignore_index=True)

    return canonical_df, id_mapping, cluster_members


def write_dedupe_report(report_path: Path, clusters: Dict[str, List[str]], df: pd.DataFrame) -> None:
    """Write a dedupe report to ``report_path`` in CSV format."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["canonical_post_id", "member_post_ids", "subreddits"])
        for canonical_id, member_ids in sorted(clusters.items()):
            mask = df["canonical_post_id"] == canonical_id
            if mask.any():
                subreddits_json = df.loc[mask, "duplicate_subreddits"].iat[0]
                subreddit_list = sorted(json.loads(subreddits_json)) if subreddits_json else []
            else:
                subreddit_list = []
            writer.writerow([canonical_id, "|".join(member_ids), "|".join(subreddit_list)])


# ---------------------------------------------------------------------------
# Version 2 rule-based classifier
# ---------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    intent: str
    is_problem: str
    is_software_solvable: str
    is_external: str
    problem_reason: str
    software_reason: str
    external_reason: str
    detected_patterns: str
    confidence: float


CLASSIFICATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "is_problem": {"type": "string"},
        "is_software_solvable": {"type": "string"},
        "is_external": {"type": "string"},
        "problem_reason": {"type": "string"},
        "software_reason": {"type": "string"},
        "external_reason": {"type": "string"},
        "detected_patterns": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "intent",
        "is_problem",
        "is_software_solvable",
        "is_external",
        "problem_reason",
        "software_reason",
        "external_reason",
        "detected_patterns",
        "confidence",
    ],
}


INTENT_LABELS = {
    "seeking_help": "1",
    "sharing_advice": "2",
    "showcasing": "3",
    "discussing": "4",
}


class Version2RuleEngine:
    """Rule-based classifier that codifies Version 2 guidance."""

    PROBLEM_CUES = {
        "problem",
        "issue",
        "bug",
        "error",
        "help",
        "can't",
        "cannot",
        "stuck",
        "frustrated",
        "need",
        "looking for",
        "struggling",
        "anyone else",
        "recommend",
        "how do i",
        "how to",
        "should i",
        "broken",
        "glitch",
        "fail",
        "fails",
        "failed",
        "failing",
        "failure",
    }

    NEGATED_PROBLEM_PATTERNS = {
        "no problem": {"problem"},
        "no problems": {"problem"},
        "not a problem": {"problem"},
        "not an issue": {"issue"},
        "no issue": {"issue"},
        "no issues": {"issue"},
        "without issue": {"issue"},
        "without issues": {"issue"},
        "never an issue": {"issue"},
        "not broken": {"broken"},
        "no longer broken": {"broken"},
        "no bug": {"bug"},
        "no bugs": {"bug"},
        "without fail": {"fail"},
        "never failed": {"fail", "failed"},
        "never fails": {"fail", "fails"},
        "never failing": {"fail", "failing"},
        "never failure": {"fail", "failure"},
    }

    RESOLVED_PATTERNS = {
        "already fixed",
        "finally solved",
        "fixed it by",
        "fixed this by",
        "i fixed it by",
        "i fixed this by",
        "issue was resolved when",
        "it was resolved when",
        "ended up fixing it by",
        "randomly got",
        "solved it by",
        "solved this by",
        "turns out it was",
        "was able to fix it by",
    }

    ADVICE_PATTERNS = {
        "here's my advice",
        "here's how i",
        "here is how i",
        "i learned",
        "lessons learned",
        "guide",
        "how i",
        "playbook",
        "tips",
        "tips and tricks",
        "tutorial",
        "walkthrough",
        "workflow",
    }

    ADVICE_CONTEXT_CUES = {
        "for anyone else",
        "in case it helps",
        "hope this helps",
        "sharing my experience",
        "wanted to share",
        "here's what worked",
        "here's what i did",
        "here is what i did",
        "here is what worked",
        "my solution",
        "the solution was",
        "solution:",
        "i fixed",
        "i solved",
        "fixed it by",
        "solved it by",
        "fixed this by",
        "solved this by",
        "fix was",
        "worked for me",
    }

    SOFTWARE_CUES = {
        "app",
        "software",
        "application",
        "program",
        "tool",
        "code",
        "script",
        "automation",
        "api",
        "driver",
        "update",
        "crash",
        "bug",
        "error",
        "config",
        "settings",
        "install",
    }

    NON_SOFTWARE_CUES = {
        "hardware",
        "firmware",
        "motherboard",
        "cpu",
        "gpu",
        "graphics card",
        "power supply",
        "printer",
        "camera",
        "device",
        "replacement",
        "warranty",
        "career",
        "job",
        "education",
        "course",
        "class",
        "buy",
        "purchase",
        "recommend a",
        "looking for a",
        "alternative",
    }

    EXTERNAL_CUES = {
        "warranty",
        "manufacturer",
        "support",
        "customer service",
        "rma",
        "repair",
        "buy",
        "buy a",
        "buy new",
        "buying",
        "looking to buy",
        "purchase",
        "purchase a",
        "replacement",
        "recommend a",
        "recommend",
        "alternative",
        "which should i",
        "vendor",
        "supplier",
        "retailer",
        "store",
        "dealer",
        "quote",
        "estimate",
        "career",
        "job",
        "school",
        "college",
        "printer",
        "hardware",
        "device",
        "samsung",
        "apple",
        "dell",
        "hp",
        "lenovo",
        "asus",
        "acer",
        "microsoft",
        "sony",
        "lg",
        "toshiba",
        "msi",
        "vevor",
        "goxlr",
    }

    PROTOTYPE_PATTERNS = {
        "prototype",
        "sensor",
        "manufacturing",
        "deployment",
    }

    QUESTION_CUES = (
        "how do i",
        "how to",
        "what should i",
        "can anyone",
        "does anyone",
        "anyone know",
        "is there a way",
        "who knows",
        "could someone",
    )

    UNCERTAINTY_WORDS = {
        "maybe",
        "not sure",
        "probably",
        "i think",
        "i guess",
        "perhaps",
    }

    def classify(self, text: str) -> ClassificationResult:
        """Classify ``text`` and return labels with reasoning."""

        lowered = text.lower()

        intent = self._infer_intent(lowered)
        intent_label = INTENT_LABELS[intent]

        is_problem, problem_reason = self._classify_problem(lowered, intent)
        is_software, software_reason = self._classify_software(lowered, is_problem)
        is_external, external_reason = self._classify_external(lowered, is_problem)

        detected_patterns = self._detect_edge_cases(lowered)
        confidence = self._confidence(lowered, detected_patterns)

        return ClassificationResult(
            intent=intent_label,
            is_problem=is_problem,
            is_software_solvable=is_software,
            is_external=is_external,
            problem_reason=problem_reason,
            software_reason=software_reason,
            external_reason=external_reason,
            detected_patterns=", ".join(detected_patterns),
            confidence=confidence,
        )

    # --- private helpers -------------------------------------------------

    def _infer_intent(self, text: str) -> str:
        stripped = text.strip()

        has_direct_question = "?" in text or any(
            stripped.startswith(cue) or cue in text for cue in self.QUESTION_CUES
        )
        if has_direct_question:
            return "seeking_help"

        if any(pattern in text for pattern in self.ADVICE_PATTERNS):
            return "sharing_advice"

        if any(pattern in text for pattern in self.RESOLVED_PATTERNS):
            return "sharing_advice"

        if any(pattern in text for pattern in self.ADVICE_CONTEXT_CUES):
            return "sharing_advice"

        if any(phrase in text for phrase in ["i built", "i made", "launch", "showcase"]):
            return "showcasing"

        if any(pattern in text for pattern in self.PROBLEM_CUES):
            return "seeking_help"

        return "discussing"

    def _classify_problem(self, text: str, intent: str) -> Tuple[str, str]:
        if any(pattern in text for pattern in self.RESOLVED_PATTERNS):
            return "0", "Story explains how the issue was already resolved."

        cues_found = [pattern for pattern in self.PROBLEM_CUES if pattern in text]
        filtered_cues = self._filter_negated_cues(text, cues_found)
        if intent == "showcasing" and not cues_found:
            return "0", "Showcase post without any unresolved frustration."

        if filtered_cues:
            reason = f"Found unresolved problem cues: {', '.join(sorted(set(filtered_cues))[:3])}."
            return "1", reason

        if cues_found and not filtered_cues:
            return "0", "Problem words only show up in negated phrases."

        if intent == "sharing_advice":
            return "1", "Post shares advice based on a real past problem."

        return "0", "No unresolved pain point or help request detected."

    def _filter_negated_cues(self, text: str, cues: List[str]) -> List[str]:
        """Remove cues that only appear as part of negated phrases."""

        negated_cues = set()
        for phrase, blocked in self.NEGATED_PROBLEM_PATTERNS.items():
            if phrase in text:
                negated_cues.update(blocked)

        return [cue for cue in cues if cue not in negated_cues]

    def _classify_software(self, text: str, is_problem: str) -> Tuple[str, str]:
        if is_problem != "1":
            return "0", "No active problem, so this label stays at 0."

        software_hits = [cue for cue in self.SOFTWARE_CUES if cue in text]
        non_software_hits = [cue for cue in self.NON_SOFTWARE_CUES if cue in text]

        if (
            "looking for" in text
            and any(
                keyword in text
                for keyword in {"software", "app", "application", "tool", "program"}
            )
            and not any(word in text for word in {"build", "create", "develop"})
        ):
            return "0", "Looking for an existing product rather than fixing software."

        if software_hits and not non_software_hits:
            return "1", f"Clear software troubleshooting cues: {', '.join(sorted(set(software_hits))[:3])}."

        if non_software_hits and not software_hits:
            return "0", f"Signals point to hardware or market research ({', '.join(sorted(set(non_software_hits))[:3])})."

        if "how do i" in text or "how to" in text:
            return "0", "Question is about learning, not repairing software."

        if "driver" in software_hits or "update" in software_hits:
            return "1", "Mentions drivers or updates linking problem to software."

        if software_hits and non_software_hits:
            return "0", "Mix of hardware and product cues, so not purely software solvable."

        return "0", "No strong software-related clues detected."

    def _classify_external(self, text: str, is_problem: str) -> Tuple[str, str]:
        if is_problem != "1":
            return "0", "No active problem, so this label stays at 0."

        external_hits = [cue for cue in self.EXTERNAL_CUES if cue in text]
        prototype_hits = [cue for cue in self.PROTOTYPE_PATTERNS if cue in text]

        if "which" in text and "should i" in text:
            return "1", "Choosing between outside options, so external help is needed."

        if "job" in text and "advice" in text:
            return "0", "Career advice discussion can be handled individually."

        if any(word in text for word in {"career", "school", "college"}) and "advice" in text:
            return "0", "Education or career advice does not need outside partners."

        if external_hits:
            return "1", f"Needs outside support: {', '.join(sorted(set(external_hits))[:3])}."

        if prototype_hits:
            return "1", "Prototype work calls for external hardware or fabrication help."

        if "how do i" in text or "how to" in text:
            return "0", "User can act alone with the right guidance."

        return "0", "No signs that external coordination is required."

    def _detect_edge_cases(self, text: str) -> List[str]:
        patterns = []
        if "how do i" in text or "how to" in text:
            patterns.append("learning_question")
        if "looking for" in text:
            patterns.append("seeking_existing_solution")
        if any(pattern in text for pattern in self.ADVICE_PATTERNS):
            patterns.append("advice_sharing")
        if any(pattern in text for pattern in self.RESOLVED_PATTERNS):
            patterns.append("resolved_problem")
        return patterns

    def _confidence(self, text: str, patterns: List[str]) -> float:
        if any(word in text for word in self.UNCERTAINTY_WORDS):
            return 0.3
        if patterns:
            return 0.7
        return 1.0


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------


def assign_splits(df: pd.DataFrame, cluster_members: Dict[str, List[str]], config: SplitConfig, seed: int = 42) -> pd.DataFrame:
    """Assign train/val/test splits ensuring clusters remain together."""

    if not config.enabled:
        df["split"] = "unsplit"
        return df

    ratios = [config.train_ratio, config.val_ratio, config.test_ratio]
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError("Split ratios must sum to 1.0")

    canonical_ids = list(cluster_members.keys())
    random.Random(seed).shuffle(canonical_ids)

    total = len(canonical_ids)
    train_cut = int(total * config.train_ratio)
    val_cut = train_cut + int(total * config.val_ratio)

    assignments: Dict[str, str] = {}
    for i, canonical_id in enumerate(canonical_ids):
        if i < train_cut:
            split = "train"
        elif i < val_cut:
            split = "val"
        else:
            split = "test"
        assignments[canonical_id] = split

    df["split"] = df["canonical_post_id"].map(assignments)
    return df


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def classify_dataframe(
    df: pd.DataFrame,
    *,
    run_config: Optional[RunConfig] = None,
    cache: Optional[ResponseCache] = None,
    rate_limiter: Optional[RateLimiter] = None,
    historical_metrics: Optional[Sequence[Mapping[str, Any]]] = None,
    dedupe_config: Optional[DedupeConfig] = None,
    split_config: Optional[SplitConfig] = None,
) -> (
    Tuple[pd.DataFrame, Dict[str, str], Dict[str, List[str]], Dict[str, Any]]
    | Tuple[pd.DataFrame, Dict[str, str], Dict[str, List[str]]]
):
    """Run deduplication and classification on ``df``.

    The function supports both the modern pipeline interface (supplying a
    ``RunConfig`` along with cache and rate limiter instances) and the legacy
    test-oriented interface that only provides dedupe/split configuration.  When
    ``run_config`` is omitted, a simplified, fully in-process path is used and
    the summary payload is not returned.
    """

    if run_config is None:
        dedupe_cfg = dedupe_config or DedupeConfig()
        split_cfg = split_config or SplitConfig()
        canonical_df, id_mapping, clusters = deduplicate_dataframe(df.copy(), dedupe_cfg)

        for field in CLASSIFICATION_SCHEMA["properties"].keys():
            if field not in canonical_df.columns:
                canonical_df[field] = ""

        engine = Version2RuleEngine()
        for idx, row in canonical_df.iterrows():
            text = f"{row.get('title', '')}\n\n{row.get('body', '')}".strip()
            result = engine.classify(text)
            for field in CLASSIFICATION_SCHEMA["properties"].keys():
                canonical_df.at[idx, field] = getattr(result, field)

        canonical_df = assign_splits(canonical_df, clusters, split_cfg)
        return canonical_df, id_mapping, clusters

    random.seed(run_config.model.seed)
    structured_log(
        logging.INFO,
        event="dedupe_start",
        enabled=run_config.dedupe.enabled,
        similarity_threshold=run_config.dedupe.similarity_threshold,
        soft_similarity_threshold=run_config.dedupe.soft_similarity_threshold,
    )
    dedupe_start = time.time()
    canonical_df, id_mapping, clusters = deduplicate_dataframe(df, run_config.dedupe)
    dedupe_elapsed = time.time() - dedupe_start
    duplicates_removed = len(df) - len(canonical_df)
    structured_log(
        logging.INFO,
        event="dedupe_complete",
        total=len(df),
        canonical=len(canonical_df),
        clusters=len(clusters),
        duplicates_removed=duplicates_removed,
        elapsed_seconds=dedupe_elapsed,
    )

    if run_config.dedupe.report_path:
        write_dedupe_report(run_config.dedupe.report_path, clusters, canonical_df)
        logger.info("Dedupe report written to %s", run_config.dedupe.report_path)

    for field in CLASSIFICATION_SCHEMA["properties"].keys():
        if field not in canonical_df.columns:
            canonical_df[field] = ""

    processed_ids: set[str] = set()
    if run_config.resume.enabled and run_config.resume.checkpoint_path and run_config.resume.checkpoint_path.exists():
        resume_df = pd.read_csv(run_config.resume.checkpoint_path).fillna("")
        if "canonical_post_id" in resume_df.columns:
            processed_ids = set(resume_df["canonical_post_id"].astype(str))
            for field in CLASSIFICATION_SCHEMA["properties"].keys():
                if field in resume_df.columns:
                    mapping = dict(zip(resume_df["canonical_post_id"].astype(str), resume_df[field].astype(str)))
                    canonical_df[field] = canonical_df["canonical_post_id"].astype(str).map(mapping).fillna(canonical_df[field])
            structured_log(
                logging.INFO,
                event="resume_loaded",
                resume_path=str(run_config.resume.checkpoint_path),
                recovered=len(processed_ids),
            )

    classifier = Version2RuleEngine()
    cache_stats: Dict[str, int] = {"hits": 0, "misses": 0}
    base_members = _create_member_callables(
        engine=classifier,
        cache=cache,
        cache_stats=cache_stats,
        rate_limiter=rate_limiter,
        model_cfg=run_config.model,
    )

    def _active_member_callables() -> Dict[str, Any]:
        if not run_config.ensemble.enabled:
            return {"rules": base_members["rules"]}
        enabled_set = set(run_config.ensemble.members) | {"rules"}
        return {name: base_members[name] for name in enabled_set if name in base_members}

    ensemble_members = _active_member_callables()

    latencies: List[float] = []
    metadata_records: List[Dict[str, Any]] = []
    dead_letter_queue: List[Dict[str, Any]] = []
    disagreement_buckets: Dict[str, int] = {"high": 0, "low": 0}

    total = len(canonical_df)
    processed = 0
    start_time = time.time()

    def worker(index: int, row: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        canonical_id = str(row.get("canonical_post_id") or row.get("post_id") or f"row_{index}")
        if canonical_id in processed_ids:
            payload = {field: row.get(field, "") for field in CLASSIFICATION_SCHEMA["properties"].keys()}
            metadata = {"post_id": canonical_id, "disagreement": 0.0, "latency": 0.0, "member_rationales": {}}
            return payload, metadata

        text = f"{row.get('title', '')}\n\n{row.get('body', '')}".strip()
        attempts = 0
        backoff = 1.0
        while attempts < 3:
            attempts += 1
            start = time.time()
            try:
                payload, metadata = ensemble_classify(
                    text=text,
                    engine=classifier,
                    config=run_config.ensemble,
                    member_callables=ensemble_members,
                    disagreement_stats=None,
                )
                latency = time.time() - start
                metadata.setdefault("field_confidence", {})
                payload = dict(payload)
                payload.setdefault(
                    "confidence",
                    float(metadata.get("field_confidence", {}).get("intent", 1.0)),
                )
                for field, schema in CLASSIFICATION_SCHEMA["properties"].items():
                    default_value = 0.0 if schema.get("type") == "number" else ""
                    payload.setdefault(field, default_value)
                validate(instance=payload, schema=CLASSIFICATION_SCHEMA)
                metadata.setdefault("member_rationales", {})
                metadata["member_rationales"] = {
                    key: redact_pii(value or "") for key, value in metadata["member_rationales"].items()
                }
                metadata["post_id"] = canonical_id
                metadata["latency"] = latency
                metadata.setdefault("disagreement", 0.0)
                return payload, metadata
            except ValidationError as exc:
                logger.warning("Schema validation failed on attempt %d: %s", attempts, exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Classification attempt %d failed", attempts)
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
        raise RuntimeError("classification_failed")

    for start_idx in range(0, total, run_config.parallel.chunk_size):
        end_idx = min(total, start_idx + run_config.parallel.chunk_size)
        chunk = [(idx, canonical_df.iloc[idx].copy()) for idx in range(start_idx, end_idx)]
        results = parallel_process_batch(chunk, worker, parallel_cfg=run_config.parallel)
        for (idx, _row), result in zip(chunk, results):
            if isinstance(result, Exception):
                dead_letter_queue.append(
                    {
                        "canonical_post_id": canonical_df.at[idx, "canonical_post_id"],
                        "error": str(result),
                    }
                )
                continue
            payload, metadata = result
            for field, value in payload.items():
                canonical_df.at[idx, field] = value
            metadata_records.append(metadata)
            latencies.append(metadata.get("latency", 0.0))
            bucket = "high" if metadata.get("disagreement", 0.0) >= run_config.ensemble.disagreement_threshold else "low"
            disagreement_buckets[bucket] = disagreement_buckets.get(bucket, 0) + 1
        processed += len(chunk)
        elapsed = time.time() - start_time
        remaining = total - processed
        eta = (elapsed / processed) * remaining if processed else 0.0
        structured_log(
            logging.INFO,
            event="progress",
            processed=processed,
            total=total,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
        )

    canonical_df = assign_splits(canonical_df, clusters, run_config.split)

    evaluation_metrics = evaluate_against_gold(canonical_df, run_config.evaluation)
    historical_drift = compare_against_history(evaluation_metrics, historical_metrics or []) if evaluation_metrics else {}

    ensemble_summary = summarise_member_agreement(metadata_records)
    high_disagreement_cases = [
        record.get("post_id")
        for record in metadata_records
        if record.get("disagreement", 0.0) >= run_config.ensemble.disagreement_threshold
    ]
    ensemble_summary["high_disagreement_ids"] = high_disagreement_cases
    ensemble_summary["buckets"] = disagreement_buckets

    latencies_sorted = sorted(latencies)
    p95_latency = 0.0
    if latencies_sorted:
        index = max(0, min(len(latencies_sorted) - 1, int(0.95 * len(latencies_sorted)) - 1))
        p95_latency = latencies_sorted[index]

    performance_stats: Dict[str, Any] = {
        "total_rows": len(df),
        "total_time_seconds": time.time() - start_time,
        "avg_latency_seconds": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "p95_latency_seconds": float(p95_latency),
        "cache_enabled": run_config.cache.enabled,
        "cache_hits": cache_stats.get("hits", 0),
        "cache_misses": cache_stats.get("misses", 0),
        "dead_letter_count": len(dead_letter_queue),
    }

    duplicate_stats = {"duplicates_removed": duplicates_removed, "clusters": len(clusters)}

    summary_payload = {
        "duplicate_stats": duplicate_stats,
        "cache_stats": cache_stats,
        "ensemble_summary": ensemble_summary,
        "evaluation_metrics": evaluation_metrics,
        "historical_drift": historical_drift,
        "performance_stats": performance_stats,
        "dead_letter_queue": dead_letter_queue,
    }

    return canonical_df, id_mapping, clusters, summary_payload


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path).fillna("")
    df.reset_index(drop=True, inplace=True)
    return df


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Version 2 classification pipeline with deduplication and ensembles")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to configuration YAML")
    parser.add_argument("--input", type=Path, default=Path("data/raw_data.csv"), help="Path to the raw Reddit CSV file")
    parser.add_argument("--output", type=Path, default=Path("data/labeled_sample.csv"), help="Output path for labeled CSV")
    parser.add_argument("--dedupe", choices=["on", "off"], default="on", help="Enable or disable deduplication")
    parser.add_argument("--similarity-threshold", type=float, default=0.5, help="Duplicate similarity threshold")
    parser.add_argument("--soft-similarity-threshold", type=float, default=0.35, help="Soft duplicate similarity threshold")
    parser.add_argument("--cross-subreddit-dedupe", action="store_true", help="Enable cross-subreddit deduplication")
    parser.add_argument("--canonical-policy", choices=["earliest", "longest"], default="earliest", help="Canonical selection policy")
    parser.add_argument("--dedupe-report", type=Path, default=None, help="Optional CSV report for duplicate clusters")
    parser.add_argument("--no-split", action="store_true", help="Disable train/val/test splitting")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--ensemble", choices=["on", "off"], default="off", help="Enable ensemble voting")
    parser.add_argument("--ensemble-members", default="direct,reasoning,rules", help="Comma separated ensemble members")
    parser.add_argument("--ensemble-disagreement-threshold", type=float, default=0.3, help="Threshold for high disagreement logging")
    parser.add_argument("--model", choices=["gpt-4o", "gpt-4o-mini"], default="gpt-4o", help="Model selection")
    parser.add_argument("--temperature", type=float, default=None, help="Override model temperature")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--cache", choices=["on", "off"], default=None, help="Enable or disable response cache")
    parser.add_argument("--cache-ttl", type=int, default=None, help="Cache TTL in hours")
    parser.add_argument("--cache-path", type=Path, default=None, help="Path for cache persistence")
    parser.add_argument("--max-workers", type=int, default=None, help="Number of worker threads")
    parser.add_argument("--rate-limit", type=int, default=None, help="Rate limit in requests per minute")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size for batch processing")
    parser.add_argument("--evaluation", choices=["on", "off"], default=None, help="Enable evaluation mode")
    parser.add_argument("--evaluation-gold-set", type=Path, default=None, help="Path to gold set CSV")
    parser.add_argument("--report-path", type=Path, default=None, help="Path to summary report JSON")
    parser.add_argument("--resume", action="store_true", help="Resume from existing labeled output")
    parser.add_argument("--resume-from", type=Path, default=None, help="Path to existing labeled CSV for resume")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser.parse_args(args)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")

    df = load_dataframe(args.input)

    run_config = build_run_config(args)
    if run_config.resume.enabled and not run_config.resume.checkpoint_path:
        run_config.resume.checkpoint_path = args.output

    cache = ResponseCache(run_config.cache)
    rate_limiter = RateLimiter(run_config.parallel.rate_limit)

    historical_payloads: List[Mapping[str, Any]] = []
    if run_config.report.path and run_config.report.path.exists():
        try:
            with run_config.report.path.open("r", encoding="utf-8") as handle:
                historical_payloads.append(json.load(handle))
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load historical report %s", run_config.report.path)

    canonical_df, id_mapping, clusters, summary_payload = classify_dataframe(
        df,
        run_config=run_config,
        cache=cache,
        rate_limiter=rate_limiter,
        historical_metrics=[payload.get("evaluation", {}) for payload in historical_payloads],
    )

    save_dataframe(canonical_df, args.output)

    config_payload = _normalise_config(dataclasses.asdict(run_config))

    report_summary = generate_summary_report(
        df=canonical_df,
        duplicate_stats=summary_payload.get("duplicate_stats", {}),
        cache_stats=summary_payload.get("cache_stats", {}),
        ensemble_records=summary_payload.get("ensemble_summary", {}),
        evaluation_metrics=summary_payload.get("evaluation_metrics", {}),
        performance_stats=summary_payload.get("performance_stats", {}),
        config_payload=config_payload,
        report_config=run_config.report,
    )

    config_output_path = args.output.with_suffix(".config.json")
    with config_output_path.open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, ensure_ascii=False, indent=2)

    structured_log(
        logging.INFO,
        event="run_complete",
        output=str(args.output),
        canonical_rows=len(canonical_df),
        clusters=len(clusters),
        duplicates_removed=summary_payload.get("duplicate_stats", {}).get("duplicates_removed", 0),
        report=str(run_config.report.path) if run_config.report.path else None,
    )

    if summary_payload.get("dead_letter_queue"):
        structured_log(
            logging.WARNING,
            event="dead_letter_queue",
            count=len(summary_payload["dead_letter_queue"]),
        )

    if summary_payload.get("historical_drift"):
        structured_log(
            logging.INFO,
            event="prompt_drift",
            payload=summary_payload["historical_drift"],
        )

    if summary_payload.get("evaluation_metrics"):
        structured_log(
            logging.INFO,
            event="evaluation_metrics",
            payload=summary_payload["evaluation_metrics"],
        )

    logger.info("Saved labeled dataset to %s", args.output)
    if run_config.report.path:
        logger.info("Summary report available at %s", run_config.report.path)

    mapping_path = args.output.with_suffix(".mapping.json")
    mapping_path.write_text(json.dumps(id_mapping, indent=2), encoding="utf-8")
    logger.info("Wrote ID mapping to %s", mapping_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


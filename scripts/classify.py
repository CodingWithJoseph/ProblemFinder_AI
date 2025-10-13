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
import csv
import dataclasses
import html
import json
import logging
import math
import os
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd


logger = logging.getLogger(__name__)


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


@dataclass
class SplitConfig:
    """Configuration for train/validation/test split ratios."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    enabled: bool = True


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


def _remove_urls(text: str) -> Tuple[str, List[str]]:
    """Remove URLs from ``text`` while collecting them."""

    urls: List[str] = []

    def _collect(match: re.Match[str]) -> str:
        url = match.group(0)
        urls.append(url)
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
    text_index: Dict[str, List[int]] = defaultdict(list)

    for idx, post in enumerate(posts):
        text_index[post.normalised_text].append(idx)
        for url in post.urls:
            url_index[url].append(idx)

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
        if post_a.normalised_text == post_b.normalised_text or shared_url or similarity >= config.similarity_threshold:
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
    confidence: str


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
    }

    RESOLVED_PATTERNS = {
        "randomly got",
        "turns out it was",
        "finally solved",
        "already fixed",
    }

    ADVICE_PATTERNS = {
        "here's my advice",
        "i learned",
        "guide",
        "tips",
        "workflow",
    }

    SOFTWARE_CUES = {
        "app",
        "software",
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
        "purchase",
        "replacement",
        "recommend a",
        "recommend",
        "alternative",
        "which should i",
        "career",
        "job",
        "school",
        "college",
        "printer",
        "hardware",
        "device",
        "samsung",
        "vevor",
        "goxlr",
    }

    PROTOTYPE_PATTERNS = {
        "prototype",
        "sensor",
        "manufacturing",
        "deployment",
    }

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
        if any(pattern in text for pattern in self.PROBLEM_CUES):
            return "seeking_help"
        if any(pattern in text for pattern in self.ADVICE_PATTERNS):
            return "sharing_advice"
        if any(phrase in text for phrase in ["i built", "i made", "launch", "showcase"]):
            return "showcasing"
        return "discussing"

    def _classify_problem(self, text: str, intent: str) -> Tuple[str, str]:
        if any(pattern in text for pattern in self.RESOLVED_PATTERNS):
            return "0", "Describes a resolved situation without an active pain point."

        cues_found = [pattern for pattern in self.PROBLEM_CUES if pattern in text]
        if intent == "showcasing" and not cues_found:
            return "0", "Showcase content without evidence of a pain point."

        if cues_found:
            reason = f"Detected problem cues: {', '.join(sorted(set(cues_found))[:3])}."
            return "1", reason

        if intent == "sharing_advice":
            return "1", "Advice-sharing anchored in a real frustration (Version 2 guidance)."

        return "0", "No unresolved pain point, request, or frustration detected."

    def _classify_software(self, text: str, is_problem: str) -> Tuple[str, str]:
        if is_problem != "1":
            return "0", "Not a problem, so downstream labels forced to 0."

        software_hits = [cue for cue in self.SOFTWARE_CUES if cue in text]
        non_software_hits = [cue for cue in self.NON_SOFTWARE_CUES if cue in text]

        if "looking for" in text and "software" in text and not any(word in text for word in {"build", "create", "develop"}):
            return "0", "Requesting an existing software product (market search)."

        if software_hits and not non_software_hits:
            return "1", f"Software signals present ({', '.join(sorted(set(software_hits))[:3])})."

        if non_software_hits and not software_hits:
            return "0", f"Looks like market/knowledge request ({', '.join(sorted(set(non_software_hits))[:3])})."

        if "how do i" in text or "how to" in text:
            return "0", "Learning/information-seeking question rather than malfunction."

        if "driver" in software_hits or "update" in software_hits:
            return "1", "Hardware symptom tied to software/driver cue."

        if software_hits and non_software_hits:
            return "0", "Mixed hardware/product request — not purely software solvable."

        return "0", "No decisive software cues detected."

    def _classify_external(self, text: str, is_problem: str) -> Tuple[str, str]:
        if is_problem != "1":
            return "0", "Not a problem, so downstream labels forced to 0."

        external_hits = [cue for cue in self.EXTERNAL_CUES if cue in text]
        prototype_hits = [cue for cue in self.PROTOTYPE_PATTERNS if cue in text]

        if "which" in text and "should i" in text:
            return "1", "Choosing between market options — requires external solution."

        if "job" in text and "advice" in text:
            return "0", "Career advice rant does not require external coordination."

        if external_hits:
            return "1", f"Requires external coordination ({', '.join(sorted(set(external_hits))[:3])})."

        if prototype_hits:
            return "1", "Prototype involves hardware/manufacturing components."

        if "how do i" in text or "how to" in text:
            return "0", "Knowledge gap solvable by the user with guidance."

        return "0", "No evidence of external coordination required."

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

    def _confidence(self, text: str, patterns: List[str]) -> str:
        if any(word in text for word in self.UNCERTAINTY_WORDS):
            return "low"
        if patterns:
            return "medium"
        return "high"


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


def classify_dataframe(df: pd.DataFrame, *, dedupe_config: DedupeConfig, split_config: SplitConfig) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, List[str]]]:
    """Run deduplication and classification on ``df``."""

    logger.info("Starting deduplication (enabled=%s, threshold=%.2f)", dedupe_config.enabled, dedupe_config.similarity_threshold)
    canonical_df, id_mapping, clusters = deduplicate_dataframe(df, dedupe_config)
    logger.info("Deduplication complete: %d -> %d canonical posts (%d clusters)", len(df), len(canonical_df), len(clusters))

    if dedupe_config.report_path:
        write_dedupe_report(dedupe_config.report_path, clusters, canonical_df)
        logger.info("Dedupe report written to %s", dedupe_config.report_path)

    classifier = Version2RuleEngine()
    for idx, row in canonical_df.iterrows():
        text = f"{row.get('title', '')}\n\n{row.get('body', '')}".strip()
        classification = classifier.classify(text)
        canonical_df.at[idx, "intent"] = classification.intent
        canonical_df.at[idx, "is_problem"] = classification.is_problem
        canonical_df.at[idx, "is_software_solvable"] = classification.is_software_solvable
        canonical_df.at[idx, "is_external"] = classification.is_external
        canonical_df.at[idx, "problem_reason"] = classification.problem_reason
        canonical_df.at[idx, "software_reason"] = classification.software_reason
        canonical_df.at[idx, "external_reason"] = classification.external_reason
        canonical_df.at[idx, "detected_patterns"] = classification.detected_patterns
        canonical_df.at[idx, "confidence"] = classification.confidence

    canonical_df = assign_splits(canonical_df, clusters, split_config)
    return canonical_df, id_mapping, clusters


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
    parser = argparse.ArgumentParser(description="Run Version 2 classification pipeline with deduplication")
    parser.add_argument("--input", type=Path, default=Path("data/raw_data.csv"), help="Path to the raw Reddit CSV file")
    parser.add_argument("--output", type=Path, default=Path("data/labeled_sample.csv"), help="Output path for labeled CSV")
    parser.add_argument("--dedupe", choices=["on", "off"], default="on", help="Enable or disable deduplication")
    parser.add_argument("--similarity-threshold", type=float, default=0.5, help="Duplicate similarity threshold")
    parser.add_argument("--canonical-policy", choices=["earliest", "longest"], default="earliest", help="Canonical selection policy")
    parser.add_argument("--dedupe-report", type=Path, default=None, help="Optional CSV report for duplicate clusters")
    parser.add_argument("--no-split", action="store_true", help="Disable train/val/test splitting")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser.parse_args(args)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")

    df = load_dataframe(args.input)

    dedupe_config = DedupeConfig(
        enabled=args.dedupe == "on",
        similarity_threshold=args.similarity_threshold,
        canonical_policy=args.canonical_policy,
        report_path=args.dedupe_report,
    )

    split_config = SplitConfig(
        enabled=not args.no_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    canonical_df, id_mapping, clusters = classify_dataframe(df, dedupe_config=dedupe_config, split_config=split_config)
    save_dataframe(canonical_df, args.output)

    logger.info("Saved labeled dataset to %s", args.output)
    logger.info("Processed %d posts (canonical). Dedupe clusters: %d", len(canonical_df), len(clusters))

    mapping_path = args.output.with_suffix(".mapping.json")
    mapping_path.write_text(json.dumps(id_mapping, indent=2), encoding="utf-8")
    logger.info("Wrote ID mapping to %s", mapping_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


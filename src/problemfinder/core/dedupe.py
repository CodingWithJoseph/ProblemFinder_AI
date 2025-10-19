"""Deduplication utilities for Reddit post datasets."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd

from problemfinder.core.config import DedupeConfig
from problemfinder.utils.text import (
    combined_similarity,
    normalise_text_block,
    normalise_whitespace,
    remove_urls,
)


@dataclass(slots=True)
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


class UnionFind:
    """Union-Find/Disjoint-set implementation for clustering duplicates."""

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def find(self, item: int) -> int:
        """Find the canonical parent representative for ``item``."""

        if self.parent.get(item, item) != item:
            self.parent[item] = self.find(self.parent[item])
        else:
            self.parent.setdefault(item, item)
            self.rank.setdefault(item, 0)
        return self.parent[item]

    def union(self, a: int, b: int) -> None:
        """Union two sets using union-by-rank heuristics."""

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


def normalise_post(row: pd.Series, *, post_id: str) -> NormalisedPost:
    """Create a :class:`NormalisedPost` by applying canonical cleaning steps."""
    title = (row.get("title", "") or "")
    body = (row.get("body", "") or "")
    combined = f"{title}\n\n{body}".strip()

    normalised = normalise_text_block(combined)
    _, urls = remove_urls(combined)
    cleaned_urls = [
        normalise_whitespace(re.sub(r"[)\]\.,;:!?]+$", "", url.lower()))
        for url in urls
        if url
    ]

    subreddit = str(row.get("subreddit", "")).strip()
    created_utc_raw = row.get("created_utc", math.inf)
    created_utc = float(created_utc_raw) if pd.notna(created_utc_raw) else math.inf

    return NormalisedPost(
        post_id=post_id,
        index=int(row.name),
        combined_text=combined,
        normalised_text=normalised,
        urls=cleaned_urls,
        subreddit=subreddit,
        created_utc=created_utc,
        body_length=len(body or ""),
    )

def _iter_candidate_pairs(posts: Sequence[NormalisedPost]) -> Iterator[Tuple[int, int]]:
    """
    Takes in a list of NormalisedPost objects.
    Yields pairs of indices (i, j) where post i and post j are potential duplicates.
    It doesn’t compute similarity here — it just proposes pairs to check.
    """

    """
    Build lookup indices
    Each index maps a feature (token, URL, domain, or normalized text) → list of post indices that share it.
    
    token_index["problem"] = [3, 7, 12, 14]
    url_index["https://example.com"] = [2, 5]
    
    This lets us later say “any two posts that appear in the same list might be duplicates.”
    """
    token_index: Dict[str, List[int]] = defaultdict(list)
    url_index: Dict[str, List[int]] = defaultdict(list)
    domain_index: Dict[str, List[int]] = defaultdict(list)
    text_index: Dict[str, List[int]] = defaultdict(list)

    # Fill the indices: Loop through every normalized post.
    # If two posts have exactly the same normalized text, they’re probably duplicates.
    # So we’ll later yield pairs from text_index.
    for idx, post in enumerate(posts):
        text_index[post.normalised_text].append(idx)

        # If two posts share the same URL, they might refer to the same content.
        # If they share the same domain (like GitHub.com), they might still be related — so we record that too.
        for url in post.urls:
            url_index[url].append(idx)
            domain = urlparse(url).netloc
            if domain:
                domain_index[domain].append(idx)

        # Index by tokens: Split each post into tokens (words).
        # Keep only those longer than 3 characters to skip noise like “a”, “is”, “to”.
        # Use a set (unique_tokens) so each token per post is counted once, preventing self-pair inflation.
        tokens = post.normalised_text.split()
        unique_tokens = {tok for tok in tokens if len(tok) > 3}
        for token in unique_tokens:
            token_index[token].append(idx)

    seen_pairs: set[Tuple[int, int]] = set()
    start_time = time.time()
    count = 0

    def _yield_unique(indices: List[int]):
        nonlocal count
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                pair = (min(a, b), max(a, b))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    count += 1
                    if count % 10_000 == 0:
                        elapsed = time.time() - start_time
                        print(f"[dedupe] generated {count:,} pairs in {elapsed:.1f}s")
                    yield pair

    # Each block type — skip overly large token buckets
    for indices in text_index.values():
        if len(indices) > 1:
            yield from _yield_unique(indices)

    for indices in url_index.values():
        if len(indices) > 1:
            yield from _yield_unique(indices)

    for indices in domain_index.values():
        if len(indices) > 1:
            yield from _yield_unique(indices)

    for token, indices in token_index.items():
        if len(indices) < 2 or len(indices) > 100:
            continue  # ignore rare and super-common tokens
        yield from _yield_unique(indices)


def compute_pair_similarity(args):
    """Compute similarity and matching signals for one post pair."""
    a_idx, b_idx, posts, config = args
    post_a, post_b = posts[a_idx], posts[b_idx]

    sim = combined_similarity(post_a.normalised_text, post_b.normalised_text)
    shared_url = bool(set(post_a.urls) & set(post_b.urls))
    soft_match = (
        sim >= config.soft_similarity_threshold
        and post_a.subreddit != post_b.subreddit
    )

    return a_idx, b_idx, sim, shared_url, soft_match


# --- Helper: batch processing to reduce IPC overhead ---
def _process_batch(batch, posts, config):
    """Compute similarities for a batch of pairs."""
    return [compute_pair_similarity((a, b, posts, config)) for (a, b) in batch]


def cluster_duplicates(posts: Sequence[NormalisedPost], config: DedupeConfig, n_workers=None, batch_size=1000) -> Dict[int, List[int]]:
    """Cluster posts using similarity and URL signals."""

    if not config.enabled or len(posts) <= 1:
        return {idx: [idx] for idx in range(len(posts))}

    uf = UnionFind()

    # Generate candidate pairs
    candidate_pairs = list(_iter_candidate_pairs(posts))
    total_pairs = len(candidate_pairs)

    print(f"[dedupe] {total_pairs:,} candidate pairs generated for {len(posts):,} posts")

    # Initialize the process pool
    n_workers = n_workers or max(1, os.cpu_count()-4 or 4)
    print(f"[dedupe] Using {n_workers} CPU workers (batch size = {batch_size})")

    # Submit batches of pairs to workers for processing
    def batch_iter(seq, size):
        for post in range(0, len(seq), size):
            yield seq[post:post + size]


    processed = 0
    start_time = time.time()

    unique_pairs = set(candidate_pairs)
    print(f"Unique pairs: {len(unique_pairs):,}, raw pairs: {len(candidate_pairs):,}")


    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_process_batch, batch, posts, config) for batch in batch_iter(candidate_pairs, batch_size)
        ]

        unique_pairs = set(candidate_pairs)
        print(f"Unique pairs: {len(unique_pairs):,}, raw pairs: {len(candidate_pairs):,}")

        for future in as_completed(futures):
            results = future.result()
            for a_idx, b_idx, similarity, shared_url, soft_match in results:
                post_a, post_b = posts[a_idx], posts[b_idx]
                if (
                        post_a.normalised_text == post_b.normalised_text
                        or shared_url
                        or similarity >= config.similarity_threshold
                        or (config.cross_subreddit and soft_match)
                ):
                    uf.union(a_idx, b_idx)
            processed += len(results)
            if processed % 10_000 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed else 0
                print(f"[dedupe] {processed:,}/{total_pairs:,} pairs processed ({rate:,.0f}/s)")

    # Build the final cluster assignments
    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(posts)):
        root = uf.find(idx)
        clusters[root].append(idx)

    elapsed = time.time() - start_time
    print(f"[dedupe] Parallel dedupe complete in {elapsed:.1f}s → {len(clusters):,} clusters")

    return dict(clusters)


def _canonical_index(indices: List[int], posts: Sequence[NormalisedPost], policy: str) -> int:
    """Select the canonical representative index for a cluster."""

    candidates = [posts[i] for i in indices]

    if policy == "longest":
        candidates.sort(key=lambda p: (-p.body_length, p.created_utc, p.post_id))
    else:
        candidates.sort(key=lambda p: (p.created_utc, -p.body_length, p.post_id))

    return posts.index(candidates[0])


def deduplicate_dataframe(
    df: pd.DataFrame,
    config: DedupeConfig,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, List[str]]]:
    """Deduplicate dataframe rows and return canonical dataframe and mapping."""

    df = df.copy().reset_index(drop=True)

    if "post_id" in df.columns:
        post_ids = df["post_id"].astype(str).tolist()
    elif "url" in df.columns:
        post_ids = df["url"].fillna("").astype(str).tolist()
    else:
        post_ids = [f"row_{i}" for i in range(len(df))]

    posts = [normalise_post(df.iloc[i], post_id=post_ids[i]) for i in range(len(df))]
    clusters = cluster_duplicates(posts, config)

    canonical_rows: List[pd.DataFrame] = []
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

    if not canonical_rows:
        empty_df = df.iloc[0:0].copy()
        for column in ("canonical_post_id", "duplicate_post_ids", "duplicate_subreddits", "normalized_text"):
            if column not in empty_df.columns:
                empty_df[column] = pd.Series(dtype="object")
        return empty_df.reset_index(drop=True), id_mapping, cluster_members

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

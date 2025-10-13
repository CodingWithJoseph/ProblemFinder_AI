"""Tests for deduplication logic."""

import json

import pandas as pd

from scripts.classify import DedupeConfig, deduplicate_dataframe


def _make_df() -> pd.DataFrame:
    data = [
        {
            "title": "Help! My printer smears ink",
            "body": "The Canon 5000 is broken again.",
            "subreddit": "rprinters",
            "url": "https://reddit.com/a1",
            "created_utc": 170000,
        },
        {
            "title": "HELP! my printer smears ink",
            "body": "Canon 5000 keeps failing.",
            "subreddit": "fixit",
            "url": "https://reddit.com/a2",
            "created_utc": 170100,
        },
        {
            "title": "Crosspost from r/fixit",
            "body": "Help! My printer smears ink again.",
            "subreddit": "printers",
            "url": "https://reddit.com/a3",
            "created_utc": 170050,
        },
        {
            "title": "Unrelated story",
            "body": "I built a website yesterday.",
            "subreddit": "webdev",
            "url": "https://reddit.com/a4",
            "created_utc": 170200,
        },
    ]
    return pd.DataFrame(data)


def test_deduplicate_dataframe_clusters_near_duplicates():
    df = _make_df()
    config = DedupeConfig(enabled=True, similarity_threshold=0.5, canonical_policy="earliest")

    canonical_df, mapping, clusters = deduplicate_dataframe(df, config)

    assert len(canonical_df) == 2

    canonical_ids = {row["canonical_post_id"] for _, row in canonical_df.iterrows()}
    assert mapping["https://reddit.com/a1"] in canonical_ids
    assert mapping["https://reddit.com/a2"] == mapping["https://reddit.com/a1"]
    assert mapping["https://reddit.com/a3"] == mapping["https://reddit.com/a1"]
    assert mapping["https://reddit.com/a4"] != mapping["https://reddit.com/a1"]

    cluster_members = clusters[mapping["https://reddit.com/a1"]]
    assert sorted(cluster_members) == [
        "https://reddit.com/a1",
        "https://reddit.com/a2",
        "https://reddit.com/a3",
    ]

    subreddits = json.loads(
        canonical_df.loc[canonical_df["canonical_post_id"] == mapping["https://reddit.com/a1"], "duplicate_subreddits"].iat[0]
    )
    assert sorted(subreddits) == ["fixit", "printers", "rprinters"]

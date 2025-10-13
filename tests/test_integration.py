import json

import pandas as pd

from problemfinder.core.config import DedupeConfig, SplitConfig
from problemfinder.core.pipeline import classify_dataframe


def test_pipeline_dedupes_and_assigns_splits_consistently():
    data = [
        {
            "title": "Need budget projector recommendations",
            "body": "Looking for a projector under $400 that works in small rooms.",
            "subreddit": "gadgets",
            "url": "https://reddit.com/p1",
            "created_utc": 170,
        },
        {
            "title": "need budget projector recommendation",
            "body": "Looking for a projector under $400 that works in a small room.",
            "subreddit": "hometheater",
            "url": "https://reddit.com/p2",
            "created_utc": 171,
        },
        {
            "title": "Vevor Smart1 software rant",
            "body": "The bundled software is terrible but here is what I use now.",
            "subreddit": "makers",
            "url": "https://reddit.com/p3",
            "created_utc": 172,
        },
        {
            "title": "Horse blanket LoRa sensor prototype",
            "body": "Need help designing a sensor prototype using LoRa and external hardware.",
            "subreddit": "iot",
            "url": "https://reddit.com/p4",
            "created_utc": 173,
        },
    ]

    df = pd.DataFrame(data)
    dedupe_config = DedupeConfig(enabled=True, similarity_threshold=0.5, canonical_policy="earliest")
    split_config = SplitConfig(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, enabled=True)

    canonical_df, mapping, clusters, _ = classify_dataframe(
        df, dedupe_config=dedupe_config, split_config=split_config
    )

    assert len(canonical_df) == 3
    assert len(clusters) == 3

    mapping_values = set(mapping.values())
    assert len(mapping_values) == 3

    duplicate_canonical = mapping["https://reddit.com/p1"]
    duplicate_split = canonical_df.loc[canonical_df["canonical_post_id"] == duplicate_canonical, "split"].iat[0]
    other_split = canonical_df.loc[canonical_df["canonical_post_id"] == mapping["https://reddit.com/p3"], "split"].iat[0]
    assert duplicate_split in {"train", "val", "test"}
    assert other_split in {"train", "val", "test"}

    projector_row = canonical_df.loc[canonical_df["canonical_post_id"] == duplicate_canonical].iloc[0]
    assert projector_row["is_problem"] == "1"
    assert projector_row["is_external"] == "1"

    prototype_row = canonical_df.loc[canonical_df["canonical_post_id"] == mapping["https://reddit.com/p4"]].iloc[0]
    assert prototype_row["is_external"] == "1"

    subreddits = json.loads(projector_row["duplicate_subreddits"])
    assert sorted(subreddits) == ["gadgets", "hometheater"]

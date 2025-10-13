"""Tests for text normalisation utilities used in deduplication."""

import pandas as pd

from scripts.classify import normalise_post


def test_normalise_post_removes_urls_markdown_and_mentions():
    row = pd.Series(
        {
            "title": "Crosspost from r/Tech",
            "body": "Check this out! [link](https://example.com) `code` <b>bold</b> 😊 /u/user",
            "subreddit": "SampleSub",
            "created_utc": 1700000000,
        },
        name=0,
    )

    post = normalise_post(row, post_id="abc123")

    assert post.normalised_text == "check this out link bold"
    assert post.urls == ["https://example.com"]
    assert post.subreddit == "SampleSub"

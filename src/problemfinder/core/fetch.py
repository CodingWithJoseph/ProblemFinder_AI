"""Utilities for fetching Reddit posts via PRAW."""
from __future__ import annotations

import praw
import pandas as pd
from typing import Dict, List
from problemfinder.settings import load_environment, require_env

# Define search queries
search_queries = [
    # Direct Software Requests
    "looking for software that",
    "need an app that",
    "app recommendation for",
    "software solution for",
    "looking for program to",

    # Problem Indicators
    "can't find software",
    "wish there was an app",
    "frustrated with current app",
    "alternative to",
    "better way to",

    # Solution Seeking
    "how to automate",
    "tool to help with",
    "software to manage",
    "app to track",
    "need help organizing",

    # Pain Points
    "tired of manually",
    "waste time doing",
    "struggling with",
    "annoying problem with",
    "there must be a better way",

    # Market Research
    "why isn't there",
    "does software exist for",
    "looking for solution to",
    "anyone know tool for",
    "recommend software for"
]

def get_reddit_instance() -> praw.Reddit:
    """Return a configured Reddit API client."""

    load_environment()
    return praw.Reddit(
        client_id=require_env("REDDIT_CLIENT_ID"),
        client_secret=require_env("REDDIT_CLIENT_SECRET"),
        user_agent=require_env("REDDIT_USER_AGENT"),
    )


def fetch_posts(q: str, *, limit: int = 50) -> List[Dict[str, object]]:
    """Fetch posts matching ``query`` from r/all."""

    reddit = get_reddit_instance()
    results: List[Dict[str, object]] = []
    for p in reddit.subreddit("all").search(q, sort="new", limit=limit):
        results.append(
            {
                "title": p.title,
                "body": p.selftext,
                "subreddit": p.subreddit.display_name,
                "url": p.url,
                "score": p.score,
                "num_comments": p.num_comments,
                "upvote_ratio": p.upvote_ratio,
                "created_utc": p.created_utc,
                "author": str(p.author),
                "is_original_content": p.is_original_content,
                "edited": bool(p.edited),
                "link_flair_text": p.link_flair_text,
                "total_awards_received": p.total_awards_received,
                "gilded": p.gilded,
            }
        )
    return results


if __name__ == "__main__":
    all_posts = []

    # Fetch posts for each query
    for query in search_queries:
        posts = fetch_posts(q=query, limit=1000)
        for post in posts:
            post['search_query'] = query  # Add the search query that found this post
        all_posts.extend(posts)
        print(f"Fetched {len(posts)} posts for query: {query}")
        print(f"Total posts fetched: {len(all_posts)}")

    # Convert to DataFrame
    df = pd.DataFrame(all_posts)

    # Remove duplicates based on URL (in case same post found by different queries)
    df = df.drop_duplicates(subset=['url'])

    # Save all posts
    df.to_csv("data/raw_reddit_posts.csv", index=False)
    print(f"Saved {len(df)} posts.")

    # Create labeled dataset sample
    shuffled_data = df.sample(frac=1, random_state=42).reset_index(drop=True)
    label_data = shuffled_data[:500]
    label_data.to_csv("data/labeled_post.csv", index=False)
    print(f"Saved {len(label_data)} posts for labeling.")

__all__ = ["fetch_posts", "get_reddit_instance"]

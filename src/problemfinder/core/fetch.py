"""Utilities for fetching Reddit posts via PRAW."""

from __future__ import annotations

from typing import Dict, List

import praw

from problemfinder.settings import load_environment, require_env


def get_reddit_instance() -> praw.Reddit:
    """Return a configured Reddit API client."""

    load_environment()
    return praw.Reddit(
        client_id=require_env("REDDIT_CLIENT_ID"),
        client_secret=require_env("REDDIT_CLIENT_SECRET"),
        user_agent=require_env("REDDIT_USER_AGENT"),
    )


def fetch_posts(query: str, *, limit: int = 50) -> List[Dict[str, object]]:
    """Fetch posts matching ``query`` from r/all."""

    reddit = get_reddit_instance()
    posts: List[Dict[str, object]] = []
    for post in reddit.subreddit("all").search(query, sort="new", limit=limit):
        posts.append(
            {
                "title": post.title,
                "body": post.selftext,
                "subreddit": post.subreddit.display_name,
                "url": post.url,
                "score": post.score,
                "num_comments": post.num_comments,
                "upvote_ratio": post.upvote_ratio,
                "created_utc": post.created_utc,
                "author": str(post.author),
                "is_original_content": post.is_original_content,
                "edited": bool(post.edited),
                "link_flair_text": post.link_flair_text,
                "total_awards_received": post.total_awards_received,
                "gilded": post.gilded,
            }
        )
    return posts


__all__ = ["fetch_posts", "get_reddit_instance"]

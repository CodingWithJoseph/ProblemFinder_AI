from dotenv import load_dotenv
import praw
import os

# Load environment variables from the .env file
load_dotenv()


def get_reddit_instance():
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )


def fetch_posts(query="looking for an app", limit=50):
    reddit = get_reddit_instance()
    posts = []
    for post in reddit.subreddit("all").search(query, sort="new", limit=limit):
        posts.append({
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
            "gilded": post.gilded
        })

    return posts
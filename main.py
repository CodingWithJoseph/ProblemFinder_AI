import pandas as pd
from fetch import fetch_posts

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

if __name__ == "__main__":
    all_posts = []

    # Fetch posts for each query
    for query in search_queries:
        posts = fetch_posts(query=query, limit=1000)
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
    df.to_csv("data/raw_data.csv", index=False)
    print(f"Saved {len(df)} posts.")

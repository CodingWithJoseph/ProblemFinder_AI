from miner.fetch import fetch_posts
import pandas as pd

if __name__ == "__main__":
    posts = fetch_posts()
    df = pd.DataFrame(posts)
    df.to_csv("data/raw_reddit_posts.csv", index=False)
    print(f"Saved {len(df)} posts.")
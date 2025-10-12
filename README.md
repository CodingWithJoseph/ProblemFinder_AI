# Reddit Problem Miner

## Project Overview

**Reddit Problem Miner** is a tool that finds Reddit posts describing real-world software-solvable problems. The goal is to identify potential startup ideas by detecting pain points users express in public forums. By mining Reddit, we can discover unmet needs and explore innovative solutions through software.

## Features

- Fetches Reddit posts from all subreddits using specific search queries.
- Identifies posts that describe solvable problems.
- Outputs a list of high-signal posts with relevant details (title, body, subreddit, URL).

## Why?

This project follows the philosophy of building products quickly based on real user problems. It provides an opportunity to practice rapid iteration while creating something that could potentially lead to new software products or startups.

## How It Works

1. The tool queries Reddit using a set of predefined search patterns like:
    - “any tool for”
    - “wish there was”
    - “looking for an app”
2. Posts are fetched from across all subreddits.
3. The text is analyzed to determine whether the post describes a solvable problem.
4. Results are displayed or saved for further analysis.

## Milestones

- [x] Set up project and Reddit API access
- [x] Collect Reddit posts using broad search queries
- [X] Label 300 posts to train a machine learning model
- [ ] Build and test a problem-detection model

## Setup Instructions

### Prerequisites
- Python 3.x
- A Reddit API account (to get your `client_id`, `client_secret`, and `user_agent`)

## Contributing

Feel free to fork and contribute to the project. Pull requests are welcome.

## License

This project is open-source and available under the MIT License.


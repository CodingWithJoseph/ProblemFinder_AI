import json

"""
Classify a sample of Reddit posts using OpenAI prompts.

This script loads the first 100 rows of the raw dataset, combines the title and
body into a structured text block, and classifies each entry with three binary labels:
- is_problem: "1" if the post describes a problem/pain point, else "0".
- is_software_solvable: "1" if the problem could be solved primarily with software (apps, automation, algorithms).
- is_external: "1" if solving the problem requires coordinating physical resources, people-as-service, logistics, or hardware.

Notes
- If is_problem == "0", both is_software_solvable and is_external are set to "0".
- If is_problem == "error", both is_software_solvable and is_external are set to "error".
- All three labels are in {"0", "1", "error"} for consistency.
- Results overwrite ``data/labeled_sample.csv``.

Key features
- Modularized functions with type hints and docstrings
- Strict parsing/validation to enforce clean outputs
- Retry logic with exponential backoff
- Clean, structured prompt formatting for better model performance
"""

from __future__ import annotations

import logging
import os
import time
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------
# Prompt Templates
# -----------------------------
PROBLEM_PROMPT = """You are a classifier that determines if a Reddit post describes a problem.

A "problem" means:
- The author is frustrated by something, or
- They ask for a better solution, or
- They mention something that doesn't work well, is missing, or could be improved.

A post is NOT a problem if it is:
- Just sharing an opinion or fact,
- Promoting or showcasing something,
- Asking a neutral question without frustration or a need.

Return ONLY one of:
- 1  (if it clearly describes a problem, pain point, or unmet need)
- 0  (if it does not)

Post:
{post_text}
"""

SOFTWARE_SOLVABLE_PROMPT = """You are a classifier that determines if a problem described in a Reddit post could be solved primarily with software.

"Software-solvable" means:
- The core solution could be built as software (apps, websites, APIs, automation, algorithms).
- People may be users, customers, or participants
- People do not need to be actively coordinated as a service for the solution to function.

Return ONLY one of:
- 1  (if primarily solvable by software)
- 0  (if not primarily solvable by software)

Post (this post IS a problem):
{post_text}
"""

EXTERNAL_REQUIRED_PROMPT = """
You are a classifier that determines if solving a problem requires action or coordination beyond the individual user's direct control.

"External" means:
- The solution depends on resources, people, or systems *outside* the user's immediate ability to change, such as:
  - Manufacturer or company intervention (e.g., warranty, firmware patch, customer service)
  - Physical-world coordination (technicians, logistics, infrastructure, hardware deployment)
  - Institutional or organizational processes (government, insurance, regulations)
- "Not external" means the user can solve it alone through personal action, settings, or software.

Return ONLY one of:
- 1  (if external action or coordination is required to solve)
- 0  (if the user can fully resolve it themselves)

Post (this post IS a problem):
{post_text}
"""

FULL_REASON_PROMPT = """
You are a precise classifier for Reddit posts. 
Your task is to assign three binary labels and a short reason (one sentence each) explaining the decision for each label.

Label definitions:
1. is_problem: 1 if the post describes or implies a problem, difficulty, or question seeking a solution. Otherwise 0.
2. is_software_solvable: 1 if the main problem can be solved primarily through software, code, configuration, or digital settings. Only 1 if is_problem == 1.
3. is_external: 1 if solving the problem requires action or coordination beyond the user's control (e.g., manufacturer, company, government, or physical service). Only 1 if is_problem == 1.

Rules:
- If is_problem = 0, both is_software_solvable and is_external must be 0.
- Base decisions on how the problem can be solved, not who caused it.
- Give one short, factual s
"""

# -----------------------------
# Types
# -----------------------------
Label = Literal["0", "1", "error"]


# -----------------------------
# OpenAI Client Initialization
# -----------------------------

def init_client() -> OpenAI:
    """Initialize and return an OpenAI client using environment variables."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment.")
        raise RuntimeError("OPENAI_API_KEY not found in environment.")
    return OpenAI(api_key=api_key)


# -----------------------------
# Data IO
# -----------------------------
def load_data(path: str = "data/raw_data.csv", sample_size: int = 100) -> pd.DataFrame:
    """Load raw CSV and return first `sample_size` rows (filled NaNs)."""
    logger.info("Loading raw data from %s…", path)
    df = pd.read_csv(path).fillna("")
    sample_df = df.head(sample_size).copy()
    if sample_df.empty:
        logger.warning("No data found in raw dataset.")
    else:
        logger.info("Loaded %d rows.", len(sample_df))
    return sample_df


def save_results(df: pd.DataFrame, path: str = "data/labeled_sample.csv") -> None:
    """Save labeled DataFrame to CSV."""
    df.to_csv(path, index=False)
    logger.info("Classification complete. Results saved to %s", path)


def format_post(row: pd.Series) -> str:
    """Build structured post text for prompting."""
    title = (row.get("title", "") or "").strip()
    body = (row.get("body", "") or "").strip()
    return f"""### Title
{title}

### Body
{body}
""".strip()


# -----------------------------
# OpenAI Call with Retry
# -----------------------------
def _call_with_retry(
        client: OpenAI,
        prompt: str,
        *,
        model: str = "gpt-4o",
        max_attempts: int = 3,
        initial_delay: float = 1.0,
) -> Optional[str]:
    """Call the OpenAI Responses API with retry logic; return text or None."""
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.responses.create(model=model, input=prompt)
            text = response.output_text if hasattr(response, "output_text") else str(response)
            return (text or "").strip()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Attempt %d failed: %s", attempt, exc)
            if attempt == max_attempts:
                return None
            time.sleep(delay)
            delay *= 2
    return None


# -----------------------------
# Parsing / Validation
# -----------------------------
def _normalize_binary_token(raw: Optional[str]) -> Label:
    """
    Normalize a model output to {"0","1","error"}.
    - Take first line and first token
    - Lowercase, strip surrounding punctuation
    - Map 'yes'->'1', 'no'->'0'
    """
    if raw is None:
        return "error"
    token = (raw or "").strip().splitlines()[0].strip().split()[0].strip(".,:;\"'`").lower()
    if token in {"0", "1"}:
        return token  # type: ignore[return-value]
    if token in {"yes", "y"}:
        return "1"
    if token in {"no", "n"}:
        return "0"
    logger.warning("Unparsable binary output: %r -> 'error'", raw)
    return "error"


def _decode_raw_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.decoder.JSONDecodeError:
        return {'error': raw}


# -----------------------------
# Classifiers
# -----------------------------
def classify_is_problem(client: OpenAI, post_text: str) -> Label:
    """Return '1' if post is a problem, '0' if not, 'error' on failure."""
    raw = _call_with_retry(client, PROBLEM_PROMPT.format(post_text=post_text))
    return _normalize_binary_token(raw)


def classify_is_software_solvable(client: OpenAI, post_text: str) -> Label:
    """Return '1' if primarily solvable by software, '0' if not, 'error' on failure."""
    raw = _call_with_retry(client, SOFTWARE_SOLVABLE_PROMPT.format(post_text=post_text))
    return _normalize_binary_token(raw)


def classify_is_external(client: OpenAI, post_text: str) -> Label:
    """Return '1' if external coordination/resources required, '0' if not, 'error' on failure."""
    raw = _call_with_retry(client, EXTERNAL_REQUIRED_PROMPT.format(post_text=post_text))
    return _normalize_binary_token(raw)


def describe_rationale(client: OpenAI, post_text: str) -> dict[str, str]:
    """Return '1' if external coordination/resources required, '0' if not, 'error' on failure."""
    raw = _call_with_retry(client, FULL_REASON_PROMPT.format(post_text=post_text))
    return _decode_raw_json(raw)


# -----------------------------
# Processing
# -----------------------------
def process_posts(client: OpenAI, df: pd.DataFrame, sleep_seconds: float = 0.5) -> pd.DataFrame:
    """
    Iterate through posts, classify each, and return a labeled DataFrame with columns:
    title, body, is_problem, is_software_solvable, is_external.

    Pipeline:
    - First run is_problem.
    - If "1", run both is_software_solvable and is_external.
    - If "0", set both to "0".
    - If "error", set both to "error".
    """
    results: list[dict[str, str]] = []
    total = len(df)
    logger.info("Processing %d posts…", total)

    for idx, row in df.iterrows():
        position = len(results) + 1
        logger.info("Processing post %d/%d (index %s)…", position, total, idx)

        post_text = format_post(row)
        is_problem = classify_is_problem(client, post_text)

        if is_problem == "1":
            is_software = classify_is_software_solvable(client, post_text)
            is_external = classify_is_external(client, post_text)
            rationale = describe_rationale(client, post_text)
        elif is_problem == "0":
            is_software = "0"
            is_external = "0"
            rationale = "None"
        else:  # "error"
            is_software = "error"
            is_external = "error"
            rationale = "None"

        results.append(
            {
                "title": (row.get("title", "") or ""),
                "body": (row.get("body", "") or ""),
                "is_problem": is_problem,
                "is_software_solvable": is_software,
                "is_external": is_external,
                'rationale': rationale
            }
        )

        time.sleep(sleep_seconds)

    # Ensure desired column order
    out_df = pd.DataFrame(results)[
        ["title", "body", "is_problem", "is_software_solvable", "is_external", "rationale"]
    ]
    return out_df


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    """Entry point: load data, classify, and save results."""
    client = init_client()
    sample_df = load_data("data/raw_data.csv", sample_size=100)
    if sample_df.empty:
        logger.warning("No data to process. Exiting.")
        return
    labeled_df = process_posts(client, sample_df, sleep_seconds=0.5)
    save_results(labeled_df, "data/labeled_sample.csv")


if __name__ == "__main__":
    main()

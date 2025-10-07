"""Classify a sample of Reddit posts using OpenAI prompts.

This script loads the first 50 rows of the raw dataset, combines the title and
body into a single text block, classifies each entry for whether it describes a
problem, and determines the likely solution domain. Results are stored in
``data/labeled_sample.csv``.
"""

import logging
import os
import time
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- PROMPTS ---
PROBLEM_PROMPT = """
You are a classifier that determines if a Reddit post describes a problem.

A "problem" means:
- The author is frustrated by something.
- They are asking for a better solution.
- They mention something that doesn't work well, is missing, or could be improved.

A post is NOT a problem if:
- It is just sharing an opinion or fact.
- It is promoting or showcasing something.
- It is asking a question without any frustration or need.

Return ONLY one of:
- 1 if it clearly describes a problem, pain point, or unmet need.
- 0 if it does not.

Post:
{post_text}
"""

SOLUTION_PROMPT = """
You are a classifier that categorizes the likely type of solution needed for the following post.
Classify based only on the content of the post and its implied solution type.

Definitions:
- not_applicable: The post is not a real problem, pain point, or unmet need.
- software: The problem can be solved entirely with software — apps, websites, APIs, algorithms, or automation — without requiring new physical infrastructure or external coordination.
- software_external: The solution is primarily software but must coordinate with or interact with existing external systems such as humans, vehicles, logistics networks, or third-party services.
- software_hardware: The solution requires new or specialized hardware in addition to software (e.g., IoT devices, robotics, sensors, smart devices).
- hardware: The solution is primarily a new physical or mechanical invention. Software alone cannot solve it.
- external: The solution is mainly about deploying, organizing, or coordinating existing physical resources, infrastructure, people, or logistics — and does not fundamentally require new software or hardware to be built.
- unknown: Use this **only as a last resort** if the post clearly describes a problem but you cannot confidently assign it to any of the above solution types.

Instructions:
1. If the post is NOT a problem, return `not_applicable`.
2. If the post IS a problem, you must choose one of the five solution categories: `software`, `software_external`, `software_hardware`, `hardware`, or `external`.
3. Only use `unknown` if the post clearly describes a problem but there is insufficient context or ambiguity that prevents a confident classification.
4. Never return `not_applicable` for a problem post.

Post:
{post_text}

Return ONLY one of:
not_applicable, software, software_external, software_hardware, hardware, external, unknown
"""




def _call_with_retry(prompt: str, *, max_attempts: int = 3, initial_delay: float = 1.0) -> Optional[str]:
    """Call the Responses API with retry logic."""
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.responses.create(
                model="gpt-4o",
                input=prompt
            )
            return response.output_text.strip()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Attempt %d failed: %s", attempt, exc)
            if attempt == max_attempts:
                return None
            time.sleep(delay)
            delay *= 2
    return None


def main() -> None:
    logger.info("Loading raw data…")
    df = pd.read_csv("data/raw_data.csv").fillna("")
    sample_df = df.head(100).copy()

    if sample_df.empty:
        logger.warning("No data found in raw dataset. Exiting.")
        return

    results = []
    total = len(sample_df)
    logger.info("Processing %d posts.", total)

    for index, row in sample_df.iterrows():
        position = len(results) + 1
        logger.info("Processing post %d/%d (index %s)…", position, total, index)
        post_text = f"Title: {row.get('title', '')}\n\nBody: {row.get('body', '')}"

        problem_prompt = PROBLEM_PROMPT.format(post_text=post_text)
        problem_result = _call_with_retry(problem_prompt)

        if problem_result is None:
            logger.error("Problem classification failed after retries.")
            is_problem = "error"
        else:
            is_problem = problem_result

        solution_domain = "-"
        if is_problem == "1":
            solution_prompt = SOLUTION_PROMPT.format(post_text=post_text)
            solution_result = _call_with_retry(solution_prompt)
            if solution_result is None:
                logger.error("Solution classification failed after retries.")
            else:
                solution_domain = solution_result

        results.append(
            {
                "title": row.get("title", ""),
                "body": row.get("body", ""),
                "is_problem": is_problem,
                "solution_domain": solution_domain,
            }
        )

        time.sleep(0.5)

    output_df = pd.DataFrame(results)
    output_path = "data/labeled_sample.csv"
    output_df.to_csv(output_path, index=False)
    logger.info("Classification complete. Results saved to %s", output_path)


if __name__ == "__main__":
    main()

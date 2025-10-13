from __future__ import annotations

import json

import logging
import os
import time
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

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

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------
# Enhanced Prompt Templates with Examples and Decision Rules
# -----------------------------
PROBLEM_PROMPT = """You are a classifier that determines if a Reddit post describes a problem.

A "problem" means:
- The author is frustrated by something, or
- They ask for a better solution, or
- They mention something that doesn't work well, is missing, or could be improved.

DECISION RULES:
- If a post shares advice/workflow but references a real problem others face → PROBLEM (1)
- If a post only describes a past problem that is already resolved → NOT PROBLEM (0)
- If a post has no problem context at all (pure showcase, news, opinion) → NOT PROBLEM (0)

EXAMPLES:
✅ PROBLEM (1):
- "Why is the job market so terrible for entry-level developers? Here's my advice..."
- "Looking for affordable architecture software since I don't have space to draw"
- "My Samsung phone became harder to use after the latest update"

❌ NOT PROBLEM (0):
- "Just got a random package in the mail, turns out it was from a friend!"
- "Check out this cool project I built" (pure showcase)
- "Latest industry news about AI developments" (pure news)

⚠️ EDGE CASES:
- Posts combining problems + solutions: Focus on whether they reference genuine frustrations
- Resolved problems: Only label as problem if seeking ongoing solutions

Return ONLY one of:
- 1  (if it clearly describes a problem, pain point, or unmet need)
- 0  (if it does not)

Post:
{post_text}
"""

SOFTWARE_SOLVABLE_PROMPT = """You are a classifier that determines if a problem could be solved primarily with software.

"Software-solvable" means:
- The core solution could be built as software (apps, websites, APIs, automation, algorithms)
- Does NOT require physical coordination, manufacturing, or institutional processes
- Does NOT include learning/knowledge problems that need guidance rather than tools

DECISION RULES:
- Hardware symptoms with software causes → SOFTWARE SOLVABLE (1)
- Information/learning needs → NOT SOFTWARE SOLVABLE (0)
- Guidance-seeking without technical malfunction → NOT SOFTWARE SOLVABLE (0)
- Existing tool requests without describing software gaps → NOT SOFTWARE SOLVABLE (0)

EXAMPLES:
✅ SOFTWARE SOLVABLE (1):
- "My display is blurry after driver update" (software cause)
- "Need automation for repetitive data entry tasks"
- "App crashes when I try to export files"

❌ NOT SOFTWARE SOLVABLE (0):
- "How do I learn to make waterfall charts?" (learning problem)
- "Need guidance on career direction" (advice needed)
- "Looking for existing tools to solve X" (market solution)

⚠️ EDGE CASES:
- Hardware problems: Check if root cause is software-related
- Tutorial requests: These are learning problems, not software problems
- Rants about software: Only solvable if describing specific malfunctions

Return ONLY one of:
- 1  (if primarily solvable by software)
- 0  (if not primarily solvable by software)

Post (this post IS a problem):
{post_text}
"""

EXTERNAL_REQUIRED_PROMPT = """You are a classifier that determines if solving a problem requires external action beyond the user's control.

"External" means the solution depends on:
- Manufacturer/company intervention (warranty, firmware, customer service)
- Physical-world coordination (technicians, logistics, hardware deployment)
- Institutional processes (government, insurance, regulations)
- Purchasing existing market products/services

"Not external" means the user can solve it through:
- Personal settings/configuration changes
- Software they can install or develop themselves
- Individual learning or practice

DECISION RULES:
- Seeking existing products/alternatives → EXTERNAL (1)
- Hardware failures needing replacement → EXTERNAL (1)
- Career/education guidance → EXTERNAL (1)
- Manufacturer-specific firmware issues → EXTERNAL (1)
- Problems solvable by user configuration → NOT EXTERNAL (0)

EXAMPLES:
✅ EXTERNAL (1):
- "Which 3D printer should I buy?" (market product)
- "My GoXLR is broken, need alternatives" (replacement needed)
- "Should I stay in SDET or switch careers?" (external opportunities)

❌ NOT EXTERNAL (0):
- "How do I configure my development environment?"
- "Need to automate my personal workflow"
- "Looking for programming help with my project"

⚠️ EDGE CASES:
- Hardware + software problems: If solution involves buying hardware → EXTERNAL
- Prototype projects: If requires manufacturing/deployment → EXTERNAL
- Learning needs: If requires structured education → EXTERNAL

Return ONLY one of:
- 1  (if external action or coordination is required to solve)
- 0  (if the user can fully resolve it themselves)

Post (this post IS a problem):
{post_text}
"""

# Enhanced reasoning prompt with structured output
FULL_REASON_PROMPT = """You are a precise classifier for Reddit posts. 
Your task is to assign three binary labels and provide brief reasoning for each decision.

Label definitions:
1. is_problem: 1 if the post describes or implies a problem, difficulty, or question seeking a solution. Otherwise 0.
2. is_software_solvable: 1 if the main problem can be solved primarily through software, code, configuration, or digital settings. Only 1 if is_problem == 1.
3. is_external: 1 if solving the problem requires action or coordination beyond the user's control (e.g., manufacturer, company, government, or physical service). Only 1 if is_problem == 1.

Rules:
- If is_problem = 0, both is_software_solvable and is_external must be 0.
- Base decisions on how the problem can be solved, not who caused it.
- Consider the primary intent: seeking help vs. sharing advice vs. showcasing

Return your response as valid JSON in this exact format:
{{
  "is_problem": "0 or 1",
  "is_software_solvable": "0 or 1", 
  "is_external": "0 or 1",
  "problem_reason": "Brief explanation for is_problem decision",
  "software_reason": "Brief explanation for is_software_solvable decision",
  "external_reason": "Brief explanation for is_external decision"
}}

Post:
{post_text}
"""

# Post intent classifier to handle edge cases
POST_INTENT_PROMPT = """Classify the PRIMARY intent of this Reddit post:

1 - SEEKING_HELP: User is asking for help, solutions, or recommendations
2 - SHARING_ADVICE: User is sharing their own solution, workflow, or advice 
3 - SHOWCASING: User is demonstrating/promoting something they built
4 - DISCUSSING: User is sharing opinions, news, or general discussion

Focus on the main purpose, not secondary elements.

Return ONLY the number (1, 2, 3, or 4).

Post:
{post_text}
"""

# -----------------------------
# Types
# -----------------------------
Label = Literal["0", "1", "error"]
PostIntent = Literal["1", "2", "3", "4", "error"]


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
        max_attempts: int = 5,
        initial_delay: float = 2.0,
) -> Optional[str]:
    """Call the OpenAI API with retry logic; return text or None."""
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000
            )
            text = response.choices[0].message.content
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


def _normalize_intent_token(raw: Optional[str]) -> PostIntent:
    """Normalize intent classification output."""
    if raw is None:
        return "error"
    token = (raw or "").strip().splitlines()[0].strip().split()[0].strip(".,:;\"'`")
    if token in {"1", "2", "3", "4"}:
        return token  # type: ignore[return-value]
    logger.warning("Unparsable intent output: %r -> 'error'", raw)
    return "error"


def _decode_raw_json(raw: Optional[str]) -> dict:
    """Parse JSON response with error handling."""
    if raw is None:
        return {'error': 'No response received'}
    try:
        # Remove Markdown code fences if present
        cleaned = raw.strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        # Or more robust:
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON response: %r", raw)
        return {'error': raw}



# -----------------------------
# Enhanced Classifiers with Intent Detection
# -----------------------------
def classify_post_intent(client: OpenAI, post_text: str) -> PostIntent:
    """Classify the primary intent of the post."""
    raw = _call_with_retry(client, POST_INTENT_PROMPT.format(post_text=post_text))
    return _normalize_intent_token(raw)


def classify_is_problem(client: OpenAI, post_text: str, intent: PostIntent = None) -> Label:
    """Return '1' if post is a problem, '0' if not, 'error' on failure."""
    raw = _call_with_retry(client, PROBLEM_PROMPT.format(post_text=post_text))
    result = _normalize_binary_token(raw)

    # Apply intent-based validation
    if intent == "3":  # SHOWCASING - rarely problems unless seeking feedback
        if result == "1":
            logger.info("Intent override: Showcase post marked as non-problem")
            return "0"

    return result


def classify_is_software_solvable(client: OpenAI, post_text: str) -> Label:
    """Return '1' if primarily solvable by software, '0' if not, 'error' on failure."""
    raw = _call_with_retry(client, SOFTWARE_SOLVABLE_PROMPT.format(post_text=post_text))
    return _normalize_binary_token(raw)


def classify_is_external(client: OpenAI, post_text: str) -> Label:
    """Return '1' if external coordination/resources required, '0' if not, 'error' on failure."""
    raw = _call_with_retry(client, EXTERNAL_REQUIRED_PROMPT.format(post_text=post_text))
    return _normalize_binary_token(raw)


def describe_rationale(client: OpenAI, post_text: str) -> dict[str, str]:
    """Get comprehensive classification with reasoning."""
    print(FULL_REASON_PROMPT)
    raw = _call_with_retry(client, FULL_REASON_PROMPT.format(post_text=post_text))
    result = _decode_raw_json(raw)

    # Ensure all required keys exist
    required_keys = ["is_problem", "is_software_solvable", "is_external",
                     "problem_reason", "software_reason", "external_reason"]
    for key in required_keys:
        if key not in result:
            result[key] = "error"

    return result


# -----------------------------
# Validation Functions
# -----------------------------
def validate_labels(is_problem: Label, is_software: Label, is_external: Label) -> tuple[Label, Label, Label]:
    """Validate and correct label combinations according to rules."""
    if is_problem == "0":
        return is_problem, "0", "0"
    elif is_problem == "error":
        return is_problem, "error", "error"
    else:
        return is_problem, is_software, is_external


def detect_edge_cases(post_text: str, labels: tuple[Label, Label, Label]) -> dict:
    """Detect potential edge cases based on content patterns."""
    text_lower = post_text.lower()
    edge_cases = []

    # Common edge case patterns
    if "here's my advice" in text_lower or "here's what i learned" in text_lower:
        edge_cases.append("advice_sharing")

    if "randomly got" in text_lower or "turns out it was" in text_lower:
        edge_cases.append("resolved_problem")

    if "looking for" in text_lower and ("software" in text_lower or "tool" in text_lower):
        edge_cases.append("seeking_existing_solution")

    if "how do i" in text_lower or "how to" in text_lower:
        edge_cases.append("learning_question")

    return {
        "detected_patterns": edge_cases,
        "confidence": "high" if not edge_cases else "medium"
    }


# -----------------------------
# Enhanced Processing
# -----------------------------
def process_posts(client: OpenAI, df: pd.DataFrame, sleep_seconds: float = 1.5) -> pd.DataFrame:
    """
    Enhanced processing pipeline with intent detection and validation.

    Pipeline:
    1. Classify post intent
    2. Run is_problem classification with intent context
    3. If problem, classify software_solvable and external
    4. Validate label combinations
    5. Detect edge cases for quality control
    """
    results: list[dict[str, str]] = []
    total = len(df)
    logger.info("Processing %d posts with enhanced pipeline…", total)

    for idx, row in df.iterrows():
        position = len(results) + 1
        logger.info("Processing post %d/%d (index %s)…", position, total, idx)

        post_text = format_post(row)

        # Step 1: Classify intent
        intent = classify_post_intent(client, post_text)

        # Step 2: Classify problem with intent context
        is_problem = classify_is_problem(client, post_text, intent)

        # Step 3: Conditional classification
        if is_problem == "1":
            is_software = classify_is_software_solvable(client, post_text)
            is_external = classify_is_external(client, post_text)
            rationale = describe_rationale(client, post_text)
        elif is_problem == "0":
            is_software = "0"
            is_external = "0"
            rationale = {"reasoning": "Not a problem - no further classification needed"}
        else:  # "error"
            is_software = "error"
            is_external = "error"
            rationale = {"error": "Classification failed"}

        # Step 4: Validate label combinations
        is_problem, is_software, is_external = validate_labels(is_problem, is_software, is_external)

        # Step 5: Edge case detection
        edge_case_info = detect_edge_cases(post_text, (is_problem, is_software, is_external))

        results.append(
            {
                "title": (row.get("title", "") or ""),
                "body": (row.get("body", "") or ""),
                "intent": intent,
                "is_problem": is_problem,
                "is_software_solvable": is_software,
                "is_external": is_external,
                "rationale": json.dumps(rationale) if isinstance(rationale, dict) else str(rationale),
                "edge_cases": json.dumps(edge_case_info),
                "confidence": edge_case_info.get("confidence", "medium")
            }
        )

        time.sleep(sleep_seconds)

    # Ensure desired column order
    out_df = pd.DataFrame(results)[
        ["title", "body", "intent", "is_problem", "is_software_solvable",
         "is_external", "rationale", "edge_cases", "confidence"]
    ]
    return out_df


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    """Entry point: load data, classify, and save results."""
    client = init_client()
    sample_df = load_data("data/raw_data.csv", sample_size=30)
    if sample_df.empty:
        logger.warning("No data to process. Exiting.")
        return
    labeled_df = process_posts(client, sample_df, sleep_seconds=0.5)
    save_results(labeled_df, "data/labeled_sample.csv")


if __name__ == "__main__":
    main()
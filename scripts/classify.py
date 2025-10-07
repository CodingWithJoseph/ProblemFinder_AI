# scripts/classify.py

from dotenv import load_dotenv
import os
import pandas as pd
from openai import OpenAI
import time

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
You are a classifier that categorizes the likely type of solution needed for the following problem post.

Definitions:
- software_only: The problem can be solved with software (apps, websites, APIs, algorithms, automation, etc.) and does not require new hardware.
- software+hardware: The problem requires a combination of software and physical hardware (e.g., IoT devices, robotics, smart devices).
- hardware_only: The solution is primarily physical or mechanical and software alone would not solve it.

Choose the category that best matches the post's described problem.

Post:
{post_text}

Return ONLY one of: software_only, software+hardware, hardware_only
"""

# --- READ RAW DATA ---
df = pd.read_csv("data/raw_data.csv")

results = []

for idx, row in df.iterrows():
    # Combine title and body to give GPT more context
    post_text = f"Title: {row['title']}\n\nBody: {row['body']}"
    print(f"Processing post {idx+1}/{len(df)}...")

    # --- Stage 1: Problem classification ---
    try:
        problem_response = client.chat.completions.create(
            model="gpt-4.1",  # or gpt-4o if you have access
            messages=[{"role": "user", "content": PROBLEM_PROMPT.format(post_text=post_text)}]
        )
        is_problem = problem_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error on problem classification: {e}")
        is_problem = "error"

    # --- Stage 2: Solution domain (only if problem == 1) ---
    solution_domain = "-"
    if is_problem == "1":
        try:
            solution_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": SOLUTION_PROMPT.format(post_text=post_text)}]
            )
            solution_domain = solution_response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error on solution classification: {e}")

    results.append({
        "title": row["title"],
        "body": row["body"],
        "is_problem": is_problem,
        "solution_domain": solution_domain
    })

    # Sleep to avoid hitting rate limits
    time.sleep(0.5)

# --- SAVE OUTPUT ---
output_df = pd.DataFrame(results)
output_df.to_csv("data/labeled_data.csv", index=False)
print("✅ Classification complete! Results saved to data/labeled_data.csv")

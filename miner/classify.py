# miner/classify.py

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

"""
1. Parses the command line arguments using Python’s `argparse` library, validating and converting them into a structured Namespace object.
2. Builds a unified runtime configuration (`RunConfig`) via `build_run_config`, merging `CLI` overrides with defaults from `config.yaml`.
3. Initializes the runtime environment, including:
   - Logging configuration and verbosity level.
   - The `ResponseCache` for caching LLM responses.
   - A `RateLimiter` to control OpenAI API request speed.
   - The `OpenAI` API client for model access (e.g., `gpt-4o`).
4. Loads the raw dataset into memory using `load_dataframe`, preparing it for processing.
5. Optionally loads previous run reports, enabling tracking of evaluation metrics and prompt drift across sessions.
6. Executes the main pipeline via `run_pipeline()`, which handles:
   - Deduplication (detecting and grouping near-identical posts).
   - Canonical selection (choosing one representative post per duplicate cluster).
   - Classification (rule-based, LLM-based, or ensemble voting).
   - Optional evaluation against a gold standard.
   - Generation of structured reports and summaries.
7. Saves the outputs:
   - A labeled canonical dataset (`.csv`).
   - A configuration snapshot (`.config.json`).
   - Duplicate-to-canonical mappings (`.mapping.json`).
   - Optional evaluation and drift reports (`.json`).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from openai import OpenAI

from problemfinder.core.cache import ResponseCache
from problemfinder.core.configuration import build_run_config
from problemfinder.core.pipeline import run_pipeline
from problemfinder.utils.io import load_dataframe, save_dataframe
from problemfinder.utils.logging import structured_log
from problemfinder.utils.rate_limit import RateLimiter


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ProblemFinder classification pipeline")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to configuration YAML")
    parser.add_argument("--input", type=Path, default=Path("C:/Users/josep/Projects/ProblemFinder_AI/data/raw_data.csv"), help="Path to the raw Reddit CSV file")
    parser.add_argument("--output", type=Path, default=Path("C:/Users/josep/Projects/ProblemFinder_AI/data/labeled_sample.csv"), help="Output path for labeled CSV")
    parser.add_argument("--dedupe", choices=["on", "off"], default="on", help="Enable or disable deduplication")
    parser.add_argument("--similarity-threshold", type=float, default=0.5, help="Duplicate similarity threshold")
    parser.add_argument("--soft-similarity-threshold", type=float, default=0.35, help="Soft duplicate similarity threshold")
    parser.add_argument("--canonical-policy", choices=["earliest", "longest"], default="earliest")
    parser.add_argument("--dedupe-report", type=Path, default=None, help="Optional CSV report for duplicate clusters")
    parser.add_argument("--no-split", action="store_true", help="Disable train/val/test splitting")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--ensemble", choices=["on", "off"], default="off", help="Enable ensemble voting")
    parser.add_argument("--ensemble-members", default="direct,reasoning,rules", help="Comma separated ensemble members")
    parser.add_argument(
        "--ensemble-disagreement-threshold", type=float, default=0.3, help="Threshold for high disagreement logging"
    )
    parser.add_argument("--model", choices=["gpt-4o", "gpt-4o-mini"], default="gpt-4o", help="Model selection")
    parser.add_argument("--temperature", type=float, default=None, help="Override model temperature")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--cache", choices=["on", "off"], default=None, help="Enable or disable response cache")
    parser.add_argument("--cache-ttl", type=int, default=None, help="Cache TTL in hours")
    parser.add_argument("--cache-path", type=Path, default=None, help="Path for cache persistence")
    parser.add_argument("--max-workers", type=int, default=None, help="Number of worker threads")
    parser.add_argument("--rate-limit", type=int, default=None, help="Rate limit in requests per minute")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size for batch processing")
    parser.add_argument("--evaluation", choices=["on", "off"], default=None, help="Enable evaluation mode")
    parser.add_argument("--evaluation-gold-set", type=Path, default=None, help="Path to gold set CSV")
    parser.add_argument("--report-path", type=Path, default=None, help="Path to summary report JSON")
    parser.add_argument("--resume", action="store_true", help="Resume from existing labeled output")
    parser.add_argument("--resume-from", type=Path, default=None, help="Path to existing labeled CSV for resume")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "DEBUG"))
    return parser.parse_args(args)


def main(argv: Optional[Sequence[str]] = None) -> None:
    # Build a dictionary like object from command line arguments.
    args = parse_args(argv)

    # Initialize logging: Ensures a consistent log format and level across the pipeline.
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load the input data and build the run configuration.
    df = load_dataframe(args.input)
    run_config = build_run_config(args)

    # Ensures resume mode has a valid checkpoint path.
    if run_config.resume.enabled and not run_config.resume.checkpoint_path:
        run_config.resume.checkpoint_path = args.output

    # ResponseCache — manages cached API responses to avoid re-calling GPT for identical prompts.
    cache = ResponseCache(run_config.cache)
    # RateLimiter — throttles API requests per minute to respect OpenAI rate caps.
    rate_limiter = RateLimiter(run_config.parallel.rate_limit)
    client = OpenAI()

    # Load historical reports for drift comparison.
    historical_payloads: list[Mapping[str, Any]] = []
    if run_config.report.path and run_config.report.path.exists():
        try:
            with run_config.report.path.open("r", encoding="utf-8") as handle:
                historical_payloads.append(json.load(handle))
        except Exception:  # pragma: no cover - defensive
            logging.getLogger(__name__).exception("Failed to load historical report %s", run_config.report.path)

    # Run the full labeling pipeline and save the results.
    results = run_pipeline(
        df=df,
        run_config=run_config,
        cache=cache,
        rate_limiter=rate_limiter,
        historical_payloads=historical_payloads,
        client=client,
    )

    # Writes the final labeled dataset to the specified output CSV.
    canonical_df = results["canonical_df"]
    save_dataframe(canonical_df, args.output)

    # .config.json — stores the full configuration used for this run.
    config_output_path = args.output.with_suffix(".config.json")
    with config_output_path.open("w", encoding="utf-8") as handle:
        json.dump(results["config_payload"], handle, ensure_ascii=False, indent=2)

    # .mapping.json — maps original post-IDs to their canonical counterparts.
    mapping_path = args.output.with_suffix(".mapping.json")
    mapping_path.write_text(json.dumps(results["id_mapping"], indent=2), encoding="utf-8")

    #
    structured_payload = {
        "output": str(args.output),
        "canonical_rows": len(canonical_df),
        "clusters": len(results["clusters"]),
        "duplicates_removed": results["summary_payload"].get("duplicate_stats", {}).get("duplicates_removed", 0),
        "report": str(run_config.report.path) if run_config.report.path else None,
    }
    structured_log(logging.INFO, event="run_complete", **structured_payload)

    # Dead Letter Queue — posts that failed to classify.
    if results["summary_payload"].get("dead_letter_queue"):
        structured_log(
            logging.WARNING,
            event="dead_letter_queue",
            count=len(results["summary_payload"]["dead_letter_queue"]),
        )

    # Prompt Drift — compares current run against historical baselines.
    if results["summary_payload"].get("historical_drift"):
        structured_log(
            logging.INFO,
            event="prompt_drift",
            payload=results["summary_payload"]["historical_drift"],
        )

    # Evaluation Metrics — accuracy/F1 metrics if evaluation mode enabled.
    if results["summary_payload"].get("evaluation_metrics"):
        structured_log(
            logging.INFO,
            event="evaluation_metrics",
            payload=results["summary_payload"]["evaluation_metrics"],
        )

    logging.getLogger(__name__).info("Saved labeled dataset to %s", args.output)
    if run_config.report.path:
        logging.getLogger(__name__).info("Summary report available at %s", run_config.report.path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

"""Command-line interface for running the ProblemFinder pipeline."""

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
    parser.add_argument("--input", type=Path, default=Path("data/raw_data.csv"), help="Path to the raw Reddit CSV file")
    parser.add_argument("--output", type=Path, default=Path("data/labeled_sample.csv"), help="Output path for labeled CSV")
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
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser.parse_args(args)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    df = load_dataframe(args.input)
    run_config = build_run_config(args)
    if run_config.resume.enabled and not run_config.resume.checkpoint_path:
        run_config.resume.checkpoint_path = args.output

    cache = ResponseCache(run_config.cache)
    rate_limiter = RateLimiter(run_config.parallel.rate_limit)
    client = OpenAI()

    historical_payloads: list[Mapping[str, Any]] = []
    if run_config.report.path and run_config.report.path.exists():
        try:
            with run_config.report.path.open("r", encoding="utf-8") as handle:
                historical_payloads.append(json.load(handle))
        except Exception:  # pragma: no cover - defensive
            logging.getLogger(__name__).exception("Failed to load historical report %s", run_config.report.path)

    results = run_pipeline(
        df=df,
        run_config=run_config,
        cache=cache,
        rate_limiter=rate_limiter,
        historical_payloads=historical_payloads,
        client=client,
    )

    canonical_df = results["canonical_df"]
    save_dataframe(canonical_df, args.output)

    config_output_path = args.output.with_suffix(".config.json")
    with config_output_path.open("w", encoding="utf-8") as handle:
        json.dump(results["config_payload"], handle, ensure_ascii=False, indent=2)

    mapping_path = args.output.with_suffix(".mapping.json")
    mapping_path.write_text(json.dumps(results["id_mapping"], indent=2), encoding="utf-8")

    structured_payload = {
        "output": str(args.output),
        "canonical_rows": len(canonical_df),
        "clusters": len(results["clusters"]),
        "duplicates_removed": results["summary_payload"].get("duplicate_stats", {}).get("duplicates_removed", 0),
        "report": str(run_config.report.path) if run_config.report.path else None,
    }
    structured_log(logging.INFO, event="run_complete", **structured_payload)

    if results["summary_payload"].get("dead_letter_queue"):
        structured_log(
            logging.WARNING,
            event="dead_letter_queue",
            count=len(results["summary_payload"]["dead_letter_queue"]),
        )

    if results["summary_payload"].get("historical_drift"):
        structured_log(
            logging.INFO,
            event="prompt_drift",
            payload=results["summary_payload"]["historical_drift"],
        )

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

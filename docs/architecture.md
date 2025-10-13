# ProblemFinder Architecture

## System Overview
ProblemFinder is a modular pipeline that ingests Reddit exports, deduplicates related posts, classifies each canonical entry with rule and LLM-based heuristics, and emits analytics-ready reports. The project follows a layered package structure under `src/problemfinder` that separates configuration, core pipeline logic, classification utilities, reporting, and reusable helpers.

```
src/problemfinder/
├── cli.py                    # CLI entry point for orchestration
├── core/                     # Dedupe, pipeline, configuration, cache, fetching
├── classification/           # Rule engine, ensemble logic, OpenAI interface
├── reporting/                # Evaluation metrics and reporting helpers
└── utils/                    # Cross-cutting helpers (I/O, logging, rate limiting, text)
```

## Module Responsibilities & Dependencies
- **`problemfinder.cli`** – Parses CLI arguments, loads configs, instantiates shared services, and invokes `core.pipeline.run_pipeline`.
- **`problemfinder.core.configuration`** – Normalises CLI flags with YAML config defaults to produce a `RunConfig` dataclass consumed by the pipeline.
- **`problemfinder.core.pipeline`** – Coordinates deduplication, optional resume support, threaded classification, evaluation, and report generation. Depends on `core.dedupe`, `classification.rules`, `classification.ensemble`, `classification.llm_interface`, `reporting.*`, and utility packages.
- **`problemfinder.core.dedupe`** – Provides canonical text normalisation, clustering heuristics, and CSV reporting for duplicate groups. Relies on `utils.text` for cleaning routines.
- **`problemfinder.classification.rules`** – Implements the deterministic Version 2 rule engine and JSON schema used for validation.
- **`problemfinder.classification.ensemble`** – Aggregates rule, direct, and reasoning members with configurable voting and disagreement tracking.
- **`problemfinder.classification.llm_interface`** – Wraps the OpenAI Chat Completions API, handles caching, retry-safe parsing, and produces ensemble-compatible member callables.
- **`problemfinder.reporting.evaluation`** – Calculates accuracy, macro-F1, and drift summaries against historical metrics.
- **`problemfinder.reporting.summary`** – Builds the JSON report persisted by the CLI.
- **`problemfinder.utils`** – Houses shared helpers for structured logging, I/O, rate limiting, text normalisation, and PII scrubbing.

The dependency graph flows in one direction—from high-level orchestration (`cli` and `core.pipeline`) down into increasingly specialised modules—ensuring packages remain acyclic and testable.

## Data Flow
1. **Input**: The CLI reads a Reddit CSV export (from `fetch.py` or external collection).
2. **Deduplication**: `core.dedupe.normalise_post` canonicalises text and metadata, `cluster_duplicates` groups near-duplicates, and `deduplicate_dataframe` selects canonical rows.
3. **Classification**:
   - Resume logic restores previously processed rows when available.
   - `classification.rules.Version2RuleEngine` provides baseline labels.
   - Optional OpenAI members (`direct` and `reasoning`) run through `classification.llm_interface` with caching and rate limiting.
   - `classification.ensemble` combines member payloads and records disagreement statistics.
4. **Splitting**: `assign_splits` guarantees entire duplicate clusters share the same dataset split.
5. **Evaluation & Reporting**: `reporting.evaluation` compares predictions against an optional gold set and historical benchmarks. `reporting.summary` produces a JSON artifact capturing label distributions, performance, cache metrics, and configuration snapshots.
6. **Outputs**: The CLI writes the canonical CSV, config snapshot, ID mapping, and optional summary report. Structured logs surface runtime stats suitable for ingestion into logging pipelines.

## Configuration Strategy
- The CLI accepts comprehensive flags for runtime overrides.
- `core.configuration.build_run_config` merges YAML defaults with CLI inputs and produces a strongly typed `RunConfig` dataclass used throughout the pipeline.
- Complex sections (ensemble, cache, evaluation, reporting, parallelism) are represented as nested dataclasses to simplify validation and testing.
- Normalised configs are persisted alongside outputs (`*.config.json`) for reproducibility and audit trails.

## Future Model Roadmap
The current pipeline couples deterministic rules with optional GPT-4o reasoning members. Planned enhancements include:
- **DeBERTa Fine-Tuning**: Introduce a dedicated `classification.models` subpackage housing a DeBERTa-v3 based classifier fine-tuned on the labeled corpus. The model would integrate via the ensemble layer with deterministic fallbacks to maintain reliability.
- **Calibration & Drift Monitoring**: Add probability calibration for transformer outputs, integrate drift detection dashboards into the reporting module, and surface alerts via structured logs.
- **Extended Channel Support**: Expand the fetch layer to ingest additional social media sources (Twitter, LinkedIn groups) while reusing the same dedupe and classification stack.

# Refactor Summary

## New Package Layout
- `src/problemfinder/cli.py` – replaces ad-hoc logic in `scripts/classify.py` for orchestrating the pipeline.
- `src/problemfinder/core/` – hosts configuration loading, pipeline orchestration, deduplication, caching, and Reddit fetching utilities.
- `src/problemfinder/classification/` – contains the Version 2 rule engine, ensemble logic, and OpenAI interface extracted from `scripts/classify.py` and `scripts/ensemble.py`.
- `src/problemfinder/reporting/` – consolidates evaluation (`scripts/evaluation.py`) and reporting (`scripts/report.py`).
- `src/problemfinder/utils/` – centralises shared helpers for logging, concurrency, rate limiting, text normalisation, PII redaction, and I/O.

## Retired / Replaced Scripts
- `scripts/classify.py` → split across `core.pipeline`, `core.dedupe`, `classification.rules`, `classification.ensemble`, `classification.llm_interface`, and `cli.py`.
- `scripts/cache.py` → `core.cache`.
- `scripts/ensemble.py` → `classification.ensemble`.
- `scripts/evaluation.py` → `reporting.evaluation`.
- `scripts/report.py` → `reporting.summary`.
- `scripts/fetch.py` → `core.fetch` with environment management via `settings.py`.

## Additional Deliverables
- `docs/architecture.md` describing system boundaries, dependencies, data flow, and the future DeBERTa plan.
- Updated `README.md` and new `CONTRIBUTING.md` for developer onboarding.
- `notebooks/pipeline_demo.ipynb` demonstrating programmatic pipeline execution.
- Expanded unit test scaffolding now referencing the modular package structure.

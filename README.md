# ProblemFinder

ProblemFinder is an AI-assisted pipeline for discovering software-solvable problems in Reddit content. The system collects posts, deduplicates near-identical submissions, labels each canonical example with intent/problem/external dimensions, and produces analytics-ready reports for product research teams.

## Key Capabilities
- Deterministic duplicate clustering that respects URLs, text similarity, and cross-subreddit heuristics.
- Hybrid classification engine blending a rule-based model with optional OpenAI GPT-4o reasoning members and caching.
- Configurable evaluation against gold labels with drift detection across historical runs.
- Structured JSON summary reports and CSV exports suitable for downstream analytics.

## Project Structure
```
src/
├── main.py                   # Thin launcher delegating to problemfinder.cli
└── problemfinder/
    ├── cli.py                # CLI entry point
    ├── core/                 # Pipeline, configuration, dedupe, cache, fetch utilities
    ├── classification/       # Rule engine, ensemble logic, OpenAI interface
    ├── reporting/            # Evaluation + summary helpers
    └── utils/                # Logging, rate limiting, text normalisation, I/O
```

## Setup
1. **Create environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configure environment variables**
   Create a `.env` file (or export variables) with:
   ```bash
   REDDIT_CLIENT_ID=...
   REDDIT_CLIENT_SECRET=...
   REDDIT_USER_AGENT=ProblemFinder/1.0
   OPENAI_API_KEY=...
   ```
   The CLI automatically loads the `.env` file when `problemfinder.core.fetch` is used.
3. **Prepare input data**
   - Use `problemfinder.core.fetch.fetch_posts` or your own ingestion to build `data/raw_data.csv`.
   - Ensure the CSV contains at least `title`, `body`, `subreddit`, `url`, and `created_utc` columns.

## Running the Pipeline
```bash
python -m problemfinder.cli \
  --config config.yaml \
  --input data/raw_data.csv \
  --output data/labeled_sample.csv \
  --ensemble on \
  --report-path data/reports/latest.json
```

### Common CLI Flags
| Flag | Description |
| ---- | ----------- |
| `--dedupe on/off` | Enable duplicate clustering (default `on`). |
| `--similarity-threshold` | Minimum combined similarity for duplicates (default `0.5`). |
| `--soft-similarity-threshold` | Cross-subreddit soft duplicate threshold (default `0.35`). |
| `--ensemble on/off` | Toggle OpenAI ensemble members (default `off`). |
| `--ensemble-members` | Comma-separated members (`direct`, `reasoning`, `rules`). |
| `--cache on/off` | Force-enable or disable the response cache. |
| `--evaluation on/off` | Enable evaluation using `--evaluation-gold-set`. |
| `--resume` | Resume a partially labeled output CSV. |

### Configuration File (`config.yaml`)
The YAML file mirrors CLI flags and can provide defaults. Key sections include:
- `model`: `name`, `temperature`, `seed`
- `ensemble`: `enabled`, `members`, `disagreement_threshold`
- `cache`: `enabled`, `path`, `ttl_hours`
- `parallel`: `max_workers`, `rate_limit`, `chunk_size`
- `evaluation`: `enabled`, `gold_set_path`, `metrics`
- `report`: `path`
- `dedupe`: `similarity_threshold`, `soft_similarity_threshold`, `cross_subreddit`, `canonical_policy`
- `split`: `train_ratio`, `val_ratio`, `test_ratio`

CLI options override YAML defaults, and the final merged configuration is written to `<output>.config.json`.

## Development Workflow
1. Install development dependencies (`pip install -r requirements.txt`).
2. Format code using `black`/`ruff` conventions; all modules include type hints and Google-style docstrings.
3. Run unit tests: `pytest`.
4. Submit changes following the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).

## Example Notebook
See [`notebooks/pipeline_demo.ipynb`](notebooks/pipeline_demo.ipynb) for a step-by-step demonstration of running the dedupe and classification pipeline from Python.

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.

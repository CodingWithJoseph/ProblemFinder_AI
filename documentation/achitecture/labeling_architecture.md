# Phase 1: System Entry and Pipeline Bootstrapping

## 1.1 Entry Point: `Classify.py`

`classify.py:`

Acts purely as a compatibility wrapper for the command line interface. It just imports
and runs main from `cli.py` located in `/src/problemfinder`

## 1.2 CLI Entry and Pipeline Initialization `cli.py`
### `parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace`
Accepts an optional sequence of strings as input and returns an `argparse.Namespace` object.  
Used to parse command-line arguments that control the labeling pipeline.

#### **Arguments**

- **config** — Path to configuration YAML that defines default parameters (`config.yaml`).  
- **input** — Path to the raw dataset (`raw_data.csv`) to be labeled.  
- **output** — Output path for the labeled CSV.  
- **dedupe** — Enables or disables deduplication of near-identical posts before classification.  
- **similarity-threshold** — Controls how strictly duplicate posts are detected. Higher values are more conservative.  
- **soft-similarity-threshold** — Used for *fuzzy* (approximate, not completely identical) duplicate detection.  
- **canonical-policy** — Determines which duplicate post to keep as canonical within each cluster (`earliest` or `longest`).  
- **dedupe-report** — Optional CSV output listing duplicate clusters and their canonical post-mappings.  
- **no-split** — If present, disables automatic train/validation/test splitting.  
- **train-ratio** — Ratio of data assigned to the training split (only used when splitting is active).  
- **val-ratio** — Ratio of data assigned to the validation split.  
- **test-ratio** — Ratio of data assigned to the test split. Ratios must sum to 1.0 when splitting is enabled.  
- **ensemble** — Toggles the ensemble voting mechanism (combines rule-based + LLM predictions).  
- **ensemble-members** — Comma-separated list of ensemble participants (`direct`, `reasoning`, `rules`).  
- **ensemble-disagreement-threshold** — Logs high disagreement when ensemble members differ beyond this threshold.  
- **model** — Specifies which OpenAI model to use for labeling (`gpt-4o`, `gpt-4o-mini`, etc.).  
- **temperature** — Overrides the model’s temperature parameter (controls randomness in LLM output).  
- **seed** — Overrides the random seed for reproducibility.  
- **cache** — Enables or disables response caching (prevents repeated API calls for identical prompts).  
- **cache-ttl** — Cache time-to-live in hours; determines how long cached responses remain valid.  
- **cache-path** — Directory or file path where cache data is stored.  
- **max-workers** — Maximum number of worker threads for parallel classification or deduplication.  
- **rate-limit** — Rate limit (requests per minute) for API calls to avoid exceeding OpenAI rate caps.  
- **chunk-size** — Batch size for processing posts; affects throughput and memory footprint.  
- **evaluation** — Enables evaluation mode (requires a gold set) to compute metrics such as accuracy and F1.  
- **evaluation-gold-set** — Path to a CSV containing manually labeled “gold standard” examples.  
- **report-path** — JSON output path for the summary report (metrics, drift, and configuration info).  
- **resume** — Enables resume mode; continues from a previous labeled output to avoid reprocessing.  
- **resume-from** — Path to an existing labeled CSV to resume from (used with `--resume`).  
- **log-level** — Controls verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

### `main(argv: Optional[Sequence[str]] = None) -> None`
Serves as the primary runtime entry point for the labeling pipeline. It parses the command line arguments, builds a runtime
configuration, sets up core components (logging, caching, rate limiting, etc.), and executes the labeling pipeline.

##### **Steps**
1. **Parses** the command line arguments using Python’s `argparse` library, validating and converting them into a structured Namespace object.
2. **Builds** a unified runtime configuration (`RunConfig`) via `build_run_config`, merging `CLI` overrides with defaults from `config.yaml`.
3. **Initializes the runtime environment, including:**
   - Logging configuration and verbosity level.
   - The `ResponseCache` for caching LLM responses.
   - A `RateLimiter` to control OpenAI API request speed.
   - The `OpenAI` API client for model access (e.g., `gpt-4o`).
4. Loads the raw dataset into memory using `load_dataframe`, preparing it for processing.
5. **Optionally** loads previous run reports, enabling tracking of evaluation metrics and prompt drift across sessions.
6. **Executes the main pipeline via `run_pipeline()`, which handles:**
   - Deduplication (detecting and grouping near-identical posts).
   - Canonical selection (choosing one representative post per duplicate cluster).
   - Classification (rule-based, LLM-based, or ensemble voting).
   - Optional evaluation against a gold standard.
   - Generation of structured reports and summaries.
7. **Saves the outputs:**
   - A labeled canonical dataset (`.csv`). 
   - A configuration snapshot (`.config.json`). 
   - Duplicate-to-canonical mappings (`.mapping.json`). 
   - Optional evaluation and drift reports (`.json`).

Throughout execution, `cli.py` provides structured logging to track progress, errors, and key metrics such as:
- Number of canonical posts retained.
- Duplicate statistics and cluster counts.
- Ensemble disagreement rates.
- Evaluation results (if applicable).

---

# Phase 2: Configuration and Core Components
## 2.1 Configuration Layer: `core.configuration.py`
### `_load_yaml_config(path: Optional[Path]) -> Dict[str, Any]:`
Loads the default configuration from a YAML file and returns it as a dictionary with keys being the configuration 
sections and values being the configuration values.

### `build_run_config(args: argparse.Namespace) -> RunConfig:`
Accepts the Namespace containing the parsed command line argument built from `parse_args` and returns a `RunConfig` object.
RunConfig is a dataclass that contains the entire pipeline configuration state. RunConfig contains other configuration 
dataclasses in core/config.py as its attributes (2.1.1 Configuration Types).

##### **Steps**
1. **Loads** the default configuration from a YAML file using `_load_yaml_config`.
2. **Merges** the default configuration with the command line arguments.
3. **Validates** the configuration.
4. **Returns** the final configuration as a `RunConfig` object.

#### 2.1.1 Configuration Types:
- `RunConfig`: Master configuration object that contains all other configuration objects as its attributes.
- `ModelConfig`: Configuration for the OpenAI model. Defines the model name, temperature, and seed.
- `DedupeConfig`: Configuration for the deduplication stage. Defines the similarity threshold, canonical policy, report path, soft similarity threshold, and cross-subreddit setting.
- `SplitConfig`: Configuration for the train/validation/test split. Defines the split ratios and whether splitting is enabled.
- `ParallelConfig`: Configuration for parallel processing. Defines the maximum number of workers, rate limit, and chunk size.
- `ResumeConfig`: Configuration for resuming a partially completed labeling run. Defines whether resuming is enabled and the path to the checkpoint file.
- `EnsembleConfig`: Configuration for the ensemble voting mechanism. Defines whether ensemble voting is enabled, the members of the ensemble, and the disagreement threshold.
- `CacheConfig`: Configuration for the response cache. Defines whether caching is enabled, the path to the cache file, and the cache TTL.
- `ReportConfig`: Configuration for the summary report. Defines the path to the report file.
- `EvaluationConfig`: Configuration for evaluation against a gold set. Defines whether evaluation is enabled, the path to the gold set, and the metrics to evaluate.
 
---

## 2.2 Cache Layer: `core.cache.py`
Provides response caching for the labeling pipeline. Its purpose is to avoid redundant API calls to OpenAI by caching the responses
keyed on (model, prompt_version, text).

### `CacheConfig`: 
Configuration for the response cache. Defines whether caching is enabled, the path to the cache file, and the cache TTL.
    - `enabled`: Whether caching is enabled.
    - `path`: Path to the cache file.
    - `ttl_hours`: Cache time-to-live measured in hours.

### `ResponseCache`: 
Manages the cache state. Provides methods to get, set, and prune the cache.
    - `get`: Retrieves a cached response.
    - `set`: Stores a response in the cache.
    - `make_key`: Creates a cache key from the model, prompt version, and text.
    - `_load`: Loads the cache from disk.
    - `_persist`: Persists the cache to disk.
    - `_prune`: Prunes expired entries from the cache.

### `cached_api_call`: 
Wraps an API call with cache semantics. If the response is not in the cache, it calls the API and stores the response in the cache.
 - `model`: Name of the OpenAI model.
 - `prompt_version`: Version of the prompt.
 - `text`: Text to be classified.
 - `cache`: Instance of `ResponseCache`.
 - `fetch_fn`: Function that fetches the response from the API.
 - `cache_stats`: Optional dictionary to track cache hits and misses.

#### **Steps**
 1. **Creates a cache key** from the model, prompt version, and text and tries to retrieve the response from the cache (`cache.get`).
 2. **Checks if the response is in the cache**. If it is, returns the cached response.
    - If hit → return the cached result and log hit.
 3. **If the response is not in the cache**, calls the API to fetch the response.
    - If miss → call `fetch_fn()` (the real API), store a result via `cache.set()`, and return it.
 
Tracks hits/misses via an optional cache_stats dict for telemetry.

---

## 2.3 Pipeline Orchestration: `core.pipeline.py`
The `core.pipeline.py` module implements the **end-to-end orchestration** of the ProblemFinder labeling pipeline.  
It integrates deduplication, classification (rule-based and LLM-based), evaluation, and reporting into a single coordinated process.

The pipeline consumes a raw Reddit dataset and a `RunConfig`, applies all configured behaviors (deduplication, caching, rate-limiting, ensemble logic, etc.), and outputs a fully labeled canonical dataset along with detailed reports.

### **Key Functions**

#### `assign_splits(df: pd.DataFrame, cluster_members: Dict[str, List[str]], config: SplitConfig, seed: int = 42) -> pd.DataFrame`
- Divides posts into **train**, **validation**, and **test** splits while ensuring that **duplicate clusters stay together**.
- Uses ratios from `SplitConfig` (`train_ratio`, `val_ratio`, `test_ratio`).
- If splitting is disabled, marks all rows as `"unsplit"`.
- Ensures split ratios sum to 1.0 and applies random shuffling using a reproducible seed.

#### `_create_member_callables(...) -> Dict[str, Any]`
- Builds the **ensemble member callables** used for classification.
- Uses `OpenAIEnsembleFactory` to create model-based classifiers:
  - `"direct"` → standard LLM label prediction.
  - `"reasoning"` → reasoning-style labeling with explanations.
  - `"rules"` → placeholder for the deterministic rule engine.
- Returns a mapping of member names → callable classification functions.

#### `classify_dataframe(...) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, List[str]], Dict[str, Any]]`
The **core classification engine** for the pipeline.

**Purpose:**  
Deduplicate the input, classify each canonical post, evaluate results, and return structured outputs.

##### **Steps**
1. **Deduplication**
   - Runs `deduplicate_dataframe()` using `DedupeConfig`.
   - Removes near-duplicate posts and retains only **canonical** representatives.
   - Logs deduplication metrics (clusters, removed duplicates).
   - Writes a deduplication report if `report_path` is configured.

2. **Resume Support**
   - If `resume.enabled` and a checkpoint exists, loads previously labeled posts to skip reprocessing.

3. **Classifier Setup**
   - Initializes the `Version2RuleEngine` (rule-based classifier).
   - Creates cache and rate-limiter instances from the configuration.
   - Instantiates `OpenAI` client and constructs active ensemble members (`rules`, `direct`, `reasoning`).

4. **Classification Loop**
   - Iterates through canonical posts in parallel batches.
   - For each post:
     - Combines title + body → classification text.
     - Calls `ensemble_classify()` to get predictions and metadata.
     - Handles validation, retries, and schema enforcement.
     - Records confidence, disagreement, and latency.
   - Failed or invalid rows are added to the **dead-letter queue**.

5. **Progress Tracking**
   - Uses `structured_log()` to emit progress updates and estimated time remaining.

6. **Split Assignment**
   - Calls `assign_splits()` to ensure splits respect cluster groupings.

7. **Evaluation and Drift**
   - Computes metrics via `evaluate_against_gold()` (if gold set available).
   - Compares results with previous runs using `compare_against_history()`.

8. **Summary Metrics**
   - Aggregates cache stats, ensemble agreement, disagreement rates, latency distribution (avg, P95), and duplicate stats.
   - Returns:
     - `canonical_df` → labeled canonical posts.
     - `id_mapping` → mapping from duplicates to canonical IDs.
     - `clusters` → dedupe clusters.
     - `summary_payload` → performance, evaluation, and ensemble metadata.


#### `run_pipeline(...) -> Dict[str, Any]`
Top-level **orchestrator** function called by `cli.py`.

**Responsibilities:**
1. Calls `classify_dataframe()` with all runtime components (cache, rate limiter, client, and config).
2. Normalizes the `RunConfig` dataclass using `_normalise_config()` for serialization.
3. Generates a JSON summary report via `generate_summary_report()`.
4. Returns all final outputs:
   - `canonical_df`
   - `id_mapping`
   - `clusters`
   - `summary_payload`
   - `config_payload`
   - `report_summary`

---

## 2.4 Rate Limiter: `utils.rate_limit.py`

### ⚙️ Overview
The `utils.rate_limit.py` module ensures the labeling pipeline does not exceed external API rate limits — especially for OpenAI model requests.  
It regulates how frequently classification calls are made, preventing overloads or HTTP 429 (“Too Many Requests”) errors.

This component is critical for maintaining **stability**, **compliance with API quotas**, and **fair resource distribution** when running in parallel.

### **Key Class**

#### `RateLimiter`
Implements a simple time-based rate-limiting mechanism.

##### **Core Behavior**
- Accepts a maximum number of requests per minute (from `ParallelConfig.rate_limit`).
- Before each API call, it ensures the minimum required delay between calls has passed.
- Uses internal timing logic (based on `time.time()` or threading locks) to coordinate access safely across multiple threads.

##### **Key Responsibilities**
1. **Throttle Requests**  
   Prevents the system from exceeding the configured requests-per-minute threshold.

2. **Sleep Between Calls**  
   If the next call occurs too soon, it pauses execution just long enough to stay within the limit.

3. **Integrate with Concurrency**  
   Works seamlessly with multithreaded classification (`parallel_process_batch`) so all workers share the same global rate budget.

4. **Provide Predictable Throughput**  
   Smooths out spikes in API usage, improving reliability for long labeling runs.

### **Configuration**
- Controlled by `ParallelConfig.rate_limit` (set in `RunConfig`).
- Defaults to **30 requests per minute**, adjustable via CLI (`--rate-limit`).
- Fully disabled if the rate limit is set to `0` or `None`.






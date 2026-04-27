# Sobol + Gradient Orchestration Plan (Current Spec)

## 1) Goal
Run a deterministic, restart-safe optimization pipeline where expensive backtests are reused from a single parquet cache whenever possible.

Core priorities:
- Avoid rerunning already-computed parameter points.
- Fill global space first (phase 1), then explore around strong seeds (phase 2), then run gradient descent in those areas.
- Keep one single parquet as source of truth.
- Periodically log best run by the required ranking.

## 2) Single Source of Truth
- Trials parquet: `sobol_gradient_trials.parquet`
- Only the main/orchestrator process writes to this parquet.
- Worker threads/processes run backtests and return results; they do not write files.

## 3) Deterministic Cache Key
Cache key includes:
- `symbol`
- `side`
- `start_date`
- `end_date`
- all sampled params (with int rounding for integer dimensions)

Rule:
- Same key means same stable backtest result.
- Existing matching params are always reusable, regardless of seed/radius used in a later run.

## 4) Ranking Rule (Authoritative)
Used for best-run logging, seed ranking, and final output sorting:
1. `trades` descending (`metric__total`)
2. `ITM` ascending (`metric__itm_expiries`)
3. `drawdown` descending (`metric__max_drawdown`, closer to zero is better; values are negative)
4. `avg_pnl` descending (`metric__avg_pnl`)

## 5) Phase 1: Global Sobol Fill
Inputs:
- `--sobol-samples`
- fixed/default seed (typically `--seed 42`)

Flow:
1. Generate Sobol points for the full search space.
2. For each point: check parquet cache key first.
3. If found: skip backtest.
4. If missing: dispatch worker to run backtest.
5. Main process appends new rows to the same parquet periodically.
6. Phase 1 completes once all `--sobol-samples` points are processed (executed or cache-hit).

## 6) Seed Eligibility and Selection
Seed eligibility:
- `metric__total > --min-seed-trades` (default threshold equivalent to “more than 3”)
- positive PnL (`metric__total_pnl > 0`)

Seed selection:
- Sort eligible rows by the authoritative ranking.
- Keep top `--seed-top-ratio` (default `0.30`).

## 7) Phase 2: Local Sobol Around Seeds
For each selected seed:
1. Estimate number of existing runs already inside seed neighborhood radius (`--local-radius`).
2. If count `>= --local-probe-per-seed`, skip this seed (enough local coverage already).
3. Otherwise generate local Sobol points around seed center (within radius in normalized space).
4. For each local point: cache lookup first, run backtest only if missing.
5. Main process periodically checkpoints parquet.

## 8) Gradient Descent Stage
After phase 2:
1. Run gradient descent for seed neighborhoods.
2. Every gradient evaluation must check parquet cache key first.
3. Run backtest only for missing points.
4. New results are appended by orchestrator to the same parquet.

## 9) Periodic Logging Requirement
At `--progress-seconds` cadence, print current best row by ranking with:
- `avg_pnl`
- `total_pnl`
- `number of trades`
- `itm`

Also print a final best summary at the end.

## 10) Checkpointing Requirement
At `--checkpoint-seconds` cadence:
- Orchestrator flushes accumulated new trial rows to parquet.
- Final forced checkpoint at each stage end and run end.

## 11) Final JSON Output
Output should contain runs from seed neighborhoods, sorted by the same ranking.

Requirements:
- collect runs in areas around selected seeds
- sort by authoritative ranking
- keep top `--final-top-n` (default `100`, configurable)
- each JSON row includes run parameters (and may include core metrics for readability)

## 12) Dimension + Sampling Rules
- Sobol samples are generated in `[0,1]^N`.
- Each dimension mapped to configured bounds.
- Integer dimensions use `round()` and clamp to bounds.
- Dimension bounds can come from YAML (preferred) and/or parquet-derived fallback.

## 13) Backtest Engine Contract
Workers invoke the fast precomputed-feature backtester (`--features-parquet`) and return:
- feasibility flag
- required metrics
- objective/scoring value (internal)

Backtest must not recompute features per trial; it reads precomputed data.

## 14) Non-Goals
- No backward compatibility requirement.
- No unit test updates required for this refactor.
- Legacy CLI knobs not used by this flow should be removed/simplified.

## 15) Acceptance Criteria
Pipeline is complete when:
1. Phase 1 processes all `--sobol-samples` with cache reuse.
2. Phase 2 explores selected seed neighborhoods with density skip and cache reuse.
3. Gradient stage runs with cache checks.
4. Single parquet remains the authoritative store.
5. Periodic best-metric logs appear as specified.
6. Final top-N JSON is generated with sorted runs and parameters.

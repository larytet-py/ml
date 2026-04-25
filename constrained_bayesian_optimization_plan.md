# Constrained Bayesian Optimization Plan for ETF Weekly Option Strategy

## Goal
Build a two-step process that:
1. Generates a large feature dataset once (disk-first, memory-safe) for symbols in `data/etfs.csv`.
2. Runs constrained Bayesian optimization with up to 3 configurable goals, using a small-function or class-based goal API.

The default optimization target order:
1. Minimize ITM expirations.
2. Minimize drawdown.
3. Maximize average PnL.

## Step 1: One-Time Feature Generation (Disk-First)

## Requirements
- Input universe must be restricted to symbols present in `data/etfs.csv`.
- If output CSV already exists, skip regeneration and use it in Step 2.
- Data is large, so processing must be chunked/streamed and append to CSV.
- Avoid in-memory full matrix joins across all symbols and all dates.

## Output
- Primary output file: `data/option_strategy_features.csv`
- Schema file: `data/option_strategy_features.schema.json`
- Metadata file: `data/option_strategy_features.meta.json` (row count, symbol count, date range, feature version, lookbacks used)

## Feature List (Complete)
All features are calculated per `(symbol, date)` row unless stated otherwise.

| Group | Feature Name | Description |
|---|---|---|
| Identity | `symbol` | Ticker |
| Identity | `date` | Trading date |
| Identity | `side` | `put` or `call` context row (optional if creating side-specific rows) |
| Raw price | `open` `high` `low` `close` | Raw OHLC |
| Raw volume | `volume` | Raw volume |
| Return | `ret_1d_close` | `close.pct_change(1)` |
| Return | `ret_1d_open` | `open.pct_change(1)` |
| ROC | `roc_close_L` | ROC on `close` for each configured `L` |
| ROC | `roc_open_L` | ROC on `open` for each configured `L` |
| Acceleration | `accel_close_L_A` | `(close.pct_change(L) - close.pct_change(L).shift(A))` |
| Acceleration | `accel_open_L_A` | `(open.pct_change(L) - open.pct_change(L).shift(A))` |
| Volatility | `realized_vol_close_W` | Rolling std of close returns * sqrt(252), window `W` |
| Volatility | `realized_vol_open_W` | Rolling std of open returns * sqrt(252), window `W` |
| Volatility | `downside_vol_W` | Downside rolling std (annualized), window `W` |
| Volatility | `upside_vol_W` | Upside rolling std (annualized), window `W` |
| Price trend | `price_momentum_L` | `open.pct_change(L)` or `close.pct_change(L)` (configurable) |
| Price trend | `price_stddev_W` | Rolling std of price level |
| Volume lag | `volume_lag1` | `volume.shift(1)` for entry-time safety |
| Volume trend | `volume_momentum_L` | `(volume.shift(1) / volume.shift(1+L)) - 1` |
| Volume trend | `volume_roc_L` | Same as above with ROC lookback |
| Volume vol | `volume_stddev_W` | Rolling std of lagged volume |
| Intraday shape | `high_open_over_open` | `(high - open) / open` |
| Intraday shape | `open_low_over_open` | `(open - low) / open` |
| Intraday shape | `high_close_over_close` | `(high - close) / close` |
| Intraday shape | `close_low_over_low` | `(close - low) / low` |
| Intraday shape | `high_low_over_high` | `(high - low) / high` |
| Intraday shape | `high_low_over_low` | `(high - low) / low` |
| Cross-equity corr | `corr_to_<peer>_W` | Rolling return correlation to each peer symbol in universe, window `W` |
| Cross-equity summary | `corr_mean_W` | Mean correlation across peers |
| Cross-equity summary | `corr_std_W` | Std correlation across peers |
| Cross-equity summary | `corr_min_W` | Minimum peer correlation |
| Cross-equity summary | `corr_max_W` | Maximum peer correlation |
| Cross-equity summary | `corr_pos_count_W` | Count of peers with correlation > threshold |
| Cross-equity summary | `corr_neg_count_W` | Count of peers with correlation < -threshold |
| Neighbor aggregates | `nbr_<f>_mean` | Mean of feature `f` across correlation-selected neighbors |
| Neighbor aggregates | `nbr_<f>_std` | Std of feature `f` across neighbors |
| Neighbor deltas | `delta_<f>_vs_nbr_mean` | Symbol feature minus neighbor mean |
| Neighbor meta | `nbr_count` | Number of neighbors used on that date |

`L`, `A`, and `W` are lists from config, for example:
- ROC lookbacks: `[1, 3, 5, 10, 20]`
- Acceleration windows: `[1, 3, 5]`
- Vol windows: `[5, 10, 21, 42]`
- Corr windows: `[10, 21, 42]`

## Step 1 Implementation Tasks
1. Add a new script: `build_option_strategy_features.py`.
2. Add symbol sanitization for `data/etfs.csv` (drop duplicated header rows like `symbol`).
3. Add universe extraction from `data/etfs.csv` and filter all data to this set.
4. Add chunk-by-symbol processing and append writes to `data/option_strategy_features.csv`.
5. Add optional partition temp files in `data/_tmp_features/` and final concatenation pass if needed.
6. Add idempotent behavior:
7. If `data/option_strategy_features.csv` exists and `--force-rebuild` is not passed, skip compute.
8. Write schema + metadata JSON files.

## Step 1 CLI Contract
Proposed command:

```bash
python3 build_option_strategy_features.py \
  --input-csv data/etfs.csv \
  --output-csv data/option_strategy_features.csv \
  --feature-config option_signal_notifier.config \
  --workers 8
```

Behavior:
- Default: reuse existing output CSV.
- Rebuild mode: `--force-rebuild`.

## Step 2: Constrained Bayesian Optimization

## Requirements
- Load features from `data/option_strategy_features.csv`.
- Allow up to 3 goals configurable from config/CLI.
- Make each goal easy to add as a small function.
- If BO library needs richer interface, provide class wrappers exposing required API.

## Proposed Architecture

### New files
- `optimization/goals.py`
- `optimization/goal_registry.py`
- `optimization/constrained_bo.py`
- `optimize_option_strategy_region.py`

### Goal API (small function level)
Each goal starts as a function:

```python
def goal_itm(metrics: dict) -> float: ...
def goal_drawdown(metrics: dict) -> float: ...
def goal_avg_pnl(metrics: dict) -> float: ...
```

### Goal API (class wrapper for BO integration)
If constraints/objectives/probability-of-feasibility need extra methods:

```python
class Goal:
    name: str
    direction: str  # "min" or "max"
    kind: str       # "constraint" or "objective"

    def value(self, metrics: dict) -> float: ...
    def is_satisfied(self, metrics: dict) -> bool: ...
    def distance_to_target(self, metrics: dict) -> float: ...
```

`Goal` objects are thin wrappers around small functions so adding goals remains easy.

## BO Runner Design
1. Ask strategy backtest for metrics for a candidate parameter vector.
2. Evaluate all configured goals via goal registry.
3. Return:
4. objective value(s),
5. constraint violation values,
6. auxiliary diagnostics for logging.
7. BO proposes next candidate using surrogate + acquisition with constraints.
8. Persist every trial to CSV (`data/bo_trials.csv`) for restart/reproducibility.

## Config Model (up to 3 goals)
Example config section:

```yaml
optimization:
  goals:
    - name: itm_expiries
      kind: constraint
      direction: min
      target: 0
    - name: max_drawdown_abs
      kind: constraint
      direction: min
      target: 5000
    - name: avg_pnl
      kind: objective
      direction: max
      target: null
```

## Goal Ordering and Behavior
- Primary behavior should emulate lexicographic preference.
- Recommended implementation:
1. Use ITM and drawdown as hard constraints.
2. Use avg PnL as objective inside feasible region.
3. If no feasible region exists early, allow soft penalties until first feasible points are found.

## Region Discovery (not a single best vector)
To support robust N-dimensional connected regions:
1. Keep all feasible, high-quality trials from BO history.
2. Normalize parameter dimensions.
3. Build kNN graph among feasible points.
4. Extract connected components.
5. Pick largest component passing uniformity threshold (low variance in key metrics).
6. Export region bounds to `data/bo_region_summary.json`.

## Integration with Existing Code
1. Reuse backtest computation from `backtest_weekly_option_reversal.py`.
2. Extract metrics computation to a reusable function if needed.
3. Keep current optimizer as fallback mode (`--optimizer gradient|bayes`).
4. Add flags for selected goals and goal targets.

## Testing Plan
1. Unit test feature generation for one symbol and one date range.
2. Unit test CSV reuse logic (existing file should skip rebuild).
3. Unit test goal registry loads up to 3 goals from config.
4. Unit test each goal function and wrapper class methods.
5. Integration test constrained BO with small trial budget on tiny dataset.
6. Integration test region extraction from synthetic trial points.

## Deliverables Checklist
- `build_option_strategy_features.py` with disk-first generation and CSV reuse logic.
- Feature schema + metadata outputs.
- Goal function module and class wrapper API.
- Constrained BO runner with pluggable goals (max 3).
- Trial logging and restart support.
- Region extraction output for robust connected parameter zones.
- Tests for feature pipeline, goals, BO loop, and region extraction.

